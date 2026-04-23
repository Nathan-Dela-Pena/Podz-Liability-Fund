"""
extract_pose.py

Two-stage pose extraction pipeline:
  1. YOLOv8n detects all persons in each frame (bounding boxes)
  2. The correct player is identified by jersey color, falling back to the
     largest detected person if color matching is ambiguous
  3. MediaPipe Pose runs only on the selected player crop

This replaces the original single-person approach which blindly used whatever
MediaPipe picked — often a referee or defender.

Player identification
---------------------
Each player is associated with a team (Warriors or Spurs).  We detect the
dominant jersey color in each bounding-box crop and pick the person whose
jersey best matches the target team's colors:

  Warriors home : royal blue  (#1D428A) — HSV hue 105–125
  Warriors away : white/gold
  Spurs home    : black
  Spurs away    : white

If no crop passes the color threshold (e.g. intro/overlay frames), we fall
back to the largest bounding box — same heuristic MediaPipe uses implicitly.

Output schema (per player):
  {POSE_DIR}/{player_slug}.csv

Columns: game_id, event_id, lm_{i}_{dim}_mean, lm_{i}_{dim}_std (264 cols),
         vertical_accel, stride_length, shoulder_droop

Dependencies: mediapipe, opencv-python, numpy, pandas, ultralytics (YOLOv8)

Usage:
  python -m src.processing.extract_pose
"""

import logging
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import FRAMES_DIR, PLAYERS, POSE_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [extract_pose] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress ultralytics verbose output
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Team jersey color profiles (HSV ranges)
# ---------------------------------------------------------------------------

# Each entry: (h_lo, h_hi, s_lo, v_lo) — tuned for NBA jerseys under arena lighting
_JERSEY_PROFILES: dict[str, list[tuple]] = {
    # Warriors royal blue home
    "warriors_blue":  [(105, 125, 120, 60)],
    # Warriors gold (home numbers / away primary)
    "warriors_gold":  [(22, 38, 140, 140)],
    # Spurs black (home)
    "spurs_black":    [(0, 180, 0, 0)],     # any hue, low sat+val
    # Spurs white (away)
    "spurs_white":    [(0, 180, 0, 180)],   # any hue, low sat, high val
}

# Map player name → list of color profile keys to match (home + away)
def _team_profiles(player_name: str) -> list[str]:
    warriors = {
        "Stephen Curry", "Draymond Green", "Jonathan Kuminga", "Moses Moody",
        "Gary Payton II", "Kevon Looney", "Brandin Podziemski", "Gui Santos",
        "Quinten Post", "Will Richard", "Pat Spencer", "Jimmy Butler",
        "Seth Curry", "Al Horford", "De'Anthony Melton", "Kristaps Porzingis",
        "Andrew Wiggins",
    }
    spurs = {
        "Victor Wembanyama", "Keldon Johnson", "Stephon Castle", "Dylan Harper",
        "De'Aaron Fox",
    }
    if player_name in warriors:
        return ["warriors_blue", "warriors_gold"]
    if player_name in spurs:
        return ["spurs_black", "spurs_white"]
    return []   # fallback: use largest person


def _jersey_score(crop_bgr: np.ndarray, profile_keys: list[str]) -> float:
    """
    Returns the fraction of pixels in *crop_bgr* that match any of the
    jersey color profiles.  Focuses on the torso (middle vertical third).
    """
    h, w = crop_bgr.shape[:2]
    torso = crop_bgr[h // 3: 2 * h // 3, w // 5: 4 * w // 5]
    if torso.size == 0:
        return 0.0

    hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    total = torso.shape[0] * torso.shape[1]
    mask  = np.zeros((torso.shape[0], torso.shape[1]), dtype=np.uint8)

    for key in profile_keys:
        for (h_lo, h_hi, s_lo, v_lo) in _JERSEY_PROFILES[key]:
            m = cv2.inRange(hsv,
                            np.array([h_lo, s_lo, v_lo]),
                            np.array([h_hi, 255,  255]))
            mask = cv2.bitwise_or(mask, m)

    return float(mask.sum()) / (255.0 * total)


# ---------------------------------------------------------------------------
# MediaPipe constants
# ---------------------------------------------------------------------------

_LM_LEFT_SHOULDER  = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_HIP       = 23
_LM_RIGHT_HIP      = 24
_LM_LEFT_ANKLE     = 27
_LM_RIGHT_ANKLE    = 28

N_LANDMARKS = 33
N_DIMS      = 4
N_LM_FEATS  = N_LANDMARKS * N_DIMS   # 132


def _lm_col_names() -> list[str]:
    dims = ["x", "y", "z", "vis"]
    return [f"lm_{i}_{d}" for i in range(N_LANDMARKS) for d in dims]


LM_COL_NAMES = _lm_col_names()


# ---------------------------------------------------------------------------
# YOLO — lazy singleton
# ---------------------------------------------------------------------------

_yolo_model: YOLO | None = None


def _get_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")   # downloads ~6MB on first run
    return _yolo_model


# ---------------------------------------------------------------------------
# Frame-level: detect target player crop
# ---------------------------------------------------------------------------

def _select_player_crop(
    image_bgr: np.ndarray,
    profile_keys: list[str],
    conf_threshold: float = 0.35,
    jersey_threshold: float = 0.10,
) -> np.ndarray | None:
    """
    Run YOLO on *image_bgr*, find all persons, and return the crop of the
    player whose jersey best matches *profile_keys*.

    Falls back to the largest bounding box if no crop meets the jersey
    color threshold.

    Returns the selected crop (BGR ndarray) or None if no persons detected.
    """
    yolo = _get_yolo()
    results = yolo(image_bgr, classes=[0], conf=conf_threshold, verbose=False)

    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Clamp to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image_bgr.shape[1], x2)
            y2 = min(image_bgr.shape[0], y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    if not boxes:
        return None

    # Score each box by jersey color match
    best_crop  = None
    best_score = -1.0

    for (x1, y1, x2, y2) in boxes:
        crop  = image_bgr[y1:y2, x1:x2]
        score = _jersey_score(crop, profile_keys) if profile_keys else 0.0
        area  = (x2 - x1) * (y2 - y1)

        # Use jersey score if it clears the threshold; else use area as tiebreak
        effective_score = score if score >= jersey_threshold else area / 1e6

        if effective_score > best_score:
            best_score = effective_score
            best_crop  = crop

    return best_crop


# ---------------------------------------------------------------------------
# Frame-level: landmarks from crop
# ---------------------------------------------------------------------------

def _extract_landmarks_from_frame(
    image_bgr: np.ndarray,
    pose: mp.solutions.pose.Pose,
    profile_keys: list[str],
) -> np.ndarray | None:
    """
    Select the target player crop via YOLO, then run MediaPipe Pose on it.
    Returns a (132,) float32 vector or None.
    """
    crop = _select_player_crop(image_bgr, profile_keys)
    if crop is None:
        # Fallback: run MediaPipe on the full frame (original behaviour)
        crop = image_bgr

    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results   = pose.process(image_rgb)

    if results.pose_landmarks is None:
        return None

    lms = results.pose_landmarks.landmark
    vec = np.empty(N_LM_FEATS, dtype=np.float32)
    for i, lm in enumerate(lms):
        base = i * N_DIMS
        vec[base]     = lm.x
        vec[base + 1] = lm.y
        vec[base + 2] = lm.z
        vec[base + 3] = lm.visibility
    return vec


# ---------------------------------------------------------------------------
# Derived biomechanical signals
# ---------------------------------------------------------------------------

def _hip_y_series(m: np.ndarray) -> np.ndarray:
    return (m[:, _LM_LEFT_HIP * N_DIMS + 1] + m[:, _LM_RIGHT_HIP * N_DIMS + 1]) / 2.0

def _ankle_distance_series(m: np.ndarray) -> np.ndarray:
    la_x = m[:, _LM_LEFT_ANKLE  * N_DIMS];     la_y = m[:, _LM_LEFT_ANKLE  * N_DIMS + 1]
    ra_x = m[:, _LM_RIGHT_ANKLE * N_DIMS];     ra_y = m[:, _LM_RIGHT_ANKLE * N_DIMS + 1]
    return np.sqrt((la_x - ra_x) ** 2 + (la_y - ra_y) ** 2)

def _shoulder_y_series(m: np.ndarray) -> np.ndarray:
    return (m[:, _LM_LEFT_SHOULDER * N_DIMS + 1] + m[:, _LM_RIGHT_SHOULDER * N_DIMS + 1]) / 2.0

def _compute_derived_signals(lm_matrix: np.ndarray) -> tuple[float, float, float]:
    hip_y      = _hip_y_series(lm_matrix)
    ankle_dist = _ankle_distance_series(lm_matrix)
    shoulder_y = _shoulder_y_series(lm_matrix)
    vertical_accel = float(np.mean(np.abs(np.diff(hip_y)))) if len(hip_y) >= 2 else 0.0
    stride_length  = float(np.var(ankle_dist))
    shoulder_droop = float(shoulder_y[-1] - shoulder_y[0]) if len(shoulder_y) >= 2 else 0.0
    return vertical_accel, stride_length, shoulder_droop


# ---------------------------------------------------------------------------
# Clip-level aggregation
# ---------------------------------------------------------------------------

def _process_clip(
    clip_frame_dir: Path,
    pose: mp.solutions.pose.Pose,
    profile_keys: list[str],
) -> dict | None:
    parts = clip_frame_dir.name.split("_", 1)
    if len(parts) != 2:
        return None
    game_id, event_id = parts[0], parts[1]

    frame_paths = sorted(clip_frame_dir.glob("*.jpg"))
    if not frame_paths:
        return None

    vectors = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        vec = _extract_landmarks_from_frame(img, pose, profile_keys)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        return None

    lm_matrix = np.stack(vectors, axis=0)
    mean_lm   = lm_matrix.mean(axis=0)
    std_lm    = lm_matrix.std(axis=0)
    vertical_accel, stride_length, shoulder_droop = _compute_derived_signals(lm_matrix)

    row: dict = {"game_id": game_id, "event_id": event_id}
    for idx, col in enumerate(LM_COL_NAMES):
        row[f"{col}_mean"] = float(mean_lm[idx])
        row[f"{col}_std"]  = float(std_lm[idx])
    row["vertical_accel"]  = vertical_accel
    row["stride_length"]   = stride_length
    row["shoulder_droop"]  = shoulder_droop
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_pose_features(
    frames_root: str | None = None,
    pose_root:   str | None = None,
) -> None:
    frames_root = Path(frames_root or FRAMES_DIR)
    pose_root   = Path(pose_root   or POSE_DIR)
    pose_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        log.error("Frames root does not exist: %s", frames_root)
        return

    player_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    if not player_dirs:
        log.warning("No player directories found under %s", frames_root)
        return

    log.info("Processing %d player(s) with YOLO+MediaPipe.", len(player_dirs))

    # Build slug → player name map for jersey color lookup
    from config import PLAYERS as ALL_PLAYERS
    slug_to_name = {
        n.lower().replace(" ", "_").replace("'", "").replace(".", ""): n
        for n in ALL_PLAYERS
    }

    mp_pose = mp.solutions.pose

    for player_dir in player_dirs:
        player_slug = player_dir.name
        csv_path    = pose_root / f"{player_slug}.csv"

        if csv_path.exists():
            log.info("[skip] %s — CSV already exists.", player_slug)
            continue

        clip_dirs = sorted([d for d in player_dir.iterdir() if d.is_dir()])
        if not clip_dirs:
            log.info("[skip] %s — no clip directories.", player_slug)
            continue

        player_name  = slug_to_name.get(player_slug, "")
        profile_keys = _team_profiles(player_name)
        log.info("=== %s (%d clips, profiles=%s) ===",
                 player_slug, len(clip_dirs), profile_keys or ["largest-person"])

        rows = []
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            for clip_dir in clip_dirs:
                row = _process_clip(clip_dir, pose, profile_keys)
                if row is not None:
                    rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            log.info("  Saved %d rows → %s", len(df), csv_path)
        else:
            log.warning("  No valid clips for %s — CSV not written.", player_slug)

    log.info("extract_pose done.")


if __name__ == "__main__":
    extract_pose_features()
