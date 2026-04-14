"""
extract_pose.py

Runs MediaPipe Pose on every extracted frame in FRAMES_DIR and produces
per-player CSV files in POSE_DIR with one row per play clip.

Output schema (per player):
  data/pose/{player_slug}.csv

Columns
-------
  game_id          : str
  event_id         : str
  lm_{i}_{dim}_mean: float  — mean across frames for landmark i, dimension dim
  lm_{i}_{dim}_std : float  — std  across frames for landmark i, dimension dim
      i in 0..32, dim in {x, y, z, vis}  → 33 * 4 * 2 = 264 columns
  vertical_accel   : float  — mean absolute frame-to-frame change in hip-y
  stride_length    : float  — variance in ankle-distance across frames
  shoulder_droop   : float  — total drift (last − first) of shoulder-midpoint y

The 132-dim raw landmark vector (33 × 4) is summarised as mean + std over all
frames in the clip, giving 264 landmark columns.  Three derived biomechanical
signals are appended (total: 264 + 3 = 267 columns, excluding game/event IDs).

NOTE: The project proposal specifies a 135-dim pose vector for the CNN pose
head.  That head receives [mean_landmarks(132) + derived_signals(3)], which is
the mean-only summary (132) concatenated with the 3 derived signals → 135 dims.
The full CSV stores both mean and std for richer downstream analysis.

Dependencies: mediapipe, opencv-python, numpy, pandas

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

from config import FRAMES_DIR, POSE_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [extract_pose] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmarks of interest
# ---------------------------------------------------------------------------

# MediaPipe Pose landmark indices (from mp.solutions.pose.PoseLandmark)
_LM_LEFT_SHOULDER  = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_HIP       = 23
_LM_RIGHT_HIP      = 24
_LM_LEFT_ANKLE     = 27
_LM_RIGHT_ANKLE    = 28

N_LANDMARKS = 33
N_DIMS      = 4    # x, y, z, visibility
N_LM_FEATS  = N_LANDMARKS * N_DIMS   # 132

# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def _lm_col_names() -> list[str]:
    """
    Returns the list of landmark column names in the CSV, interleaved as:
      lm_0_x, lm_0_y, lm_0_z, lm_0_vis, lm_1_x, …
    """
    dims = ["x", "y", "z", "vis"]
    names = []
    for i in range(N_LANDMARKS):
        for d in dims:
            names.append(f"lm_{i}_{d}")
    return names


LM_COL_NAMES = _lm_col_names()   # 132 names


# ---------------------------------------------------------------------------
# Frame-level pose extraction
# ---------------------------------------------------------------------------

def _extract_landmarks_from_frame(
    image_bgr: np.ndarray,
    pose: mp.solutions.pose.Pose,
) -> np.ndarray | None:
    """
    Run MediaPipe Pose on one frame.

    Returns a (132,) float32 array [x0,y0,z0,v0, x1,y1,z1,v1, …] or None if
    no pose was detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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

def _hip_y_series(landmark_matrix: np.ndarray) -> np.ndarray:
    """
    landmark_matrix: (T, 132) — T frames

    Returns (T,) array of hip midpoint y coordinate.
    Hip midpoint = mean of left-hip-y and right-hip-y.
    """
    left_hip_y  = landmark_matrix[:, _LM_LEFT_HIP  * N_DIMS + 1]
    right_hip_y = landmark_matrix[:, _LM_RIGHT_HIP * N_DIMS + 1]
    return (left_hip_y + right_hip_y) / 2.0


def _ankle_distance_series(landmark_matrix: np.ndarray) -> np.ndarray:
    """
    Returns (T,) array of Euclidean distance between ankle x-coords (2-D proxy
    for stride length — z not reliable from monocular video).
    """
    la_x = landmark_matrix[:, _LM_LEFT_ANKLE  * N_DIMS]
    ra_x = landmark_matrix[:, _LM_RIGHT_ANKLE * N_DIMS]
    la_y = landmark_matrix[:, _LM_LEFT_ANKLE  * N_DIMS + 1]
    ra_y = landmark_matrix[:, _LM_RIGHT_ANKLE * N_DIMS + 1]
    return np.sqrt((la_x - ra_x) ** 2 + (la_y - ra_y) ** 2)


def _shoulder_y_series(landmark_matrix: np.ndarray) -> np.ndarray:
    """Returns (T,) array of shoulder midpoint y coordinate."""
    ls_y = landmark_matrix[:, _LM_LEFT_SHOULDER  * N_DIMS + 1]
    rs_y = landmark_matrix[:, _LM_RIGHT_SHOULDER * N_DIMS + 1]
    return (ls_y + rs_y) / 2.0


def _compute_derived_signals(landmark_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Compute three scalar biomechanical signals for a clip from its (T, 132)
    landmark matrix.

    Returns
    -------
    vertical_accel : mean absolute frame-to-frame change in hip-y
    stride_length  : variance of ankle-distance across frames
    shoulder_droop : total drift of shoulder-midpoint y (last − first frame)
    """
    hip_y      = _hip_y_series(landmark_matrix)
    ankle_dist = _ankle_distance_series(landmark_matrix)
    shoulder_y = _shoulder_y_series(landmark_matrix)

    if len(hip_y) < 2:
        vertical_accel = 0.0
    else:
        vertical_accel = float(np.mean(np.abs(np.diff(hip_y))))

    stride_length  = float(np.var(ankle_dist))
    shoulder_droop = float(shoulder_y[-1] - shoulder_y[0]) if len(shoulder_y) >= 2 else 0.0

    return vertical_accel, stride_length, shoulder_droop


# ---------------------------------------------------------------------------
# Clip-level aggregation
# ---------------------------------------------------------------------------

def _process_clip(
    clip_frame_dir: Path,
    pose: mp.solutions.pose.Pose,
) -> dict | None:
    """
    Process all frames in *clip_frame_dir* and return a feature dict or None
    if no usable frames are found.

    The frame directory is named  {game_id}_{event_id}.
    """
    parts = clip_frame_dir.name.split("_", 1)
    if len(parts) != 2:
        log.warning("Unexpected clip dir name: %s", clip_frame_dir.name)
        return None
    game_id, event_id = parts[0], parts[1]

    frame_paths = sorted(clip_frame_dir.glob("*.jpg"))
    if not frame_paths:
        log.debug("  No frames in %s", clip_frame_dir)
        return None

    vectors = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        vec = _extract_landmarks_from_frame(img, pose)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        log.debug("  No poses detected in %s", clip_frame_dir.name)
        return None

    lm_matrix = np.stack(vectors, axis=0)   # (T, 132)

    mean_lm = lm_matrix.mean(axis=0)        # (132,)
    std_lm  = lm_matrix.std(axis=0)         # (132,)

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
    """
    Walk *frames_root*, process every clip directory, and write one CSV per
    player to *pose_root*.

    Skips player CSVs that already exist (re-run safety — delete the CSV to
    reprocess a player).

    Parameters
    ----------
    frames_root : root of extracted frames tree (default: FRAMES_DIR)
    pose_root   : output directory for pose CSVs (default: POSE_DIR)
    """
    frames_root = Path(frames_root or FRAMES_DIR)
    pose_root   = Path(pose_root   or POSE_DIR)
    pose_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        log.error("Frames root does not exist: %s", frames_root)
        return

    # Each immediate sub-directory of frames_root is a player slug
    player_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    if not player_dirs:
        log.warning("No player directories found under %s", frames_root)
        return

    log.info("Processing %d player(s).", len(player_dirs))

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

        log.info("=== %s (%d clips) ===", player_slug, len(clip_dirs))

        rows = []
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            for clip_dir in clip_dirs:
                row = _process_clip(clip_dir, pose)
                if row is not None:
                    rows.append(row)
                    log.debug("  [+] %s (%d landmarks)", clip_dir.name, N_LM_FEATS)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            log.info("  Saved %d rows → %s", len(df), csv_path)
        else:
            log.warning("  No valid clips for %s — CSV not written.", player_slug)

    log.info("extract_pose done.")


if __name__ == "__main__":
    extract_pose_features()
