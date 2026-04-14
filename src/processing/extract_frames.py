"""
extract_frames.py

Extracts individual JPEG frames from every .mp4 clip in RAW_CLIPS_DIR and
writes them to FRAMES_DIR, preserving the player/game/event folder hierarchy.

Output layout:
  data/frames/{player_slug}/{game_id}_{event_id}/frame_{n:04d}.jpg

Only every Nth frame is kept (default: every 5th) to reduce redundancy while
still capturing movement across a short clip.  Clips that already have a
frames sub-directory are skipped on re-runs.

Dependencies: opencv-python (cv2)

Usage:
  python -m src.processing.extract_frames
  python -m src.processing.extract_frames --every 3
"""

import argparse
import logging
from pathlib import Path

import cv2

from config import FRAMES_DIR, RAW_CLIPS_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [extract_frames] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_EVERY: int = 5   # keep 1 in every N frames


def extract_frames_from_clip(
    clip_path: Path,
    out_dir: Path,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
) -> int:
    """
    Extract sampled frames from a single mp4 clip.

    Parameters
    ----------
    clip_path    : path to source .mp4 file
    out_dir      : directory where frame JPEGs will be written
    sample_every : keep 1 frame every *sample_every* frames

    Returns
    -------
    Number of frames written.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        log.warning("Cannot open clip: %s", clip_path)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx    = 0   # absolute frame counter
    written      = 0   # frames actually saved

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            dest = out_dir / f"frame_{written:04d}.jpg"
            cv2.imwrite(str(dest), frame)
            written += 1

        frame_idx += 1

    cap.release()
    return written


def extract_all_clips(
    clips_root: str | None = None,
    frames_root: str | None = None,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
) -> None:
    """
    Walk *clips_root* for all .mp4 files, extract frames into *frames_root*,
    and print a summary when complete.

    Clips whose output directory already exists (non-empty) are skipped.

    Parameters
    ----------
    clips_root   : root of raw clips tree (default: RAW_CLIPS_DIR)
    frames_root  : root of frames output tree (default: FRAMES_DIR)
    sample_every : keep 1 in every N frames per clip
    """
    clips_root  = Path(clips_root  or RAW_CLIPS_DIR)
    frames_root = Path(frames_root or FRAMES_DIR)

    clips = sorted(clips_root.rglob("*.mp4"))
    if not clips:
        log.warning("No .mp4 files found under %s", clips_root)
        return

    log.info("Found %d clips under %s", len(clips), clips_root)

    total_clips_processed = 0
    total_clips_skipped   = 0
    total_frames_written  = 0

    for clip_path in clips:
        # Derive relative path from clips root: {player_slug}/{stem}
        rel        = clip_path.relative_to(clips_root)          # e.g. stephen_curry/0022400123_45.mp4
        player_dir = rel.parts[0]                               # e.g. stephen_curry
        clip_stem  = clip_path.stem                             # e.g. 0022400123_45

        out_dir = frames_root / player_dir / clip_stem

        # Skip if already extracted (directory exists and is non-empty)
        if out_dir.exists() and any(out_dir.iterdir()):
            total_clips_skipped += 1
            log.debug("  [skip] %s (already extracted)", rel)
            continue

        n = extract_frames_from_clip(clip_path, out_dir, sample_every)

        if n > 0:
            log.info("  [+] %s → %d frames", rel, n)
            total_clips_processed += 1
            total_frames_written  += n
        else:
            log.warning("  [!] %s → 0 frames (skipping)", rel)

    log.info(
        "Summary: processed=%d  skipped=%d  total_frames_written=%d",
        total_clips_processed,
        total_clips_skipped,
        total_frames_written,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from NBA play clips.")
    p.add_argument(
        "--every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        metavar="N",
        help=f"Keep 1 in every N frames (default: {DEFAULT_SAMPLE_EVERY})",
    )
    p.add_argument("--clips-dir",  default=None, help="Override RAW_CLIPS_DIR")
    p.add_argument("--frames-dir", default=None, help="Override FRAMES_DIR")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_all_clips(
        clips_root=args.clips_dir,
        frames_root=args.frames_dir,
        sample_every=args.every,
    )
