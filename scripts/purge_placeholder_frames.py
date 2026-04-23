"""
purge_placeholder_frames.py

Scans FRAMES_DIR for NBA 'Video Not Available' placeholder frames and deletes
them.  Also removes any clip directories that are left empty after purging.

These placeholders have ~82% NBA blue coverage and are useless for training.

Usage:
  python scripts/purge_placeholder_frames.py           # dry-run (prints counts)
  python scripts/purge_placeholder_frames.py --delete  # actually delete
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from config import FRAMES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [purge] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BLUE_THRESHOLD = 0.5


def is_placeholder(path: Path) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    blue_pct = blue_mask.sum() / (255.0 * img.shape[0] * img.shape[1])
    return blue_pct > BLUE_THRESHOLD


def purge(delete: bool = False) -> None:
    frames_root = Path(FRAMES_DIR)
    jpgs = sorted(frames_root.rglob("*.jpg"))
    log.info("Scanning %d frames under %s", len(jpgs), frames_root)

    placeholder_count = 0
    real_count = 0

    for jpg in jpgs:
        if is_placeholder(jpg):
            placeholder_count += 1
            if delete:
                jpg.unlink()
        else:
            real_count += 1

    log.info("Placeholder frames: %d", placeholder_count)
    log.info("Real frames:        %d", real_count)

    if delete:
        log.info("Deleted %d placeholder frames.", placeholder_count)
        # Remove empty clip directories
        empty = 0
        for d in sorted(frames_root.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                empty += 1
        log.info("Removed %d empty directories.", empty)
    else:
        log.info("Dry run — pass --delete to actually remove files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete bad frames")
    args = parser.parse_args()
    purge(delete=args.delete)
