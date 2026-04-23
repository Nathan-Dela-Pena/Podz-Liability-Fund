"""
run_cnn_pipeline.py

Combined pipeline that:
  1. Fetches YouTube NBA highlight videos via yt-dlp (replaces blocked NBA API)
  2. Extracts sampled JPEG frames, skipping placeholder screens
  3. Runs MediaPipe Pose on frames → data/pose/*.csv
  4. Trains the CNN branch → checkpoints/cnn_best.pt

Storage strategy
----------------
Videos are downloaded to a temp directory and deleted immediately after frame
extraction.  Only frames are kept on Google Drive (~2-5 GB vs ~100+ GB for video).

Why YouTube instead of stats.nba.com
-------------------------------------
The NBA CDN (videos.nba.com) blocks programmatic downloads — every clip returns
a 'Video Not Available' placeholder screen.  YouTube highlight reels provide
the same player footage needed for pose analysis without API restrictions.

Usage:
  python scripts/run_cnn_pipeline.py               # full pipeline
  python scripts/run_cnn_pipeline.py --skip-fetch  # frames already in Drive
  python scripts/run_cnn_pipeline.py --skip-pose   # pose CSVs already exist
  python scripts/run_cnn_pipeline.py --skip-train  # skip CNN training
"""

import argparse
import logging
from pathlib import Path

from config import FRAMES_DIR, POSE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cnn_pipeline] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1+2: Fetch YouTube highlights → extract frames
# ---------------------------------------------------------------------------

def fetch_and_extract(
    players: list[str] | None = None,
    seasons: list[str] | None = None,
    videos_per_query: int = 3,
    sample_every: int = 30,
) -> None:
    from src.ingestion.fetch_youtube_clips import fetch_youtube_frames
    log.info("=== Fetching YouTube highlights and extracting frames ===")
    fetch_youtube_frames(
        players=players,
        seasons=seasons,
        videos_per_query=videos_per_query,
        sample_every=sample_every,
    )


# ---------------------------------------------------------------------------
# Step 3: Pose extraction
# ---------------------------------------------------------------------------

def run_pose() -> None:
    from src.processing.extract_pose import extract_pose_features
    log.info("=== Running MediaPipe Pose extraction ===")
    extract_pose_features()


# ---------------------------------------------------------------------------
# Step 4: Train CNN
# ---------------------------------------------------------------------------

def run_cnn_training() -> None:
    from src.models.cnn import train_cnn
    log.info("=== Training CNN branch ===")
    train_cnn()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN data pipeline + training")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip YouTube fetch+extract (frames already in Drive)")
    parser.add_argument("--skip-pose",  action="store_true",
                        help="Skip pose extraction (CSVs already in Drive)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip CNN training")
    parser.add_argument("--players", nargs="+", default=None,
                        help="Subset of players (default: all)")
    parser.add_argument("--seasons", nargs="+", default=["2025-26"],
                        help="Seasons to fetch highlights for")
    parser.add_argument("--videos-per-query", type=int, default=3,
                        help="YouTube results to download per search query (default: 3)")
    parser.add_argument("--every", type=int, default=30,
                        help="Sample 1 frame every N frames (default: 30)")
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch_and_extract(
            players=args.players,
            seasons=args.seasons,
            videos_per_query=args.videos_per_query,
            sample_every=args.every,
        )
    else:
        log.info("Skipping fetch (--skip-fetch)")

    frames_root = Path(FRAMES_DIR)
    has_frames  = frames_root.exists() and any(frames_root.rglob("*.jpg"))

    if not has_frames:
        log.warning(
            "No frames found in %s — skipping pose and CNN training.\n"
            "  Run without --skip-fetch to download YouTube highlights first.",
            FRAMES_DIR,
        )
    else:
        if not args.skip_pose:
            run_pose()
        else:
            log.info("Skipping pose extraction (--skip-pose)")

        if not args.skip_train:
            run_cnn_training()
        else:
            log.info("Skipping CNN training (--skip-train)")
