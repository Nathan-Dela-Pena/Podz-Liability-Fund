"""
run_cnn_pipeline.py

Combined pipeline that:
  1. Fetches each FGA video clip from stats.nba.com
  2. Extracts sampled JPEG frames immediately
  3. Deletes the clip — frames only (~2-5 GB vs ~100+ GB for clips)
  4. Runs MediaPipe Pose on frames → data/pose/*.csv
  5. Trains the CNN branch → checkpoints/cnn_best.pt

Storage strategy
----------------
Clips are never kept on disk long-term.  Only frames are stored, which are
~10-50x smaller.  Back up data/frames/ and data/pose/ to Google Drive after
this script completes — those are the only outputs you need.

Usage:
  python scripts/run_cnn_pipeline.py               # full pipeline
  python scripts/run_cnn_pipeline.py --skip-fetch  # if frames already exist
  python scripts/run_cnn_pipeline.py --skip-pose   # if pose CSVs already exist
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

from config import FRAMES_DIR, PLAYERS, PLAYER_IDS, SEASONS, POSE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cnn_pipeline] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 + 2 + 3: Fetch clip → extract frames → delete clip
# ---------------------------------------------------------------------------

def fetch_and_extract(
    players: list[str] | None = None,
    seasons: list[str] | None = None,
    sample_every: int = 5,
) -> None:
    """
    For each player × season × game × FGA event:
      - Download the mp4 to a temp file
      - Extract every Nth frame to data/frames/{slug}/{game_id}_{event_id}/
      - Delete the temp file immediately

    Skips events whose frame directory already exists and is non-empty.
    """
    from src.ingestion.fetch_clips import (
        _fetch_game_ids, _fetch_fga_events, _fetch_video_url,
        _player_slug, SLEEP_BETWEEN_REQUESTS, CDN_HEADERS,
    )
    from src.processing.extract_frames import extract_frames_from_clip
    import requests

    players = players or PLAYERS
    seasons = seasons or ["2025-26"]
    frames_root = Path(FRAMES_DIR)

    total_events   = 0
    total_skipped  = 0
    total_no_video = 0

    for name in players:
        player_id = PLAYER_IDS.get(name)
        if not player_id:
            log.warning("No ID for %s — skipping", name)
            continue

        slug = _player_slug(name)
        log.info("=== %s ===", name)

        for season in seasons:
            game_ids = _fetch_game_ids(player_id, season)
            log.info("  %s: %d games", season, len(game_ids))

            for game_id in game_ids:
                events = _fetch_fga_events(game_id, player_id)

                for gid, eid in events:
                    out_dir = frames_root / slug / f"{gid}_{eid}"

                    # Skip if already extracted
                    if out_dir.exists() and any(out_dir.iterdir()):
                        total_skipped += 1
                        continue

                    video_url = _fetch_video_url(gid, eid)
                    if not video_url:
                        total_no_video += 1
                        continue

                    # Download to temp file, extract frames, delete immediately
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                            tmp_path = Path(tmp.name)

                        with requests.get(video_url, headers=CDN_HEADERS,
                                          stream=True, timeout=60,
                                          allow_redirects=True, max_redirects=10) as resp:
                            resp.raise_for_status()
                            with open(tmp_path, "wb") as fh:
                                for chunk in resp.iter_content(chunk_size=1 << 16):
                                    fh.write(chunk)

                        n = extract_frames_from_clip(tmp_path, out_dir, sample_every)
                        log.info("  [+] %s/%s_%s → %d frames", slug, gid, eid, n)
                        total_events += 1

                    except Exception as exc:
                        log.warning("  Failed %s/%s: %s", gid, eid, exc)
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()   # always delete the clip

                    time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(
        "Done. Extracted: %d  Skipped: %d  No video: %d",
        total_events, total_skipped, total_no_video,
    )


# ---------------------------------------------------------------------------
# Step 4: Pose extraction
# ---------------------------------------------------------------------------

def run_pose() -> None:
    from src.processing.extract_pose import extract_pose_features as extract_pose_all
    log.info("=== Running MediaPipe Pose extraction ===")
    extract_pose_all()


# ---------------------------------------------------------------------------
# Step 5: Train CNN
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
                        help="Skip fetch+extract (frames already in data/frames/)")
    parser.add_argument("--skip-pose",  action="store_true",
                        help="Skip pose extraction (CSVs already in data/pose/)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip CNN training")
    parser.add_argument("--every", type=int, default=5,
                        help="Sample 1 frame every N frames (default: 5)")
    args = parser.parse_args()

    if not args.skip_fetch:
        log.info("=== Step 1-3: Fetch clips → extract frames → delete clips ===")
        fetch_and_extract(seasons=["2025-26"], sample_every=args.every)
    else:
        log.info("Skipping fetch (--skip-fetch)")

    # Check frames exist before attempting pose / training
    frames_root = Path(FRAMES_DIR)
    has_frames  = frames_root.exists() and any(frames_root.rglob("*.jpg"))

    if not has_frames:
        log.warning(
            "No frames found in %s — skipping pose extraction and CNN training.\n"
            "  Fix clip fetching first, then re-run with --skip-fetch.",
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
