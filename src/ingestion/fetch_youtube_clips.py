"""
fetch_youtube_clips.py

Downloads NBA player highlight videos from YouTube using yt-dlp and extracts
frames for pose analysis.  Replaces the NBA stats.nba.com video endpoint which
blocks programmatic downloads.

Strategy
--------
For each player we search YouTube for:
  "{player_name} highlights {season} NBA"
  "{player_name} {season} all shots NBA"

We download the top N videos per query (default: 3), extract frames every
SAMPLE_EVERY frames, write them to FRAMES_DIR/{player_slug}/{video_id}/,
and skip any frames detected as NBA "Video Not Available" placeholders.

Requirements
------------
  pip install yt-dlp
  brew install ffmpeg  (strongly recommended for best quality)

Usage
-----
  python -m src.ingestion.fetch_youtube_clips
  python -m src.ingestion.fetch_youtube_clips --players "Stephen Curry" --videos-per-query 5
"""

import argparse
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from config import FRAMES_DIR, PLAYERS, SEASONS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEOS_PER_QUERY: int   = 3      # YouTube search results to download per query
SAMPLE_EVERY:     int   = 30     # keep 1 frame every N (highlight vids are longer)
MAX_DURATION:     int   = 600    # skip videos longer than 10 min (avoid full games)
MIN_DURATION:     int   = 30     # skip very short clips (<30 s)
SLEEP_BETWEEN:    float = 2.0    # seconds between yt-dlp calls

# Blue placeholder detection threshold (NBA "Video Not Available" screen)
PLACEHOLDER_BLUE_THRESHOLD: float = 0.5

# YouTube search queries per player (filled with .format(name=..., season=...))
SEARCH_QUERIES = [
    "{name} highlights {season_short} NBA",
    "{name} {season_short} best shots NBA highlights",
]


def _player_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


def _season_short(season: str) -> str:
    """'2024-25' → '2024-25'  (kept as-is; YouTube understands this)"""
    return season


def is_placeholder_frame(frame: np.ndarray) -> bool:
    """
    Returns True if *frame* is an NBA 'Video Not Available' placeholder.
    These frames are ~82% NBA blue (HSV hue 100-130).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    blue_pct = blue_mask.sum() / (255.0 * frame.shape[0] * frame.shape[1])
    return blue_pct > PLACEHOLDER_BLUE_THRESHOLD


def _extract_frames(video_path: Path, out_dir: Path, sample_every: int) -> int:
    """Extract sampled non-placeholder frames from *video_path* into *out_dir*."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open video: %s", video_path)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = written = skipped_placeholder = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            if is_placeholder_frame(frame):
                skipped_placeholder += 1
            else:
                dest = out_dir / f"frame_{written:04d}.jpg"
                cv2.imwrite(str(dest), frame)
                written += 1
        frame_idx += 1

    cap.release()
    if skipped_placeholder:
        log.debug("  Skipped %d placeholder frames in %s", skipped_placeholder, video_path.name)
    return written


def _yt_search_and_download(
    query: str,
    out_dir: Path,
    n: int = VIDEOS_PER_QUERY,
) -> list[Path]:
    """
    Search YouTube for *query* and download up to *n* videos into *out_dir*.
    Returns list of downloaded video paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    search_url = f"ytsearch{n}:{query}"

    cmd = [
        "yt-dlp",
        search_url,
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        "--format", "bestvideo[height<=480][ext=mp4]/best[height<=480][ext=mp4]/best[height<=480]/best",
        "--match-filter", f"duration >= {MIN_DURATION} & duration <= {MAX_DURATION}",
        "--no-playlist",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
    ]

    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        log.warning("yt-dlp failed for query '%s': %s", query, exc)
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp timed out for query '%s'", query)

    return sorted(out_dir.glob("*.mp4")) + sorted(out_dir.glob("*.webm"))


def fetch_youtube_frames(
    players: list[str] | None = None,
    seasons: list[str] | None = None,
    videos_per_query: int = VIDEOS_PER_QUERY,
    sample_every: int = SAMPLE_EVERY,
    frames_root: str | None = None,
) -> None:
    """
    Main entry point: search YouTube for each player × season, download
    highlight videos, and extract frames to FRAMES_DIR.

    Parameters
    ----------
    players          : player names (default: all from config)
    seasons          : season strings like '2024-25' (default: config.SEASONS)
    videos_per_query : number of YouTube results to download per search query
    sample_every     : keep 1 frame every N frames
    frames_root      : override FRAMES_DIR
    """
    players     = players or PLAYERS
    seasons     = seasons or SEASONS
    frames_root = Path(frames_root or FRAMES_DIR)

    total_videos  = 0
    total_frames  = 0

    for name in players:
        slug      = _player_slug(name)
        player_dir = frames_root / slug
        log.info("=== %s ===", name)

        for season in seasons:
            s = _season_short(season)

            for query_tpl in SEARCH_QUERIES:
                query = query_tpl.format(name=name, season_short=s)
                log.info("  Searching: %s", query)

                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path   = Path(tmp)
                    video_list = _yt_search_and_download(query, tmp_path, videos_per_query)

                    for video_path in video_list:
                        video_id = video_path.stem
                        out_dir  = player_dir / f"yt_{season.replace('-','_')}_{video_id}"

                        # Skip if already extracted
                        if out_dir.exists() and any(out_dir.iterdir()):
                            log.debug("  [skip] %s already extracted", video_id)
                            continue

                        n = _extract_frames(video_path, out_dir, sample_every)
                        log.info("  [+] %s → %d frames", video_id, n)
                        total_videos += 1
                        total_frames += n

                time.sleep(SLEEP_BETWEEN)

    log.info("Done. Videos processed: %d  Total frames: %d", total_videos, total_frames)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [yt_clips] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fetch YouTube NBA highlight frames")
    parser.add_argument("--players",          nargs="+", default=None)
    parser.add_argument("--seasons",          nargs="+", default=["2024-25", "2025-26"])
    parser.add_argument("--videos-per-query", type=int,  default=VIDEOS_PER_QUERY)
    parser.add_argument("--every",            type=int,  default=SAMPLE_EVERY)
    args = parser.parse_args()

    fetch_youtube_frames(
        players=args.players,
        seasons=args.seasons,
        videos_per_query=args.videos_per_query,
        sample_every=args.every,
    )
