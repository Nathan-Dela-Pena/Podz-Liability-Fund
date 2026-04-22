"""
fetch_clips.py

Downloads short play-by-play video clips from stats.nba.com for each player
in the PLAYERS list.  Only field-goal-attempt events are fetched (event type 1)
as those capture the isolation/drive plays most useful for pose analysis.

Clips are saved as:
  data/raw/clips/{player_slug}/{game_id}_{event_id}.mp4

The NBA stats video endpoint returns a CDN URL for each event.  We hit the
play-by-play endpoint first to collect event IDs, then resolve each to a video
URL via the videoevents endpoint.

Rate-limiting: 0.6 s sleep between every HTTP request to avoid 429s.
Already-downloaded clips are skipped on re-runs.

Usage:
  python -m src.ingestion.fetch_clips
"""

import os
import time
import logging
from pathlib import Path

import requests

from config import (
    PLAYER_IDS,
    PLAYERS,
    RAW_CLIPS_DIR,
    SEASONS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLEEP_BETWEEN_REQUESTS: float = 1.2   # seconds — stats.nba.com needs breathing room
TIMEOUT:               int   = 60    # seconds per request
MAX_RETRIES:           int   = 3     # retry on timeout/5xx before giving up

# NBA stats API endpoints
PBP_ENDPOINT      = "https://stats.nba.com/stats/playbyplayv2"
VIDEO_ENDPOINT    = "https://stats.nba.com/stats/videoeventsasset"
GAMELOG_ENDPOINT  = "https://stats.nba.com/stats/playergamelog"

# Event action type for field-goal attempts (makes + misses)
FGA_EVENT_TYPES = {1, 2}   # 1 = made FG, 2 = missed FG

# stats.nba.com requires headers that match a real Chrome browser exactly —
# missing Origin or Sec-Fetch-* causes silent timeouts or 403s.
HEADERS = {
    "Host":               "stats.nba.com",
    "User-Agent":         (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":             "application/json, text/plain, */*",
    "Accept-Language":    "en-US,en;q=0.9",
    "Accept-Encoding":    "gzip, deflate, br",
    "Origin":             "https://www.nba.com",
    "Referer":            "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "Sec-Fetch-Site":     "same-site",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Dest":     "empty",
    "Connection":         "keep-alive",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fetch_clips] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> dict | None:
    """
    GET request with shared headers, retry logic, and graceful error handling.
    Retries up to MAX_RETRIES times on timeout or 5xx errors.
    Returns parsed JSON dict or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            log.warning("HTTP %s for %s (attempt %d/%d)", status, url, attempt, MAX_RETRIES)
            if status < 500:
                return None  # 4xx — no point retrying
        except requests.exceptions.Timeout:
            log.warning("Timeout for %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error: %s (attempt %d/%d)", exc, attempt, MAX_RETRIES)
            return None
        # Exponential backoff before retry
        time.sleep(SLEEP_BETWEEN_REQUESTS * (2 ** attempt))
    log.warning("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None


def _player_slug(name: str) -> str:
    """'Stephen Curry' → 'stephen_curry'"""
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


def _fetch_game_ids(player_id: int, season: str) -> list[str]:
    """
    Returns a list of game IDs the player appeared in during *season*.
    season format: '2024-25'
    """
    data = _get(
        GAMELOG_ENDPOINT,
        {"PlayerID": player_id, "Season": season, "SeasonType": "Regular Season"},
    )
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if data is None:
        return []

    result_sets = data.get("resultSets", [])
    if not result_sets:
        return []

    headers = result_sets[0]["headers"]
    rows    = result_sets[0]["rowSet"]
    gid_idx = headers.index("Game_ID")
    return [row[gid_idx] for row in rows]


def _fetch_fga_events(game_id: str, player_id: int) -> list[tuple[str, str]]:
    """
    Pulls play-by-play for *game_id* using nba_api (handles headers internally)
    and returns (game_id, event_id) tuples for every FGA by *player_id*.
    Uses PlayByPlayV3 — V2 was deprecated and returns empty JSON.
    """
    try:
        from nba_api.stats.endpoints import PlayByPlayV3
        pbp = PlayByPlayV3(game_id=game_id, start_period=1, end_period=10,
                           timeout=TIMEOUT)
        df = pbp.get_data_frames()[0]
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    except Exception as exc:
        log.warning("PlayByPlayV3 failed for game %s: %s", game_id, exc)
        return []

    if df.empty:
        return []

    # V3 schema: isFieldGoal == 1 for all FGA; personId is the shooter
    fga = df[
        (df["isFieldGoal"] == 1) &
        (df["personId"] == player_id)
    ]
    return [(game_id, str(eid)) for eid in fga["actionNumber"].tolist()]


def _fetch_video_url(game_id: str, event_id: str) -> str | None:
    """
    Resolves a CDN mp4 URL for the given game/event via the videoeventsasset
    endpoint.  Returns None if no video is available.
    """
    data = _get(VIDEO_ENDPOINT, {"GameEventID": event_id, "GameID": game_id})
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if data is None:
        return None

    # Response structure:
    # {"resultSets": {"Meta": {"videoUrls": [{"lurl": "..._1280x720.mp4", "murl": ..., "surl": ...}]}}}
    result_sets = data.get("resultSets", {})
    video_urls  = result_sets.get("Meta", {}).get("videoUrls", [])

    if not video_urls:
        return None
    # Prefer highest quality (lurl=1280x720), fall back to medium then small
    entry = video_urls[0]
    return entry.get("lurl") or entry.get("murl") or entry.get("surl") or None


CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}


def _download_clip(url: str, dest: Path) -> bool:
    """
    Streams an mp4 from *url* to *dest*.  Returns True on success.
    Uses plain CDN headers — not the stats.nba.com headers — to avoid
    redirect loops and connection drops from videos.nba.com.
    """
    try:
        session = requests.Session()
        session.max_redirects = 10
        with session.get(url, headers=CDN_HEADERS, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        return True
    except requests.exceptions.TooManyRedirects:
        log.debug("Too many redirects for %s — no video available", url)
        return False
    except requests.exceptions.RequestException as exc:
        log.warning("Download failed for %s: %s", url, exc)
        if dest.exists():
            dest.unlink()   # remove partial file
        return False


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_clips(
    players: list[str] | None = None,
    seasons: list[str] | None = None,
) -> None:
    """
    For each player × season, collects all FGA event IDs and downloads the
    corresponding video clips to RAW_CLIPS_DIR.

    Parameters
    ----------
    players : list of player names from config.PLAYERS (default: all)
    seasons : list of season strings like '2024-25' (default: config.SEASONS)
    """
    players = players or PLAYERS
    seasons = seasons or SEASONS

    total_downloaded = 0
    total_skipped    = 0

    for name in players:
        player_id = PLAYER_IDS.get(name)
        if player_id is None:
            log.warning("No ID found for %s — skipping.", name)
            continue

        slug      = _player_slug(name)
        clip_dir  = Path(RAW_CLIPS_DIR) / slug
        clip_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== %s (id=%d) ===", name, player_id)

        for season in seasons:
            log.info("  Season %s: fetching game IDs …", season)
            game_ids = _fetch_game_ids(player_id, season)
            log.info("  Found %d games.", len(game_ids))

            for game_id in game_ids:
                events = _fetch_fga_events(game_id, player_id)
                if not events:
                    continue

                for gid, eid in events:
                    dest = clip_dir / f"{gid}_{eid}.mp4"

                    if dest.exists():
                        total_skipped += 1
                        continue

                    video_url = _fetch_video_url(gid, eid)
                    if video_url is None:
                        log.debug("No video for game=%s event=%s", gid, eid)
                        continue

                    success = _download_clip(video_url, dest)
                    if success:
                        log.info("    [+] %s", dest.name)
                        total_downloaded += 1
                    time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(
        "Done. Downloaded: %d  Skipped (already existed): %d",
        total_downloaded,
        total_skipped,
    )


if __name__ == "__main__":
    # Current season only — all tracked players from config
    fetch_clips(seasons=["2025-26"])
