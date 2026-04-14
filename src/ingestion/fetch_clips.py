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

SLEEP_BETWEEN_REQUESTS: float = 0.6   # seconds

# NBA stats API endpoints
PBP_ENDPOINT      = "https://stats.nba.com/stats/playbyplayv2"
VIDEO_ENDPOINT    = "https://stats.nba.com/stats/videoeventsasset"
GAMELOG_ENDPOINT  = "https://stats.nba.com/stats/playergamelog"

# Event action type for field-goal attempts (makes + misses)
FGA_EVENT_TYPES = {1, 2}   # 1 = made FG, 2 = missed FG

HEADERS = {
    "Host":             "stats.nba.com",
    "User-Agent":       (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "Referer":          "https://www.nba.com/",
    "Connection":       "keep-alive",
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
    GET request with shared headers and graceful error handling.
    Returns parsed JSON dict or None on failure.
    """
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        log.warning("HTTP %s for %s %s", exc.response.status_code, url, params)
    except requests.exceptions.RequestException as exc:
        log.warning("Request error: %s", exc)
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
    Pulls play-by-play for *game_id* and returns (game_id, event_id) tuples
    for every field-goal attempt by *player_id*.
    """
    data = _get(PBP_ENDPOINT, {"GameID": game_id, "StartPeriod": 1, "EndPeriod": 10})
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if data is None:
        return []

    result_sets = data.get("resultSets", [])
    if not result_sets:
        return []

    headers = result_sets[0]["headers"]
    rows    = result_sets[0]["rowSet"]

    try:
        event_id_idx   = headers.index("EVENTNUM")
        event_type_idx = headers.index("EVENTMSGTYPE")
        p1_id_idx      = headers.index("PLAYER1_ID")
    except ValueError as exc:
        log.warning("Unexpected PBP schema for game %s: %s", game_id, exc)
        return []

    events = []
    for row in rows:
        if (
            row[event_type_idx] in FGA_EVENT_TYPES
            and row[p1_id_idx] == player_id
        ):
            events.append((game_id, str(row[event_id_idx])))
    return events


def _fetch_video_url(game_id: str, event_id: str) -> str | None:
    """
    Resolves a CDN mp4 URL for the given game/event via the videoeventsasset
    endpoint.  Returns None if no video is available.
    """
    data = _get(VIDEO_ENDPOINT, {"GameEventID": event_id, "GameID": game_id})
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if data is None:
        return None

    # Response structure: {"resultSets": {"Meta": ..., "video": {"Location": [...]}}}
    result_sets = data.get("resultSets", {})
    video_info  = result_sets.get("video", {})
    locations   = video_info.get("Location", [])

    if not locations:
        return None
    # Prefer the first (highest quality) location
    return locations[0] if isinstance(locations[0], str) else None


def _download_clip(url: str, dest: Path) -> bool:
    """
    Streams an mp4 from *url* to *dest*.  Returns True on success.
    """
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        return True
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
    fetch_clips()
