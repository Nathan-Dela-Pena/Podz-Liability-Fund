"""
fetch_odds.py

Pulls historical player prop lines (player_points over/under) and money-line
odds for Warriors games from The Odds API across both target seasons:
  • 2023-24: 2023-10-24 → 2024-04-14
  • 2024-25: 2024-10-22 → 2025-04-13

Storage
-------
All results are written to a SINGLE file: data/raw/odds/all_odds.json
Structure:
  {
    "2023-10-24": {
      "event_id":   "...",
      "home_team":  "Golden State Warriors",
      "away_team":  "Phoenix Suns",
      "season":     "2023-24",
      "moneyline":  { "Golden State Warriors": -140, "Phoenix Suns": 118 },
      "prop_lines": { "Stephen Curry": 29.5, ... }
    },
    ...
  }

Multi-snapshot strategy
------------------------
Bookmakers post player prop lines 2-4 hours before tip-off (~7-9pm ET).
We try three UTC snapshots in order and take the first that returns props:
  1. 19:00Z  (~2-3pm ET)  — props often live by now for evening games
  2. 22:00Z  (~5-6pm ET)  — most props posted by this point
  3. 01:00Z next day      — catches late tip-offs and west-coast games

If none return props, the game is still recorded with empty prop_lines so
the date is not re-fetched on subsequent runs.

Usage:
  python -m src.ingestion.fetch_odds          # both seasons
  python -m src.ingestion.fetch_odds 2024-25  # single season
  python -m src.ingestion.fetch_odds --refetch-empty  # retry dates with no props
"""

import json
import os
import sys
import time
from datetime import date, timedelta

import requests

from config import (
    ODDS_API_KEY,
    ODDS_BASE_URL,
    SPORT,
    REGION,
    ODDS_FORMAT,
    TEAMS,
    PLAYERS,
    TRAIN_START, VAL_END,
    TEST_START,  TEST_END,
    RAW_ODDS_DIR,
    SEASONS,
)

_SLEEP = 0.6  # seconds between API calls

# Single output file for all odds data
ALL_ODDS_PATH = os.path.join(RAW_ODDS_DIR, "all_odds.json")

# UTC snapshots to try per day, in order — stop at first that returns props
SNAPSHOTS = ["T19:00:00Z", "T22:00:00Z"]
# 01:00Z next day handled separately (different date string)
SNAPSHOT_NEXT_DAY = "T01:00:00Z"

SEASON_RANGES = {
    "2023-24": (TRAIN_START, "2024-04-14"),
    "2024-25": ("2024-10-22", VAL_END),
    "2025-26": (TEST_START,   TEST_END),
}


# ---------------------------------------------------------------------------
# Single-file cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if os.path.exists(ALL_ODDS_PATH):
        with open(ALL_ODDS_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(RAW_ODDS_DIR, exist_ok=True)
    with open(ALL_ODDS_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, end: str):
    current = date.fromisoformat(start)
    stop    = date.fromisoformat(end)
    while current <= stop:
        yield current.isoformat()
        current += timedelta(days=1)


def _next_day(date_str: str) -> str:
    return (date.fromisoformat(date_str) + timedelta(days=1)).isoformat()


def _season_for_date(d: str) -> str:
    y = int(d[:4])
    m = int(d[5:7])
    season_year = y if m >= 10 else y - 1
    return f"{season_year}-{str(season_year + 1)[-2:]}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> dict:
    params = dict(params)
    params["apiKey"] = ODDS_API_KEY
    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"    [api] {resp.status_code} | used={used} remaining={remaining}")
    resp.raise_for_status()
    return resp.json()


def _fetch_team_events(date_str: str, snapshot_suffix: str) -> list[dict]:
    """
    Returns all target-team events at the given snapshot.
    One call covers all teams — filters for any team in TEAMS.
    """
    if snapshot_suffix == SNAPSHOT_NEXT_DAY:
        query_date = _next_day(date_str)
    else:
        query_date = date_str

    snapshot = f"{query_date}{snapshot_suffix}"
    url      = f"{ODDS_BASE_URL}/historical/sports/{SPORT}/odds/"
    params   = {
        "regions":    REGION,
        "markets":    "h2h",
        "oddsFormat": ODDS_FORMAT,
        "date":       snapshot,
    }
    resp_data = _get(url, params)
    events    = resp_data.get("data", [])
    return [
        e for e in events
        if e.get("home_team") in TEAMS or e.get("away_team") in TEAMS
    ]


def _fetch_player_props(event_id: str, date_str: str,
                        snapshot_suffix: str) -> dict:
    if snapshot_suffix == SNAPSHOT_NEXT_DAY:
        query_date = _next_day(date_str)
    else:
        query_date = date_str

    snapshot = f"{query_date}{snapshot_suffix}"
    url      = f"{ODDS_BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds/"
    params   = {
        "regions":    REGION,
        "markets":    "player_points",
        "oddsFormat": ODDS_FORMAT,
        "date":       snapshot,
    }
    return _get(url, params).get("data", {})


def _parse_prop_lines(props_data: dict) -> dict:
    lines = {}
    for bookmaker in props_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "player_points":
                continue
            for outcome in market.get("outcomes", []):
                name  = outcome.get("description", "")
                point = outcome.get("point")
                if name in PLAYERS and name not in lines and point is not None:
                    lines[name] = point
        if len(lines) == len(PLAYERS):
            break
    return lines


def _parse_moneyline(event: dict) -> dict:
    """Flatten moneyline to { team_name: best_price } across bookmakers."""
    prices: dict[str, list] = {}
    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                prices.setdefault(outcome["name"], []).append(outcome["price"])
    # Return median price per team
    return {team: sorted(ps)[len(ps) // 2] for team, ps in prices.items()}


# ---------------------------------------------------------------------------
# Core per-date fetch with multi-snapshot fallback
# ---------------------------------------------------------------------------

def _cache_key(date_str: str, team: str) -> str:
    """Cache key: 'YYYY-MM-DD|Team Name' — supports multiple games per date."""
    return f"{date_str}|{team}"


def fetch_date(date_str: str, cache: dict, refetch_empty: bool = False) -> int:
    """
    Fetches all target-team games + props for one date.
    Cache is keyed by 'date|team' so Warriors and Spurs games on the same
    date are stored independently.
    Updates cache in-place. Returns number of new games fetched.
    """
    print(f"\n[{date_str}] fetching...")

    # Step 1: find all target-team events — one API call covers everything
    events_by_team: dict[str, dict] = {}  # { team_name: event }
    for suffix in SNAPSHOTS + [SNAPSHOT_NEXT_DAY]:
        try:
            found_events = _fetch_team_events(date_str, suffix)
            time.sleep(_SLEEP)
        except requests.HTTPError as e:
            print(f"  ERROR at {suffix}: {e}")
            time.sleep(_SLEEP)
            continue

        for event in found_events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            for team in TEAMS:
                if team in (home, away) and team not in events_by_team:
                    events_by_team[team] = event
                    print(f"  Found ({team}): {away} @ {home} [{suffix}]")

        if len(events_by_team) == len(TEAMS):
            break  # found all teams, no need to try more snapshots

    if not events_by_team:
        return 0

    games_fetched = 0
    for team, event in events_by_team.items():
        key      = _cache_key(date_str, team)
        event_id = event["id"]
        home     = event.get("home_team", "")
        away     = event.get("away_team", "")

        # Skip if already cached with props
        if key in cache:
            if cache[key].get("prop_lines") and not refetch_empty:
                continue
            if not refetch_empty:
                continue

        # Step 2: try each snapshot for props
        prop_lines = {}
        for suffix in SNAPSHOTS + [SNAPSHOT_NEXT_DAY]:
            try:
                props_raw = _fetch_player_props(event_id, date_str, suffix)
                time.sleep(_SLEEP)
            except requests.HTTPError as e:
                print(f"  ERROR props at {suffix}: {e}")
                time.sleep(_SLEEP)
                continue

            lines = _parse_prop_lines(props_raw)
            if lines:
                print(f"  [{team}] Props at {suffix}: {len(lines)} players")
                prop_lines = lines
                break
            else:
                print(f"  [{team}] No props at {suffix}, trying next...")

        if not prop_lines:
            print(f"  [{team}] No prop lines found across all snapshots")

        cache[key] = {
            "date":       date_str,
            "team":       team,
            "event_id":   event_id,
            "home_team":  home,
            "away_team":  away,
            "season":     _season_for_date(date_str),
            "moneyline":  _parse_moneyline(event),
            "prop_lines": prop_lines,
        }
        games_fetched += 1

    return games_fetched


# ---------------------------------------------------------------------------
# Season / all fetchers
# ---------------------------------------------------------------------------

def fetch_season(season: str, refetch_empty: bool = False) -> None:
    if season not in SEASON_RANGES:
        raise ValueError(f"Unknown season '{season}'. Choose from {list(SEASON_RANGES)}")

    start, end = SEASON_RANGES[season]
    print(f"\n=== Fetching odds: {season} ({start} → {end}) ===")

    cache = _load_cache()
    games_found = 0

    for date_str in _date_range(start, end):
        n = fetch_date(date_str, cache, refetch_empty=refetch_empty)
        if n:
            games_found += n
            _save_cache(cache)

    with_props = sum(1 for v in cache.values() if v.get("prop_lines"))
    print(f"\nDone. {games_found} new game entries for {season}.")
    print(f"Cache total: {len(cache)} entries, {with_props} with prop lines")


def fetch_all(refetch_empty: bool = False) -> None:
    for season in SEASON_RANGES:
        fetch_season(season, refetch_empty=refetch_empty)


if __name__ == "__main__":
    args = sys.argv[1:]
    refetch = "--refetch-empty" in args
    seasons = [a for a in args if not a.startswith("--")]

    if seasons:
        fetch_season(seasons[0], refetch_empty=refetch)
    else:
        fetch_all(refetch_empty=refetch)
