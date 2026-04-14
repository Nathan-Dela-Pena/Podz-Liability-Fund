"""
fetch_game_logs.py

Pulls per-game stats for every player in PLAYER_IDS across target seasons.
Fetches by player ID directly — no Warriors roster filter needed.

USG_PCT strategy
-----------------
PlayerGameLog does not expose usage rate. We fetch it per-game from
BoxScoreAdvancedV3 (usagePercentage field) using the GAME_ID already present
in the game log. One extra API call per game, so we batch only the players
we care about and cache by GAME_ID to avoid duplicate calls.

Traded-out players (e.g. Wiggins)
-----------------------------------
Logs are filtered to games on or before their trade date so post-trade
games (with the new team) are excluded.

Traded-in players (e.g. Butler, Horford)
------------------------------------------
All games across the full season are kept regardless of team — prop lines
are player-level, not team-level, so pre-trade performance is valid signal.

Output: one CSV per player-season in RAW_GAMELOGS_DIR.

Usage:
  python -m src.ingestion.fetch_game_logs          # all players, all seasons
  python -m src.ingestion.fetch_game_logs 2024-25  # single season
  python -m src.ingestion.fetch_game_logs --player "Jimmy Butler"
"""

import os
import sys
import time

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog, boxscoreadvancedv3

from config import SEASONS, PLAYER_IDS, TRADED_OUT, RAW_GAMELOGS_DIR

_SLEEP = 0.7   # seconds between nba_api calls

# Cache USG_PCT by game_id so shared games between players aren't fetched twice
_usg_cache: dict[str, dict[int, float]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_usg(game_id: str, player_id: int) -> float:
    """Fetch usagePercentage for a single player in a single game."""
    if game_id not in _usg_cache:
        time.sleep(_SLEEP)
        try:
            df = boxscoreadvancedv3.BoxScoreAdvancedV3(
                game_id=game_id
            ).get_data_frames()[0]
            # Build { player_id: usg } for all players in this game
            _usg_cache[game_id] = dict(
                zip(df["personId"].astype(int), df["usagePercentage"].astype(float))
            )
        except Exception as e:
            print(f"      [WARN] BoxScoreAdvancedV3 failed for {game_id}: {e}")
            _usg_cache[game_id] = {}
    return _usg_cache[game_id].get(player_id, float("nan"))


def _compute_ts(pts, fga, fta):
    denom = 2 * (fga + 0.44 * fta)
    return (pts / denom).where(denom > 0, other=0.0)


def _compute_efg(fg, fg3, fga):
    return ((fg + 0.5 * fg3) / fga).where(fga > 0, other=0.0)


def _pull_player_logs(player_name: str, player_id: int,
                      season: str) -> pd.DataFrame | None:
    """
    Fetches game-by-game logs for one player-season, adds USG_PCT,
    and applies traded-out date filter if applicable.
    """
    time.sleep(_SLEEP)
    try:
        log = PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = log.get_data_frames()[0]
    except Exception as e:
        print(f"    [WARN] Could not fetch {player_name} ({season}): {e}")
        return None

    if df.empty:
        return None

    # Normalise all column names to UPPER before anything else
    df.columns = [c.upper() for c in df.columns]

    # Parse date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Apply traded-out filter — drop games after the trade date
    if player_name in TRADED_OUT:
        cutoff = pd.to_datetime(TRADED_OUT[player_name])
        before = len(df)
        df = df[df["GAME_DATE"] <= cutoff]
        dropped = before - len(df)
        if dropped:
            print(f"    → Dropped {dropped} post-trade games (after {TRADED_OUT[player_name]})")

    if df.empty:
        return None

    # Derive TS% and eFG%
    df["TS_PCT"]  = _compute_ts(df["PTS"], df["FGA"], df["FTA"])
    df["EFG_PCT"] = _compute_efg(df["FGM"], df["FG3M"], df["FGA"])

    # Fetch USG_PCT per game from BoxScoreAdvancedV3
    print(f"    Fetching USG_PCT for {len(df)} games...")
    df["USG_PCT"] = df.apply(
        lambda r: _fetch_usg(str(r["GAME_ID"]).zfill(10), player_id),
        axis=1,
    )

    df["PLAYER_NAME"] = player_name
    df["PLAYER_ID"]   = player_id
    df["SEASON"]      = season
    df["GAME_DATE"]   = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    keep = [
        "PLAYER_NAME", "PLAYER_ID", "SEASON",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
        "MIN", "PTS", "FGA", "FTA", "FGM", "FG3M",
        "TS_PCT", "EFG_PCT", "USG_PCT",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_player_season(player_name: str, season: str,
                        overwrite: bool = False) -> None:
    player_id = PLAYER_IDS.get(player_name)
    if player_id is None:
        print(f"  [SKIP] {player_name} — not in PLAYER_IDS")
        return

    safe_name = player_name.replace(" ", "_").replace("'", "")
    out_path  = os.path.join(RAW_GAMELOGS_DIR, f"{safe_name}_{season}.csv")

    if os.path.exists(out_path) and not overwrite:
        print(f"  [SKIP] Already exists: {out_path}")
        return

    print(f"  Fetching: {player_name} ({player_id}) — {season}")
    df = _pull_player_logs(player_name, player_id, season)

    if df is None or df.empty:
        print(f"    → No data.")
        return

    df.to_csv(out_path, index=False)
    print(f"    → {len(df)} games saved → {out_path}")


def fetch_season(season: str, overwrite: bool = False) -> None:
    print(f"\n=== Fetching game logs: {season} ===")
    os.makedirs(RAW_GAMELOGS_DIR, exist_ok=True)
    for player_name in PLAYER_IDS:
        fetch_player_season(player_name, season, overwrite=overwrite)


def fetch_all(overwrite: bool = False) -> None:
    for season in SEASONS:
        fetch_season(season, overwrite=overwrite)


if __name__ == "__main__":
    args = sys.argv[1:]
    overwrite = "--overwrite" in args
    args = [a for a in args if not a.startswith("--")]

    if "--player" in sys.argv:
        idx = sys.argv.index("--player")
        player = sys.argv[idx + 1]
        for season in SEASONS:
            fetch_player_season(player, season, overwrite=True)
    elif args:
        fetch_season(args[0], overwrite=overwrite)
    else:
        fetch_all(overwrite=overwrite)
