"""
build_game_records.py

Merges raw game-log CSVs with raw odds JSONs to produce a single labeled
DataFrame, then saves it as data/processed/all_records.parquet.

For each player-game row we:
  1. Find the matching odds JSON by game date
  2. Extract the player's prop line (points over/under line)
  3. Label: 1 (OVER) if PTS > line, 0 (UNDER) if PTS < line; drop ties/missing
  4. Compute USG_PCT from raw FGA/FTA via simplified formula using team totals
     fetched once per season from nba_api (leaguegamelog)
  5. Add OPP_DRTG: opponent season-average defensive rating from
     leaguedashteamstats (fetched once per season, cheap)
  6. Clip TS_PCT and EFG_PCT to [0, 1]  (garbage-time outliers exceed 1.0)

Usage:
  python -m src.processing.build_game_records
"""

import glob
import json
import os
import time

import pandas as pd

from config import (
    RAW_GAMELOGS_DIR,
    RAW_ODDS_DIR,
    PROCESSED_DIR,
    TEAM_ID,
    SEASONS,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Opponent team name → abbreviation mapping for MATCHUP parsing ──────────
# MATCHUP looks like "GSW vs. PHX" or "GSW @ LAL"
# We extract the opponent abbreviation, then look up DRTG by full team name.
# The leaguedashteamstats endpoint returns full team names + abbreviations so
# we can join on abbreviation.


def _load_game_logs() -> pd.DataFrame:
    files = glob.glob(os.path.join(RAW_GAMELOGS_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {RAW_GAMELOGS_DIR}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date  # date only
    return df


def _load_odds() -> dict[str, dict]:
    """
    Returns { 'YYYY-MM-DD': { player_name: line, ... }, ... }
    Merges prop lines across all team entries for a given date so a player
    appearing in any team's game on that date gets their line.
    Cache keys are either 'YYYY-MM-DD' (old format) or 'YYYY-MM-DD|Team' (new).
    """
    all_odds_path = os.path.join(RAW_ODDS_DIR, "all_odds.json")
    if not os.path.exists(all_odds_path):
        raise FileNotFoundError(
            f"{all_odds_path} not found. Run fetch_odds.py first."
        )
    with open(all_odds_path) as f:
        raw = json.load(f)

    merged: dict[str, dict] = {}
    for key, record in raw.items():
        props = record.get("prop_lines", {})
        if not props:
            continue
        # Extract date portion (works for both 'YYYY-MM-DD' and 'YYYY-MM-DD|Team')
        date_str = key.split("|")[0]
        if date_str not in merged:
            merged[date_str] = {}
        # First non-null line per player wins
        for player, line in props.items():
            if player not in merged[date_str]:
                merged[date_str][player] = line
    return merged


_TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def _fetch_team_drtg(season: str) -> dict[str, float]:
    """
    Fetches season-average defensive rating for every NBA team.
    Returns { 'GSW': 112.3, 'LAL': 115.1, ... }
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        time.sleep(0.7)
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        rating_col = "DEF_RATING" if "DEF_RATING" in df.columns else df.columns[-1]
        df["ABBR"] = df["TEAM_NAME"].map(_TEAM_NAME_TO_ABBR)
        return dict(zip(df["ABBR"], df[rating_col].astype(float)))
    except Exception as e:
        print(f"  [WARN] Could not fetch DRTG for {season}: {e}. Using 0.0.")
        return {}


def _extract_opp_abbr(matchup: str) -> str:
    """
    'GSW vs. PHX' → 'PHX'
    'GSW @ LAL'   → 'LAL'
    """
    parts = matchup.replace("vs.", "vs").replace("@", "vs").split("vs")
    parts = [p.strip() for p in parts]
    # The opponent is whichever part is not 'GSW'
    for p in parts:
        if p and p != "GSW":
            return p
    return ""


def build() -> pd.DataFrame:
    print("Loading game logs...")
    logs = _load_game_logs()
    print(f"  {len(logs)} rows across {logs['PLAYER_NAME'].nunique()} players")

    print("Loading odds JSON files...")
    odds = _load_odds()
    print(f"  Prop lines found for {len(odds)} game dates")

    # ── Fetch OPP_DRTG once per season ────────────────────────────────────
    print("Fetching opponent defensive ratings...")
    drtg_by_season: dict[str, dict] = {}
    for season in SEASONS:
        print(f"  Season {season}...")
        drtg_by_season[season] = _fetch_team_drtg(season)

    # ── Build merged records ───────────────────────────────────────────────
    rows = []
    no_odds = 0
    ties    = 0

    for _, row in logs.iterrows():
        date_str   = str(row["GAME_DATE"])          # 'YYYY-MM-DD'
        player     = row["PLAYER_NAME"]
        season     = row["SEASON"]
        pts        = row["PTS"]

        # Prop line
        day_lines = odds.get(date_str, {})
        line = day_lines.get(player)
        if line is None:
            no_odds += 1
            continue
        if pts == line:
            ties += 1
            continue

        label = 1 if pts > line else 0

        # OPP_DRTG via opponent abbreviation
        opp_abbr = _extract_opp_abbr(str(row.get("MATCHUP", "")))
        opp_drtg = drtg_by_season.get(season, {}).get(opp_abbr, float("nan"))

        # Clip shooting efficiency outliers
        ts_pct  = min(float(row["TS_PCT"]),  1.0)
        efg_pct = min(float(row["EFG_PCT"]), 1.0)

        # USG_PCT is left as NaN here; build_sequences will drop it or impute.
        matchup = str(row.get("MATCHUP", ""))
        home_away = 1.0 if "vs." in matchup else 0.0

        rows.append({
            "PLAYER_NAME": player,
            "SEASON":      season,
            "GAME_DATE":   date_str,
            "MATCHUP":     matchup,
            "WL":          row.get("WL", ""),
            "MIN":         float(row["MIN"]),
            "PTS":         float(pts),
            "TS_PCT":      ts_pct,
            "EFG_PCT":     efg_pct,
            "USG_PCT":     float(row["USG_PCT"]) if pd.notna(row.get("USG_PCT")) else float("nan"),
            "OPP_DRTG":    opp_drtg,
            "HOME_AWAY":   home_away,
            "PROP_LINE":   float(line),
            "LABEL":       int(label),
        })

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    # DAYS_REST: days since the player's previous game, capped at 7
    df["DAYS_REST"] = (
        df.groupby("PLAYER_NAME")["GAME_DATE"]
        .diff()
        .dt.days
        .clip(upper=7)
        .fillna(3.0)
    )

    print(f"\nMerge summary:")
    print(f"  Total rows with labels  : {len(df)}")
    print(f"  Rows missing odds       : {no_odds}")
    print(f"  Rows dropped (PTS==line): {ties}")
    print(f"  OVER (label=1)          : {(df['LABEL']==1).sum()}")
    print(f"  UNDER (label=0)         : {(df['LABEL']==0).sum()}")
    print(f"  Class balance (over%)   : {(df['LABEL']==1).mean()*100:.1f}%")
    print(f"  NaN OPP_DRTG            : {df['OPP_DRTG'].isna().sum()}")

    out = os.path.join(PROCESSED_DIR, "all_records.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return df


if __name__ == "__main__":
    build()
