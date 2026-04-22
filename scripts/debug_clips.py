"""
debug_clips.py

Quick diagnostic to check what the NBA stats API is actually returning
for play-by-play and video endpoints. Run this to diagnose why fetch_clips
downloaded 0 clips.

Usage:
  python scripts/debug_clips.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.fetch_clips import _get, _fetch_game_ids, GAMELOG_ENDPOINT, PBP_ENDPOINT, VIDEO_ENDPOINT

PLAYER_ID = 201939   # Stephen Curry
SEASON    = "2025-26"

print("=== Step 1: Fetch game IDs ===")
game_ids = _fetch_game_ids(PLAYER_ID, SEASON)
print(f"Found {len(game_ids)} games")
if game_ids:
    print(f"First game ID: {game_ids[0]}")

if not game_ids:
    print("No games found — stopping")
    sys.exit(1)

game_id = game_ids[0]

print(f"\n=== Step 2: Fetch play-by-play via nba_api for game {game_id} ===")
try:
    from nba_api.stats.endpoints import PlayByPlayV3
    pbp  = PlayByPlayV3(game_id=game_id, start_period=1, end_period=10)
    df   = pbp.get_data_frames()[0]
    print(f"Total events: {len(df)}")

    if not df.empty:
        print(f"Looking for player_id={PLAYER_ID}")
        curry_df = df[df["personId"] == PLAYER_ID]
        print(f"Events where personId == {PLAYER_ID}: {len(curry_df)}")

        fga_df = curry_df[curry_df["isFieldGoal"] == 1]
        print(f"FGA events (isFieldGoal==1) for Curry: {len(fga_df)}")

        if not fga_df.empty:
            event_id = str(fga_df.iloc[0]["actionNumber"])
            print(f"\n=== Step 3: Fetch video URL for event {event_id} ===")
            from src.ingestion.fetch_clips import _get, VIDEO_ENDPOINT
            vdata = _get(VIDEO_ENDPOINT, {"GameEventID": event_id, "GameID": game_id})
            if vdata is None:
                print("Video endpoint returned None")
            else:
                print("Video response keys:", list(vdata.keys()))
                result = vdata.get("resultSets", {})
                print("resultSets type:", type(result))
                print("resultSets content:", result)
except Exception as exc:
    print(f"PlayByPlayV3 failed: {exc}")
