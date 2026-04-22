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

print(f"\n=== Step 2: Fetch play-by-play for game {game_id} ===")
data = _get(PBP_ENDPOINT, {"GameID": game_id, "StartPeriod": 1, "EndPeriod": 10})
if data is None:
    print("Play-by-play returned None — API blocked or timed out")
    sys.exit(1)

result_sets = data.get("resultSets", [])
if not result_sets:
    print("No resultSets in response")
    print("Keys in response:", list(data.keys()))
    sys.exit(1)

headers = result_sets[0]["headers"]
rows    = result_sets[0]["rowSet"]
print(f"Columns: {headers}")
print(f"Total events: {len(rows)}")

if rows:
    print(f"\nFirst row: {rows[0]}")
    print(f"\nSample EVENTMSGTYPE values: {list(set(r[headers.index('EVENTMSGTYPE')] for r in rows[:50]))}")
    print(f"Sample PLAYER1_ID values:   {list(set(r[headers.index('PLAYER1_ID')] for r in rows[:50]))}")
    print(f"Looking for player_id={PLAYER_ID} (type={type(PLAYER_ID).__name__})")

    # Check if any events match Curry
    curry_events = [r for r in rows if r[headers.index("PLAYER1_ID")] == PLAYER_ID]
    print(f"\nEvents where PLAYER1_ID == {PLAYER_ID}: {len(curry_events)}")

    fga_events = [r for r in rows
                  if r[headers.index("PLAYER1_ID")] == PLAYER_ID
                  and r[headers.index("EVENTMSGTYPE")] in {1, 2}]
    print(f"FGA events (type 1 or 2) for Curry: {len(fga_events)}")

    if fga_events:
        event_id = str(fga_events[0][headers.index("EVENTNUM")])
        print(f"\n=== Step 3: Fetch video URL for event {event_id} ===")
        vdata = _get(VIDEO_ENDPOINT, {"GameEventID": event_id, "GameID": game_id})
        if vdata is None:
            print("Video endpoint returned None")
        else:
            print("Video response keys:", list(vdata.keys()))
            result = vdata.get("resultSets", {})
            print("resultSets type:", type(result))
            print("resultSets content:", result)
