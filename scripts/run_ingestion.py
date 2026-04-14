"""
run_ingestion.py

Orchestrates the full data pull:
  1. Game logs (nba_api)  -> data/raw/game_logs/
  2. Player prop odds      -> data/raw/odds/

Run:
  python scripts/run_ingestion.py            # both seasons, all steps
  python scripts/run_ingestion.py --odds-only
  python scripts/run_ingestion.py --logs-only
  python scripts/run_ingestion.py --season 2024-25
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingestion.fetch_game_logs import fetch_all as fetch_logs_all, fetch_season as fetch_logs_season
from src.ingestion.fetch_odds      import fetch_all as fetch_odds_all, fetch_season as fetch_odds_season


def main():
    parser = argparse.ArgumentParser(description="Run data ingestion pipeline.")
    parser.add_argument("--season",     type=str, default=None, help="e.g. 2024-25")
    parser.add_argument("--logs-only",  action="store_true")
    parser.add_argument("--odds-only",  action="store_true")
    args = parser.parse_args()

    run_logs = not args.odds_only
    run_odds = not args.logs_only

    if run_logs:
        print("\n" + "="*50)
        print("STEP 1: Game Logs (nba_api)")
        print("="*50)
        if args.season:
            fetch_logs_season(args.season)
        else:
            fetch_logs_all()

    if run_odds:
        print("\n" + "="*50)
        print("STEP 2: Player Prop Odds (The Odds API)")
        print("="*50)
        if args.season:
            fetch_odds_season(args.season)
        else:
            fetch_odds_all()

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
