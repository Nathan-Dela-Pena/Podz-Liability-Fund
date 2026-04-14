"""
run_processing.py
Runs build_game_records then build_sequences.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.processing.build_game_records import build as build_records
from src.processing.build_sequences import build as build_sequences

if __name__ == "__main__":
    print("=== Step 1: Build Game Records ===")
    build_records()
    print("\n=== Step 2: Build Sequences ===")
    build_sequences()
