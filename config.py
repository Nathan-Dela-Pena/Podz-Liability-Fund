import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# --- The Odds API ---
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT         = "basketball_nba"
REGION        = "us"
ODDS_FORMAT   = "american"

# --- Target Teams ---
# Odds are fetched for any game where one of these teams appears.
TEAMS = [
    "Golden State Warriors",
    "San Antonio Spurs",
]
# Keep singular TEAM for any code still referencing it (Warriors primary)
TEAM    = "Golden State Warriors"
TEAM_ID = 1610612744  # NBA.com team ID (Warriors)

SAS_TEAM_ID = 1610612759  # San Antonio Spurs

# --- Target Players ---
# Warriors rotation (including mid-season trades)
# Spurs core rotation
PLAYERS = [
    # ── Golden State Warriors ──────────────────────────────────────────────
    # Core rotation
    "Stephen Curry",
    "Draymond Green",
    "Jonathan Kuminga",
    "Moses Moody",
    "Gary Payton II",
    "Kevon Looney",
    "Brandin Podziemski",
    "Gui Santos",
    "Quinten Post",
    "Will Richard",
    "Pat Spencer",
    # Traded IN mid-2024-25 (full-season logs kept — all teams)
    "Jimmy Butler",
    "Seth Curry",
    "Al Horford",
    "De'Anthony Melton",
    "Kristaps Porzingis",
    # Traded OUT mid-2024-25 (Warriors-era only — filter applied in fetch)
    "Andrew Wiggins",
    # ── San Antonio Spurs ──────────────────────────────────────────────────
    "Victor Wembanyama",
    "De'Aaron Fox",
    "Keldon Johnson",
    "Stephon Castle",
    "Dylan Harper",
]

# Player IDs — fetch by ID directly, no roster dependency
PLAYER_IDS = {
    # Warriors
    "Stephen Curry":       201939,
    "Draymond Green":      203110,
    "Jonathan Kuminga":    1630228,
    "Moses Moody":         1630541,
    "Gary Payton II":      1627780,
    "Kevon Looney":        1626172,
    "Brandin Podziemski":  1641764,
    "Gui Santos":          1630611,
    "Quinten Post":        1642366,
    "Will Richard":        1642954,
    "Pat Spencer":         1630311,
    "Jimmy Butler":        202710,
    "Seth Curry":          203552,
    "Al Horford":          201143,
    "De'Anthony Melton":   1629001,
    "Kristaps Porzingis":  204001,
    "Andrew Wiggins":      203952,
    # Spurs
    "Victor Wembanyama":   1641705,
    "De'Aaron Fox":        1628368,
    "Keldon Johnson":      1629640,
    "Stephon Castle":      1642264,
    "Dylan Harper":        1642844,
}

# Players traded OUT of their primary team mid-season.
# Logs are filtered to that team's era only (before trade date).
TRADED_OUT = {
    "Andrew Wiggins": "2025-02-06",   # GSW → MIA in Butler deal
    "De'Aaron Fox":   "2025-02-06",   # SAS → SAC (traded same deadline)
}

# Players traded IN to Golden State mid-2024-25.
# Full-season logs kept across all teams.
TRADED_IN_2425 = {
    "Jimmy Butler",
    "Seth Curry",
    "Al Horford",
    "De'Anthony Melton",
    "Kristaps Porzingis",
}

# --- Season identifiers (nba_api format) ---
SEASONS = ["2023-24", "2024-25", "2025-26"]

# --- Chronological data split ---
# Train : 2023-24 full season + 2024-25 first half
# Val   : 2024-25 second half (playoff push)
# Test  : 2025-26 full season (true holdout — model never sees this)
TRAIN_START = "2023-10-24"
TRAIN_END   = "2025-02-28"
VAL_START   = "2025-03-01"
VAL_END     = "2025-04-13"
TEST_START  = "2025-10-21"
TEST_END    = "2026-04-06"

# --- LSTM hyperparameters ---
WINDOW_SIZE   = 5      # rolling games used as input sequence
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
DROPOUT       = 0.35   # mild regularisation — 0.45 was too aggressive
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 5e-5   # light L2 — prevents blow-up without slowing early learning
BATCH_SIZE    = 32
EPOCHS        = 60
PATIENCE      = 7      # stop if val_acc hasn't improved for N epochs

# --- Feature columns (must match build_sequences.py) ---
FEATURE_COLS = ["PTS", "TS_PCT", "EFG_PCT", "USG_PCT", "OPP_DRTG", "MIN"]

# --- Paths ---
DATA_DIR         = "data"
RAW_ODDS_DIR     = f"{DATA_DIR}/raw/odds"
RAW_GAMELOGS_DIR = f"{DATA_DIR}/raw/game_logs"
RAW_CLIPS_DIR    = f"{DATA_DIR}/raw/clips"
PROCESSED_DIR    = f"{DATA_DIR}/processed"
FRAMES_DIR       = f"{DATA_DIR}/frames"
POSE_DIR         = f"{DATA_DIR}/pose"
CHECKPOINTS_DIR  = "checkpoints"
