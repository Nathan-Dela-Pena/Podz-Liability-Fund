"""
api/app.py

Flask REST API for NBA player points prop predictions.

Endpoints
---------
GET /predict
    Runs inference for all tracked players using their most recent game stats.
    Returns OVER/UNDER recommendation + confidence for each player.

GET /health
    Returns model load status and checkpoint info.

Model loading
-------------
Attempts to load fusion_best.pt at startup.  Falls back to LSTM-only
predictions if the fusion checkpoint doesn't exist (CNN not yet trained).

Rate limiting
-------------
10 requests / minute per IP via flask-limiter.

Usage
-----
  python api/app.py              # dev server
  gunicorn api.app:app           # production

Environment
-----------
  CHECKPOINTS_DIR : override checkpoint directory (default: checkpoints/)
  MODEL_DEVICE    : 'cuda' | 'cpu' (default: auto-detect)
"""

import os
import sys
import logging
from pathlib import Path
from functools import lru_cache

import torch
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    PLAYERS, PLAYER_IDS, FEATURE_COLS, WINDOW_SIZE,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [api] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app + rate limiter
# ---------------------------------------------------------------------------

app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10 per minute"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Model + checkpoint paths
# ---------------------------------------------------------------------------

FRONTEND_DIR    = ROOT / "frontend"
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", ROOT / "checkpoints"))
FUSION_CKPT     = CHECKPOINTS_DIR / "fusion_best.pt"
LSTM_CKPT       = CHECKPOINTS_DIR / "lstm_best.pt"

DEVICE = os.environ.get("MODEL_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------------------------
# Lazy model loader (loaded once at first request or startup)
# ---------------------------------------------------------------------------

_model      = None
_model_mode = None   # "fusion" or "lstm"


def _load_model():
    global _model, _model_mode

    if FUSION_CKPT.exists():
        from src.models.fusion import FusionModel
        from src.models.lstm import LSTMBranch
        from src.models.cnn import CNNBranch
        m = FusionModel(
            lstm_branch=LSTMBranch().to(DEVICE),
            cnn_branch=CNNBranch().to(DEVICE),
        )
        state = torch.load(FUSION_CKPT, map_location=DEVICE)
        m.load_state_dict(state)
        m.eval()
        _model      = m
        _model_mode = "fusion"
        log.info("Loaded fusion model from %s", FUSION_CKPT)

    elif LSTM_CKPT.exists():
        from src.models.lstm import LSTMBranch
        m = LSTMBranch(
            input_size=len(FEATURE_COLS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)
        state = torch.load(LSTM_CKPT, map_location=DEVICE)
        m.load_state_dict(state)
        m.eval()
        _model      = m
        _model_mode = "lstm"
        log.info("Loaded LSTM-only model from %s", LSTM_CKPT)

    else:
        log.error("No checkpoint found in %s", CHECKPOINTS_DIR)
        raise FileNotFoundError(
            f"No checkpoint found. Looked for:\n"
            f"  {FUSION_CKPT}\n  {LSTM_CKPT}"
        )


def _get_model():
    if _model is None:
        _load_model()
    return _model, _model_mode


# ---------------------------------------------------------------------------
# Feature builder — rolling 5-game window from nba_api
# ---------------------------------------------------------------------------

def _fetch_recent_stats(player_id: int, player_name: str, n_games: int = WINDOW_SIZE):
    """
    Fetch the last *n_games* regular-season games for *player_id* using nba_api.
    Returns a (n_games, len(FEATURE_COLS)) float32 array, or None on failure.
    """
    try:
        from nba_api.stats.endpoints import PlayerGameLog
        gl  = PlayerGameLog(player_id=player_id, season="2025-26",
                            season_type_all_star="Regular Season", timeout=30)
        df  = gl.get_data_frames()[0]
    except Exception as exc:
        log.warning("nba_api failed for %s: %s", player_name, exc)
        return None

    if df.empty or len(df) < n_games:
        log.warning("Not enough games for %s (%d found)", player_name, len(df))
        return None

    df = df.head(n_games)

    # Derive TS_PCT and EFG_PCT if missing
    if "TS_PCT" not in df.columns:
        df["TS_PCT"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df.get("FTA", 0)))
    if "EFG_PCT" not in df.columns:
        df["EFG_PCT"] = (df.get("FGM", df["FGA"] * 0.45) + 0.5 * df.get("FG3M", 0)) / df["FGA"].replace(0, 1)
    if "USG_PCT" not in df.columns:
        df["USG_PCT"] = 0.0
    if "OPP_DRTG" not in df.columns:
        df["OPP_DRTG"] = 0.0

    try:
        window = df[FEATURE_COLS].values.astype(np.float32)
    except KeyError as exc:
        log.warning("Missing feature columns for %s: %s", player_name, exc)
        return None

    # Per-column z-score normalisation using training set statistics
    train_csv = Path(PROCESSED_DIR) / "train.csv"
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        for j, col in enumerate(FEATURE_COLS):
            col_t0 = f"{col}_t0"
            if col_t0 in train_df.columns:
                mu, sigma = train_df[col_t0].mean(), train_df[col_t0].std()
                if sigma > 0:
                    window[:, j] = (window[:, j] - mu) / sigma

    return window[::-1].copy()  # oldest → newest


def _fetch_prop_line(player_name: str) -> float | None:
    """Attempt to fetch today's prop line from cached odds data."""
    try:
        odds_dir = ROOT / "data" / "raw" / "odds"
        files    = sorted(odds_dir.glob("*.json"))
        if not files:
            return None
        import json
        latest = json.loads(files[-1].read_text())
        for game in latest.values():
            lines = game.get("prop_lines", {})
            if player_name in lines:
                return lines[player_name]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(window: np.ndarray) -> tuple[str, float]:
    """
    Run forward pass and return (recommendation, confidence).
    window: (WINDOW_SIZE, n_features) float32 ndarray
    """
    model, mode = _get_model()

    seq = torch.tensor(window[np.newaxis], dtype=torch.float32).to(DEVICE)  # (1, W, F)

    with torch.no_grad():
        if mode == "fusion":
            # Fusion model needs (seq, pose, frame) — use zeros for missing modalities
            pose  = torch.zeros(1, 135, dtype=torch.float32).to(DEVICE)
            frame = torch.zeros(1, 3, 224, 224, dtype=torch.float32).to(DEVICE)
            logit, _ = model(seq, pose, frame)
        else:
            # LSTM-only
            logit, _ = model(seq)

    prob = torch.sigmoid(logit.squeeze()).item()
    recommendation = "OVER" if prob >= 0.5 else "UNDER"
    confidence     = prob if prob >= 0.5 else 1.0 - prob
    return recommendation, round(confidence, 4)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    ckpt_info = {}
    if FUSION_CKPT.exists():
        ckpt_info["fusion_best.pt"] = f"{FUSION_CKPT.stat().st_size / 1e6:.1f} MB"
    if LSTM_CKPT.exists():
        ckpt_info["lstm_best.pt"] = f"{LSTM_CKPT.stat().st_size / 1e6:.1f} MB"

    return jsonify({
        "status":      "ok",
        "model_mode":  _model_mode or "not loaded",
        "device":      DEVICE,
        "checkpoints": ckpt_info,
    })


@app.route("/predict")
@limiter.limit("10 per minute")
def predict():
    results = []

    for name in PLAYERS:
        player_id = PLAYER_IDS.get(name)
        if player_id is None:
            continue

        window = _fetch_recent_stats(player_id, name)
        if window is None:
            results.append({
                "name":           name,
                "recommendation": None,
                "confidence":     None,
                "prop_line":      _fetch_prop_line(name),
                "stat_type":      "points",
                "error":          "insufficient recent data",
            })
            continue

        try:
            recommendation, confidence = _run_inference(window)
        except Exception as exc:
            log.exception("Inference failed for %s", name)
            results.append({
                "name":           name,
                "recommendation": None,
                "confidence":     None,
                "prop_line":      _fetch_prop_line(name),
                "stat_type":      "points",
                "error":          str(exc),
            })
            continue

        results.append({
            "name":           name,
            "recommendation": recommendation,
            "confidence":     confidence,
            "prop_line":      _fetch_prop_line(name),
            "stat_type":      "points",
        })

    return jsonify({"players": results, "model_mode": _model_mode})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        _load_model()
    except FileNotFoundError as exc:
        log.warning(str(exc))
        log.warning("API will return errors until a checkpoint is available.")

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
