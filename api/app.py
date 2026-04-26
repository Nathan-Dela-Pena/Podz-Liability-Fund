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
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    FEATURE_COLS, WINDOW_SIZE,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [api] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app + rate limiter
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)
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
# Feature builder — rolling 5-game window from processed CSVs (no live API)
# ---------------------------------------------------------------------------

def _load_windows_from_csv() -> dict:
    """
    Build a {player_name: (window, prop_line)} dict from the most recent
    processed CSV (test.csv preferred, train.csv as fallback).

    The CSV stores features as {FEAT}_t0 (most recent) .. {FEAT}_t4 (oldest).
    We reconstruct a (WINDOW_SIZE, n_features) array in oldest→newest order
    so it matches what the LSTM was trained on.
    """
    for csv_name in ("test.csv", "train.csv"):
        csv_path = Path(PROCESSED_DIR) / csv_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        # one row per player — their most recent game window
        latest = (
            df.sort_values("GAME_DATE")
              .groupby("PLAYER_NAME", sort=False)
              .last()
              .reset_index()
        )

        results = {}
        for _, row in latest.iterrows():
            name = row["PLAYER_NAME"]
            # Build (WINDOW_SIZE, n_features): index 0 = oldest (t4), 4 = newest (t0)
            window = np.zeros((WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float32)
            for step, t in enumerate(range(WINDOW_SIZE - 1, -1, -1)):
                for f_idx, feat in enumerate(FEATURE_COLS):
                    col = f"{feat}_t{t}"
                    if col in row.index and pd.notna(row[col]):
                        window[step, f_idx] = float(row[col])

            prop_line = None
            if "PROP_LINE" in row.index and pd.notna(row["PROP_LINE"]):
                prop_line = round(float(row["PROP_LINE"]), 1)

            results[name] = (window, prop_line)

        log.info("Loaded %d player windows from %s", len(results), csv_name)
        return results

    log.error("No processed CSV found in %s", PROCESSED_DIR)
    return {}


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
    _get_model()  # ensure loaded

    player_windows = _load_windows_from_csv()
    if not player_windows:
        return jsonify({"error": "No processed data found", "players": []}), 503

    results = []
    for name, (window, prop_line) in player_windows.items():
        try:
            recommendation, confidence = _run_inference(window)
        except Exception as exc:
            log.exception("Inference failed for %s", name)
            continue

        results.append({
            "name":           name,
            "recommendation": recommendation,
            "confidence":     confidence,
            "prop_line":      prop_line,
            "stat_type":      "PTS",
        })

    # Sort by confidence descending so highest-conviction picks come first
    results.sort(key=lambda r: r["confidence"], reverse=True)

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
