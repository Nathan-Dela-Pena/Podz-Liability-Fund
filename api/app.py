"""
api/app.py

Flask REST API for NBA player points prop predictions.

Endpoints
---------
GET /predict
    1. Fetches today's NBA player_points prop lines from The Odds API
       (per-event endpoint: /v4/sports/basketball_nba/events/{id}/odds).
    2. Fuzzy-matches each player against those in the trained dataset.
    3. Pulls their most recent rolling-stats window from data/processed/test.csv.
    4. Runs the LSTM model → OVER/UNDER + confidence.
    Returns picks sorted by confidence descending.

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
  ODDS_API_KEY    : The Odds API key (required for /predict)
  CHECKPOINTS_DIR : override checkpoint directory (default: checkpoints/)
  MODEL_DEVICE    : 'cuda' | 'cpu' (default: auto-detect)
"""

import os
import sys
import logging
from difflib import get_close_matches
from pathlib import Path

import requests as http
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
# Live prop lines from The Odds API
# ---------------------------------------------------------------------------

_ODDS_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
_ODDS_EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"


def _fetch_live_props() -> dict[str, float]:
    """
    Return {player_name: prop_line} for all player_points props available today.

    Flow:
      1. GET /events  → list of today's game IDs
      2. For each game: GET /events/{id}/odds?markets=player_points
      3. Parse outcomes — take the first bookmaker's Over line per player.
    """
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise ValueError("ODDS_API_KEY environment variable is not set")

    events_resp = http.get(
        _ODDS_EVENTS_URL,
        params={"apiKey": key},
        timeout=12,
    )
    events_resp.raise_for_status()
    events = events_resp.json()
    log.info("Odds API: %d games today (quota remaining: %s)",
             len(events),
             events_resp.headers.get("x-requests-remaining", "?"))

    props: dict[str, float] = {}
    for event in events:
        eid = event["id"]
        try:
            r = http.get(
                _ODDS_EVENT_ODDS_URL.format(event_id=eid),
                params={
                    "apiKey":      key,
                    "regions":     "us",
                    "markets":     "player_points",
                    "oddsFormat":  "american",
                },
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            log.warning("Props fetch failed for event %s: %s", eid, exc)
            continue

        # Take the first bookmaker available for each game
        for bookmaker in data.get("bookmakers", [])[:1]:
            for market in bookmaker.get("markets", []):
                if market["key"] != "player_points":
                    continue
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        player = outcome["description"]
                        if player not in props:
                            props[player] = float(outcome["point"])

    log.info("Odds API: extracted %d player prop lines", len(props))
    return props


def _normalize_name(name: str) -> str:
    return name.lower().replace("'", "").replace(".", "").replace("-", " ").strip()


def _match_player(odds_name: str, trained_names: set[str]) -> str | None:
    """Fuzzy-match an Odds API player name to a trained player name."""
    if odds_name in trained_names:
        return odds_name

    norm_odds = _normalize_name(odds_name)
    norm_map  = {_normalize_name(n): n for n in trained_names}

    if norm_odds in norm_map:
        return norm_map[norm_odds]

    matches = get_close_matches(norm_odds, norm_map.keys(), n=1, cutoff=0.82)
    if matches:
        return norm_map[matches[0]]
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
    _get_model()  # ensure model is loaded

    # ── 1. Load trained player windows from CSV ──────────────────────────
    player_windows = _load_windows_from_csv()
    if not player_windows:
        return jsonify({"error": "No processed data found", "players": []}), 503

    # ── 2. Fetch live prop lines ─────────────────────────────────────────
    if not os.environ.get("ODDS_API_KEY"):
        return jsonify({
            "error": "ODDS_API_KEY is not set — set this env var to enable live predictions",
            "players": [],
        }), 503

    try:
        live_props = _fetch_live_props()
    except ValueError as exc:
        return jsonify({"error": str(exc), "players": []}), 503
    except Exception as exc:
        log.exception("Odds API request failed")
        return jsonify({"error": f"Odds API error: {exc}", "players": []}), 502

    if not live_props:
        return jsonify({"error": "No player_points props found for today's games", "players": []}), 404

    # ── 3. Match props → trained players → inference ─────────────────────
    trained_names = set(player_windows.keys())
    results = []

    for odds_name, prop_line in live_props.items():
        matched = _match_player(odds_name, trained_names)
        if matched is None:
            log.debug("No trained match for '%s'", odds_name)
            continue

        window, _ = player_windows[matched]
        try:
            recommendation, confidence = _run_inference(window)
        except Exception:
            log.exception("Inference failed for %s", matched)
            continue

        results.append({
            "name":           matched,
            "recommendation": recommendation,
            "confidence":     confidence,
            "prop_line":      prop_line,
            "stat_type":      "PTS",
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    log.info("Returning %d predictions (%d props fetched, %d trained players)",
             len(results), len(live_props), len(trained_names))

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
