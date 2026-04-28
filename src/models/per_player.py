"""
per_player.py

Per-player logistic regression ensemble.

Why this helps:
  Stephen Curry scores 25+ ppg — his prop line and patterns are completely
  different from Kevon Looney (4 ppg). A single global model has to learn
  both simultaneously, which adds noise and dilutes the signal.  Training a
  separate lightweight model per player lets each one specialize on that
  player's individual tendencies.

  For unseen players at test time (players with no training data), we fall
  back to the global XGBoost model.

Architecture:
  - Per player: LogisticRegression(C=0.1) — strong L2 regularization because
    each player has few samples (~30-60 training rows)
  - Global fallback: LogisticRegression(C=0.05) trained on all data
  - Features: same flat window + static features as XGBoost

Usage:
  from src.models.per_player import train_per_player
  model = train_per_player()
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import (
    CHECKPOINTS_DIR,
    FEATURE_COLS,
    PROCESSED_DIR,
    WINDOW_SIZE,
)

STATIC_COLS = ["PROP_LINE_Z", "LINE_VS_RECENT"]
CKPT_PATH   = os.path.join(CHECKPOINTS_DIR, "per_player_best.pkl")
MIN_SAMPLES  = 15   # minimum training rows to fit a player-specific model


def _feature_cols(df: pd.DataFrame) -> list[str]:
    window_cols = [f"{feat}_t{t}" for t in range(WINDOW_SIZE) for feat in FEATURE_COLS]
    return window_cols + [c for c in STATIC_COLS if c in df.columns]


def _make_pipeline(C: float = 0.1) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=C, max_iter=500, random_state=42)),
    ])


def train_per_player(
    train_path: str | None = None,
    val_path:   str | None = None,
) -> dict:
    train_path = train_path or os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = val_path   or os.path.join(PROCESSED_DIR, "val.csv")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    feat_cols = _feature_cols(train_df)

    X_train_all = train_df[feat_cols].fillna(0.0).values.astype(np.float32)
    y_train_all = train_df["LABEL"].values.astype(int)

    # Global fallback model
    global_model = _make_pipeline(C=0.05)
    global_model.fit(X_train_all, y_train_all)

    # Per-player models
    player_models: dict[str, Pipeline] = {}
    for player, grp in train_df.groupby("PLAYER_NAME"):
        if len(grp) < MIN_SAMPLES:
            print(f"  [per_player] {player}: only {len(grp)} samples — using global fallback")
            continue
        X_p = grp[feat_cols].fillna(0.0).values.astype(np.float32)
        y_p = grp["LABEL"].values.astype(int)
        if len(np.unique(y_p)) < 2:
            print(f"  [per_player] {player}: single class — using global fallback")
            continue
        m = _make_pipeline(C=0.1)
        m.fit(X_p, y_p)
        player_models[player] = m

    print(f"[per_player] trained {len(player_models)} player models + 1 global fallback")

    # Validate
    bundle = {"global": global_model, "players": player_models}
    probs, labels = _predict(val_df, feat_cols, bundle)
    preds  = (np.array(probs) >= 0.5).astype(int)
    labels = np.array(labels)
    val_auc = roc_auc_score(labels, probs)
    val_acc = accuracy_score(labels, preds)
    val_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"[per_player] val_auc={val_auc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(CKPT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[per_player] saved → {CKPT_PATH}")

    return bundle


def _predict(df: pd.DataFrame, feat_cols: list[str], bundle: dict):
    global_model  = bundle["global"]
    player_models = bundle["players"]

    probs, labels = [], []
    for _, row in df.iterrows():
        player = row.get("PLAYER_NAME", "")
        x = np.array([row.get(c, 0.0) for c in feat_cols], dtype=np.float32).reshape(1, -1)
        model = player_models.get(player, global_model)
        prob  = float(model.predict_proba(x)[0, 1])
        probs.append(prob)
        labels.append(int(row["LABEL"]))
    return probs, labels


def load_per_player() -> dict | None:
    if not os.path.exists(CKPT_PATH):
        return None
    with open(CKPT_PATH, "rb") as f:
        return pickle.load(f)


def eval_per_player(csv_path: str | None = None) -> dict | None:
    csv_path = csv_path or os.path.join(PROCESSED_DIR, "test.csv")
    bundle = load_per_player()
    if bundle is None:
        print("[PerPlayer] checkpoints/per_player_best.pkl not found — skipping")
        return None
    df = pd.read_csv(csv_path)
    feat_cols = _feature_cols(df)
    probs, labels = _predict(df, feat_cols, bundle)
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "auroc":    roc_auc_score(labels, probs),
    }


if __name__ == "__main__":
    train_per_player()
