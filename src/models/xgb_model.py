"""
xgb_model.py

XGBoost over/under classifier trained on the same flat rolling-window features
as the LSTM, plus the prop-line static features.

Why XGBoost beats LSTM on small tabular data:
  - Gradient boosting handles non-linear feature interactions natively via splits
  - No gradient vanishing / sequence noise — all timesteps treated as flat features
  - Built-in regularization (L1/L2 on leaf weights) prevents overfitting
  - No need for z-score normalization (tree splits are invariant to scale)

Feature vector per sample (WINDOW_SIZE * len(FEATURE_COLS) + 2 static):
  PTS_t0..t4, TS_PCT_t0..t4, ..., HOME_AWAY_t0..t4, DAYS_REST_t0..t4,
  PROP_LINE_Z, LINE_VS_RECENT

Usage:
  from src.models.xgb_model import train_xgb
  model = train_xgb()
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from config import (
    CHECKPOINTS_DIR,
    FEATURE_COLS,
    PROCESSED_DIR,
    WINDOW_SIZE,
)

STATIC_COLS = ["PROP_LINE_Z", "LINE_VS_RECENT"]
CKPT_PATH   = os.path.join(CHECKPOINTS_DIR, "xgb_best.pkl")


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    window_cols = [f"{feat}_t{t}" for t in range(WINDOW_SIZE) for feat in FEATURE_COLS]
    feature_cols = window_cols + [c for c in STATIC_COLS if c in df.columns]
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df["LABEL"].values.astype(int)
    return X, y


def train_xgb(
    train_path: str | None = None,
    val_path:   str | None = None,
) -> XGBClassifier:
    train_path = train_path or os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = val_path   or os.path.join(PROCESSED_DIR, "val.csv")

    X_train, y_train = _load_xy(train_path)
    X_val,   y_val   = _load_xy(val_path)

    # Class balance weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 4,
        learning_rate     = 0.03,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = 5,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        scale_pos_weight  = scale_pos,
        eval_metric       = "auc",
        early_stopping_rounds = 30,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, probs)
    val_acc = accuracy_score(y_val, preds)
    val_f1  = f1_score(y_val, preds, average="macro", zero_division=0)

    print(f"\n[xgb] val_auc={val_auc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")
    print(f"[xgb] best iteration: {model.best_iteration}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(CKPT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[xgb] saved → {CKPT_PATH}")

    return model


def load_xgb() -> XGBClassifier | None:
    if not os.path.exists(CKPT_PATH):
        return None
    with open(CKPT_PATH, "rb") as f:
        return pickle.load(f)


def eval_xgb(csv_path: str | None = None) -> dict | None:
    csv_path = csv_path or os.path.join(PROCESSED_DIR, "test.csv")
    model = load_xgb()
    if model is None:
        print("[XGB]  checkpoints/xgb_best.pkl not found — skipping")
        return None
    X, y = _load_xy(csv_path)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro", zero_division=0),
        "auroc":    roc_auc_score(y, probs),
    }


if __name__ == "__main__":
    train_xgb()
