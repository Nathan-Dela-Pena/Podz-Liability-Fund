"""
baseline.py

Logistic Regression baseline for the Warriors player prop over/under prediction.

Input : data/processed/{train,val,test}.parquet  (built by build_sequences.py)
Output: prints accuracy, F1 (macro), AUROC, confusion matrix per split
        saves checkpoints/baseline_results.json

The flattened rolling-window features produced by build_sequences.py are used
directly (no sequence modelling — just a flat feature vector of WINDOW_SIZE x
num_features dimensions).  This is the simplest possible learned baseline
and serves as the bar the LSTM must beat by >= 3 percentage points.

Usage:
  python -m src.models.baseline
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from config import PROCESSED_DIR, CHECKPOINTS_DIR

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

META_COLS = {"PLAYER_NAME", "GAME_DATE", "PROP_LINE", "LABEL"}


def _load(split: str):
    path = os.path.join(PROCESSED_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run build_sequences.py first.")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["LABEL"].values.astype(int)
    return X, y, df


def evaluate(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y, y_pred)
    f1   = f1_score(y, y_pred, average="macro")
    try:
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = float("nan")
    cm   = confusion_matrix(y, y_pred).tolist()

    print(f"\n{'='*50}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  F1 (macro)  : {f1:.4f}")
    print(f"  AUROC       : {auc:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=pred):")
    print(f"    Under pred  Over pred")
    print(f"    {cm[0][0]:5d}       {cm[0][1]:5d}   <- Actual Under")
    print(f"    {cm[1][0]:5d}       {cm[1][1]:5d}   <- Actual Over")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["Under", "Over"], digits=3))

    return {"accuracy": acc, "f1_macro": f1, "auroc": auc, "confusion_matrix": cm}


def run() -> None:
    print("Loading data splits...")
    X_train, y_train, train_df = _load("train")
    X_val,   y_val,   val_df   = _load("val")
    X_test,  y_test,  test_df  = _load("test")

    print(f"  Train: {X_train.shape}  pos={y_train.mean()*100:.1f}%")
    print(f"  Val  : {X_val.shape}    pos={y_val.mean()*100:.1f}%")
    print(f"  Test : {X_test.shape}   pos={y_test.mean()*100:.1f}%")

    print("\nTraining Logistic Regression (class_weight='balanced')...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    print("  Done.")

    results = {}
    results["train"] = evaluate(model, X_train, y_train, "train")
    results["val"]   = evaluate(model, X_val,   y_val,   "val")
    results["test"]  = evaluate(model, X_test,  y_test,  "test")

    # Per-player breakdown on test set
    print(f"\n{'='*50}")
    print("PER-PLAYER TEST ACCURACY")
    print(f"{'='*50}")
    feat_cols = [c for c in test_df.columns if c not in META_COLS]
    player_results = {}
    for player, grp in test_df.groupby("PLAYER_NAME"):
        Xp = grp[feat_cols].values.astype(np.float32)
        yp = grp["LABEL"].values
        if len(yp) < 5:
            continue
        yhat = model.predict(Xp)
        acc  = accuracy_score(yp, yhat)
        print(f"  {player:<20s}: {acc*100:.1f}%  (n={len(yp)})")
        player_results[player] = {"accuracy": acc, "n": int(len(yp))}

    results["per_player_test"] = player_results

    out = os.path.join(CHECKPOINTS_DIR, "baseline_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    run()
