"""
run_evaluation.py

Loads checkpoints/lstm_best.pt and evaluates the trained LSTM on all three
data splits (train / val / test).  Prints per-split metrics and a per-player
test breakdown, then saves checkpoints/lstm_results.json in the same schema
as baseline_results.json.

Requires:
  - checkpoints/lstm_best.pt  (produced by run_training.py)
  - data/processed/{train,val,test}.parquet  (produced by run_processing.py)

Usage:
  python scripts/run_evaluation.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from config import (
    CHECKPOINTS_DIR,
    DROPOUT,
    FEATURE_COLS,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PROCESSED_DIR,
    WINDOW_SIZE,
)
from src.models.lstm import LSTMBranch, PlayerSequenceDataset

CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "lstm_best.pt")


def _load_model(device: str) -> LSTMBranch:
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"{CKPT_PATH} not found.  Run scripts/run_training.py first."
        )
    model = LSTMBranch(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model


def _run_split(model: LSTMBranch, csv_path: str, device: str):
    """Returns (all_labels, all_preds, all_probs, player_names) arrays."""
    ds = PlayerSequenceDataset(csv_path, window_size=WINDOW_SIZE)

    all_labels: list[int] = []
    all_preds:  list[int] = []
    all_probs:  list[float] = []
    player_names: list[str] = []

    # Collect player names in same order as dataset samples.
    # PlayerSequenceDataset stores samples as (window_x, label).
    # We need player names for per-player breakdown; rebuild from parquet.
    df = pd.read_csv(csv_path).sort_values(["PLAYER_NAME", "GAME_DATE"])
    player_seq: list[str] = []
    for player, grp in df.groupby("PLAYER_NAME"):
        n_games = len(grp)
        n_samples = n_games - WINDOW_SIZE
        if n_samples > 0:
            player_seq.extend([player] * n_samples)

    with torch.no_grad():
        for idx, (x, y) in enumerate(ds):
            x = x.unsqueeze(0).to(device)           # (1, W, F)
            logit, _ = model(x)
            prob = torch.sigmoid(logit).item()
            pred = int(prob >= 0.5)
            all_labels.append(int(y.item()))
            all_preds.append(pred)
            all_probs.append(prob)
            player_names.append(player_seq[idx] if idx < len(player_seq) else "unknown")

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        player_names,
    )


def _print_and_collect(
    split_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> dict:
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds).tolist()

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
    print(classification_report(labels, preds, target_names=["Under", "Over"], digits=3))

    return {"accuracy": acc, "f1_macro": f1, "auroc": auc, "confusion_matrix": cm}


def _per_player_breakdown(
    labels: np.ndarray,
    preds: np.ndarray,
    player_names: list[str],
) -> dict:
    results: dict[str, dict] = {}
    unique_players = sorted(set(player_names))
    print(f"\n{'='*50}")
    print("PER-PLAYER TEST ACCURACY")
    print(f"{'='*50}")
    for player in unique_players:
        idxs = [i for i, p in enumerate(player_names) if p == player]
        if len(idxs) < 5:
            continue
        yp = labels[idxs]
        yhat = preds[idxs]
        acc = accuracy_score(yp, yhat)
        print(f"  {player:<25s}: {acc*100:.1f}%  (n={len(idxs)})")
        results[player] = {"accuracy": float(acc), "n": len(idxs)}
    return results


def _compare_to_baseline(lstm_results: dict) -> None:
    baseline_path = os.path.join(CHECKPOINTS_DIR, "baseline_results.json")
    if not os.path.exists(baseline_path):
        print("\n[info] baseline_results.json not found — skipping comparison")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\n{'='*50}")
    print("LSTM vs BASELINE (test set)")
    print(f"{'='*50}")
    for metric in ("accuracy", "f1_macro", "auroc"):
        b = baseline.get("test", {}).get(metric, float("nan"))
        l = lstm_results.get("test", {}).get(metric, float("nan"))
        if metric == "accuracy":
            delta = (l - b) * 100
            print(f"  {metric:<12s}: baseline={b*100:.2f}%  lstm={l*100:.2f}%  delta={delta:+.2f}pp")
        else:
            delta = l - b
            print(f"  {metric:<12s}: baseline={b:.4f}  lstm={l:.4f}  delta={delta:+.4f}")


def run() -> None:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[eval] running on {device}")

    model = _load_model(device)

    results: dict = {}
    for split in ("train", "val", "test"):
        path = os.path.join(PROCESSED_DIR, f"{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found.  Run scripts/run_processing.py first."
            )
        labels, preds, probs, player_names = _run_split(model, path, device)
        results[split] = _print_and_collect(split, labels, preds, probs)

        if split == "test":
            results["per_player_test"] = _per_player_breakdown(labels, preds, player_names)

    _compare_to_baseline(results)

    out = os.path.join(CHECKPOINTS_DIR, "lstm_results.json")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    run()
