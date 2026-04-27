"""
evaluate.py

Evaluates all available model checkpoints (LSTM, CNN, Fusion) on the test set
and prints Accuracy, Macro F1, and Test AUROC in a summary table.

Usage:
  python scripts/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add project root to path so config / src are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.models.cnn import CNNBranch
from src.models.fusion import FusionDataset, FusionModel

TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def _metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "f1_macro": f1, "auroc": auc}


def eval_lstm() -> dict | None:
    ckpt = os.path.join(CHECKPOINTS_DIR, "lstm_best.pt")
    if not os.path.exists(ckpt):
        print("[LSTM]  checkpoints/lstm_best.pt not found — skipping")
        return None

    model = LSTMBranch(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    ds = PlayerSequenceDataset(TEST_CSV, window_size=WINDOW_SIZE)
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in ds:
            logit, _ = model(x.unsqueeze(0).to(DEVICE))
            prob = torch.sigmoid(logit).item()
            labels.append(int(y.item()))
            preds.append(int(prob >= 0.5))
            probs.append(prob)

    return _metrics(np.array(labels), np.array(preds), np.array(probs))


def eval_cnn() -> dict | None:
    ckpt = os.path.join(CHECKPOINTS_DIR, "cnn_best.pt")
    if not os.path.exists(ckpt):
        print("[CNN]   checkpoints/cnn_best.pt not found — skipping")
        return None

    model = CNNBranch(dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    ds = FusionDataset(TEST_CSV)
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for seq, pose, frame, y in ds:
            frame = frame.unsqueeze(0).to(DEVICE)
            pose  = pose.unsqueeze(0).to(DEVICE)
            logit, _ = model(frame, pose)
            prob = torch.sigmoid(logit).item()
            labels.append(int(y.item()))
            preds.append(int(prob >= 0.5))
            probs.append(prob)

    return _metrics(np.array(labels), np.array(preds), np.array(probs))


def eval_fusion() -> dict | None:
    ckpt = os.path.join(CHECKPOINTS_DIR, "fusion_best.pt")
    if not os.path.exists(ckpt):
        print("[Fusion] checkpoints/fusion_best.pt not found — skipping")
        return None

    model = FusionModel(
        lstm_ckpt=os.path.join(CHECKPOINTS_DIR, "lstm_best.pt"),
        cnn_ckpt=os.path.join(CHECKPOINTS_DIR, "cnn_best.pt"),
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    ds = FusionDataset(TEST_CSV)
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for seq, pose, frame, y in ds:
            seq   = seq.unsqueeze(0).to(DEVICE)
            pose  = pose.unsqueeze(0).to(DEVICE)
            frame = frame.unsqueeze(0).to(DEVICE)
            logit, _ = model(seq, pose, frame)
            prob = torch.sigmoid(logit).item()
            labels.append(int(y.item()))
            preds.append(int(prob >= 0.5))
            probs.append(prob)

    return _metrics(np.array(labels), np.array(preds), np.array(probs))


def main():
    if not os.path.exists(TEST_CSV):
        print(f"Error: test set not found at {TEST_CSV}")
        sys.exit(1)

    print(f"Evaluating on {TEST_CSV}  (device={DEVICE})\n")

    results = {
        "LSTM":   eval_lstm(),
        "CNN":    eval_cnn(),
        "Fusion": eval_fusion(),
    }

    # Print table
    header = f"{'Model':<10}  {'Accuracy':>10}  {'Macro F1':>10}  {'AUROC':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        if m is None:
            print(f"{name:<10}  {'(skipped)':>10}  {'':>10}  {'':>10}")
        else:
            print(
                f"{name:<10}  {m['accuracy']*100:>9.2f}%  "
                f"{m['f1_macro']:>10.4f}  {m['auroc']:>10.4f}"
            )
    print("=" * len(header))


if __name__ == "__main__":
    main()
