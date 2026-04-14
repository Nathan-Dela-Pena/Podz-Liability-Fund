"""
lstm.py

Bidirectional single-layer LSTM that encodes a rolling window of N game logs
per player and outputs a binary over/under prediction.

Architecture (per proposal):
  Input  → BiLSTM(hidden=64) → last hidden state (concat fwd+bwd = 128)
          → Dropout → FC(128, 64) → ReLU → FC(64, 1) → Sigmoid

Input shape:  (batch, window_size, num_features)
Output shape: (batch, 1)  — probability of going OVER the prop line

Weighted cross-entropy is used to handle class imbalance (proposal §B).

Usage:
  # standalone training
  python -m src.models.lstm

  # from run_training.py
  from src.models.lstm import train_lstm
  train_lstm()
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import (
    FEATURE_COLS,
    WINDOW_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlayerSequenceDataset(Dataset):
    """
    Loads the merged game-log + odds parquet from PROCESSED_DIR and builds
    rolling windows of length WINDOW_SIZE for each player.

    Each sample is:
      x : (WINDOW_SIZE, num_features)  — normalized feature window
      y : scalar 0/1                   — 0 = under, 1 = over
    """

    def __init__(self, csv_path: str, window_size: int = WINDOW_SIZE):
        df = pd.read_csv(csv_path).sort_values(["PLAYER_NAME", "GAME_DATE"])
        self.window = window_size
        self.samples: list[tuple[np.ndarray, int]] = []

        for player, grp in df.groupby("PLAYER_NAME"):
            grp = grp.reset_index(drop=True)
            feats  = grp[FEATURE_COLS].values.astype(np.float32)
            labels = grp["LABEL"].values.astype(np.int64)   # 0 or 1

            # z-score normalise per player sequence
            mean = feats.mean(axis=0, keepdims=True)
            std  = feats.std(axis=0, keepdims=True) + 1e-8
            feats = (feats - mean) / std

            for i in range(window_size, len(grp)):
                window_x = feats[i - window_size : i]        # (W, F)
                label    = int(labels[i])
                self.samples.append((window_x, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

    def class_weights(self) -> torch.Tensor:
        """Returns per-sample weights for WeightedRandomSampler."""
        labels = np.array([s[1] for s in self.samples])
        counts = np.bincount(labels)
        weight_per_class = 1.0 / counts
        return torch.tensor([weight_per_class[l] for l in labels], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMBranch(nn.Module):
    """
    Single-layer bidirectional LSTM encoder.

    forward() returns:
      logit  : (batch, 1) — raw pre-sigmoid score
      hidden : (batch, hidden_size * 2) — last hidden state for fusion
    """

    def __init__(
        self,
        input_size:  int  = len(FEATURE_COLS),
        hidden_size: int  = HIDDEN_SIZE,
        num_layers:  int  = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        lstm_out_dim = hidden_size * 2  # bidirectional

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_size)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size)
        # grab last layer's fwd and bwd hidden states and concat
        fwd = h_n[-2]   # (batch, hidden_size)
        bwd = h_n[-1]   # (batch, hidden_size)
        hidden = torch.cat([fwd, bwd], dim=-1)   # (batch, hidden_size*2)
        hidden = self.dropout(hidden)
        logit  = self.fc(hidden)                 # (batch, 1)
        return logit, hidden


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _make_loader(path: str, shuffle: bool, weighted: bool) -> DataLoader:
    ds = PlayerSequenceDataset(path)
    if weighted:
        weights  = ds.class_weights()
        sampler  = WeightedRandomSampler(weights, len(weights))
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_lstm(
    train_path: str | None = None,
    val_path:   str | None = None,
    device: str | None = None,
) -> LSTMBranch:
    """
    Trains the LSTM branch and saves checkpoints.
    Defaults to PROCESSED_DIR/train.csv and val.csv.
    Returns the trained model.
    """
    train_path = train_path or os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = val_path   or os.path.join(PROCESSED_DIR, "val.csv")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[lstm] training on {device}")

    train_loader = _make_loader(train_path, shuffle=True,  weighted=True)
    val_loader   = _make_loader(val_path,   shuffle=False, weighted=False)

    model     = LSTMBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit, _ = model(x)
            loss = criterion(logit.squeeze(1), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logit, _ = model(x)
                loss = criterion(logit.squeeze(1), y)
                val_loss += loss.item() * len(x)
                preds = (torch.sigmoid(logit.squeeze(1)) >= 0.5).long()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().long().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc  = accuracy_score(all_labels, all_preds)
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(
            f"  Epoch {epoch:>3}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(CHECKPOINTS_DIR, "lstm_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"    [checkpoint] saved (val_acc={val_acc:.4f})")

    print(f"\n[lstm] best val accuracy: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    train_lstm()
