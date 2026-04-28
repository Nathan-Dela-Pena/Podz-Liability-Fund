"""
lstm.py

Bidirectional single-layer LSTM that encodes a rolling window of N game logs
per player + a small static-feature vector (prop-line context) and outputs a
binary over/under prediction.

Architecture
------------
  seq      : (batch, window_size, num_features)   — z-scored game stats
  static   : (batch, STATIC_DIM)                  — prop-line context features
                                                    [PROP_LINE_Z, LINE_VS_RECENT]

  BiLSTM(hidden=64) → last hidden state (concat fwd+bwd = 128)
  ↓
  concat with static-feature MLP output (32-d)
  ↓
  Dropout → FC(160, 64) → ReLU → FC(64, 1)

The static features are the single biggest improvement to this model:
without them the LSTM had no idea what threshold it was predicting against.
PROP_LINE_Z gives absolute level ("is this a high-scoring line?"), and
LINE_VS_RECENT gives form-relative context ("is the line above or below this
player's recent average?").

The hidden returned by forward() is the 128-d LSTM state ONLY (not the
post-static-fusion vector), so the FusionModel can combine the LSTM's pure
sequence representation with its own cross-modal mixing.

Usage:
  python -m src.models.lstm                 # standalone training
  from src.models.lstm import train_lstm
  train_lstm()
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    FEATURE_COLS,
    WINDOW_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATIC_FEATURES: tuple[str, ...] = ("PROP_LINE_Z", "LINE_VS_RECENT")
STATIC_DIM:      int             = len(STATIC_FEATURES)
STATIC_PROJ_DIM: int             = 32   # MLP output dim merged into FC head


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlayerSequenceDataset(Dataset):
    """
    Loads train/val/test CSV produced by build_sequences.py.

    Each row is one ready-to-use sample with:
      • Flattened window columns {FEAT}_t0 … {FEAT}_t{W-1} (z-scored)
      • Static features PROP_LINE_Z, LINE_VS_RECENT
      • LABEL ∈ {0, 1}

    Returns (seq_tensor, static_tensor, label_tensor).

    Backward compatibility: if the CSV is old and lacks the static feature
    columns, those features are filled with zeros and a one-time warning
    is printed.  This lets the training loop run even if the user forgot
    to re-run build_sequences.py.
    """

    def __init__(self, csv_path: str, window_size: int = WINDOW_SIZE):
        df = pd.read_csv(csv_path)
        self.window = window_size

        missing = [c for c in STATIC_FEATURES if c not in df.columns]
        if missing:
            print(f"[lstm] WARNING: {csv_path} is missing static columns {missing}; "
                  f"filling with zeros. Re-run `python -m src.processing.build_sequences` "
                  f"to enable the prop-line features.")

        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        for _, row in df.iterrows():
            window_x = np.zeros((window_size, len(FEATURE_COLS)), dtype=np.float32)
            for t in range(window_size):
                for j, feat in enumerate(FEATURE_COLS):
                    window_x[t, j] = float(row.get(f"{feat}_t{t}", 0.0))

            static_x = np.array(
                [float(row[c]) if c in row else 0.0 for c in STATIC_FEATURES],
                dtype=np.float32,
            )

            self.samples.append((window_x, static_x, int(row["LABEL"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, s, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(s), torch.tensor(y, dtype=torch.float32)

    def class_weights(self) -> torch.Tensor:
        """Returns per-sample weights for WeightedRandomSampler."""
        labels = np.array([s[2] for s in self.samples])
        counts = np.bincount(labels)
        weight_per_class = 1.0 / counts
        return torch.tensor([weight_per_class[l] for l in labels], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMBranch(nn.Module):
    """
    Single-layer bidirectional LSTM encoder + static-feature MLP merged
    before the classification head.

    forward(seq, static=None) returns:
      logit  : (batch, 1)
      hidden : (batch, hidden_size * 2) — pure LSTM hidden state, used by
               FusionModel.  Static features do NOT contaminate this vector
               because fusion has its own mixing layer.
    """

    def __init__(
        self,
        input_size:  int   = len(FEATURE_COLS),
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = DROPOUT,
        static_dim:  int   = STATIC_DIM,
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
        self.dropout  = nn.Dropout(dropout)
        lstm_out_dim  = hidden_size * 2  # bidirectional

        # Attention over the sequence — lets the model focus on the most
        # informative game in the window rather than always using the last step.
        self.attn_layer = nn.Linear(lstm_out_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        # Static-feature MLP — small projection that lets the model learn
        # non-linear interactions on the prop-line context.
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, STATIC_PROJ_DIM),
            nn.ReLU(inplace=True),
        )

        head_in = lstm_out_dim + STATIC_PROJ_DIM
        self.fc = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        seq:    torch.Tensor,
        static: torch.Tensor | None = None,
    ):
        """
        seq    : (batch, seq_len, input_size)
        static : (batch, STATIC_DIM) — defaults to zeros if None
        """
        outputs, _ = self.lstm(seq)             # (batch, seq_len, hidden_size*2)
        # Attention over all timesteps — score each step, softmax, weighted sum
        scores  = self.attn_layer(outputs)      # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        hidden  = (weights * outputs).sum(dim=1)  # (batch, hidden_size*2)
        hidden  = self.layer_norm(hidden)
        hidden  = self.dropout(hidden)

        if static is None:
            static = torch.zeros(seq.size(0), STATIC_DIM,
                                 device=seq.device, dtype=seq.dtype)
        s = self.static_mlp(static)              # (batch, STATIC_PROJ_DIM)

        logit = self.fc(torch.cat([hidden, s], dim=-1))   # (batch, 1)
        return logit, hidden


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _make_loader(path: str, shuffle: bool, weighted: bool) -> DataLoader:
    ds = PlayerSequenceDataset(path)
    if weighted:
        weights = ds.class_weights()
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_lstm(
    train_path: str | None = None,
    val_path:   str | None = None,
    device: str | None = None,
) -> LSTMBranch:
    """
    Trains the LSTM branch and saves checkpoints.
    Reads from PROCESSED_DIR/train.csv and val.csv (built by run_processing.py).
    Returns the trained model.
    """
    train_path = train_path or os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = val_path   or os.path.join(PROCESSED_DIR, "val.csv")

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    print(f"[lstm] training on {device}")

    train_loader = _make_loader(train_path, shuffle=True,  weighted=True)
    val_loader   = _make_loader(val_path,   shuffle=False, weighted=False)

    model     = LSTMBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val_auc = 0.0
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, s, y in train_loader:
            x, s, y = x.to(device), s.to(device), y.to(device)
            optimizer.zero_grad()
            logit, _ = model(x, s)
            loss = criterion(logit.squeeze(1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, s, y in val_loader:
                x, s, y = x.to(device), s.to(device), y.to(device)
                logit, _ = model(x, s)
                loss = criterion(logit.squeeze(1), y)
                val_loss += loss.item() * len(x)
                probs = torch.sigmoid(logit.squeeze(1))
                preds = (probs >= 0.5).long()
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().long().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc  = accuracy_score(all_labels, all_preds)
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = float("nan")

        scheduler.step(val_loss)

        print(
            f"  Epoch {epoch:>3}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        # Checkpoint on best val_auc — better signal than accuracy at thresholds
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            ckpt = os.path.join(CHECKPOINTS_DIR, "lstm_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"    [checkpoint] saved (val_auc={val_auc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  [early stop] val_auc hasn't improved for {PATIENCE} epochs")
                break

    print(f"\n[lstm] best val AUC: {best_val_auc:.4f}")
    return model


if __name__ == "__main__":
    train_lstm()
