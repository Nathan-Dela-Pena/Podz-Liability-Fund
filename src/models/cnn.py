"""
cnn.py

MobileNetV2-based CNN branch for the multimodal fusion model.

Architecture
------------
Visual stream (per frame):
  MobileNetV2 (pretrained ImageNet) with classifier replaced by:
    Dropout(0.3) → Linear(1280, 256) → ReLU → Linear(256, 128)
  → 128-dim visual feature vector

Pose stream (per clip):
  135-dim pose feature vector  (132 landmark means + 3 derived signals)
  Linear(135, 64) → ReLU
  → 64-dim pose feature vector

Fusion head (standalone training / inference):
  concat(visual_128, pose_64) → Linear(192, 1)
  → logit for over/under binary prediction

forward() returns (logit, hidden) where *hidden* is the 192-dim concat.
The LSTM fusion model will replace the Linear(192, 1) head with a broader
cross-modal fusion layer.

Dataset
-------
CNNPoseDataset loads:
  • One representative frame per clip (the middle frame)
  • The 135-dim pose vector from data/pose/{player_slug}.csv
  • A binary label (0 = under, 1 = over) from data/processed/train.csv

Labels are joined on (player_name, game_id).  Clips without a matching label
row are silently dropped.

Training
--------
train_cnn() mirrors train_lstm():
  • Adam optimiser, BCEWithLogitsLoss
  • Best-checkpoint saved to checkpoints/cnn_best.pt (highest val accuracy)
  • Same epoch/batch/LR/dropout hyperparameters as lstm.py

Input frames are resized to 224×224 and normalised with ImageNet mean/std.

Usage:
  # standalone
  python -m src.models.cnn

  # from a script
  from src.models.cnn import train_cnn
  train_cnn()
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DROPOUT,
    EPOCHS,
    FRAMES_DIR,
    LEARNING_RATE,
    PLAYERS,
    PLAYER_IDS,
    POSE_DIR,
    PROCESSED_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSE_DIM:   int = 135   # 132 landmark means + 3 derived signals
VISUAL_DIM: int = 128
POSE_OUT:   int = 64
HIDDEN_DIM: int = VISUAL_DIM + POSE_OUT   # 192

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _player_slug(name: str) -> str:
    """'Stephen Curry' → 'stephen_curry'"""
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CNNPoseDataset(Dataset):
    """
    Loads (frame_tensor, pose_vector, label) triples.

    For each clip in FRAMES_DIR we pick the middle frame as the visual input.
    The 135-dim pose vector comes from the matching row in POSE_DIR.
    The binary label (0/1) is joined from a processed CSV keyed on
    (PLAYER_NAME, GAME_DATE) — game_id extracted from the clip filename.

    Parameters
    ----------
    labels_csv : path to train.csv or val.csv in PROCESSED_DIR
    frames_root : root of per-player frame directories (default: FRAMES_DIR)
    pose_root   : root of per-player pose CSVs (default: POSE_DIR)
    transform   : torchvision transform applied to each frame image
    """

    def __init__(
        self,
        labels_csv:  str,
        frames_root: str | None = None,
        pose_root:   str | None = None,
        transform:   transforms.Compose = _frame_transform,
    ):
        self.transform   = transform
        self.frames_root = Path(frames_root or FRAMES_DIR)
        self.pose_root   = Path(pose_root   or POSE_DIR)

        labels_df = pd.read_csv(labels_csv)
        # GAME_ID may not exist in the stats CSV — fall back to GAME_DATE as key
        if "GAME_ID" not in labels_df.columns:
            labels_df["GAME_ID"] = labels_df.get("GAME_DATE", "").astype(str)
        labels_df["GAME_ID"] = labels_df["GAME_ID"].astype(str).str.zfill(10)

        self.samples: list[tuple[Path, np.ndarray, int]] = []
        self._build_samples(labels_df)

    # ------------------------------------------------------------------
    def _build_samples(self, labels_df: pd.DataFrame) -> None:
        """Populate self.samples by joining frames + pose + labels."""
        # Build a fast lookup: (player_name, game_id) → label
        label_map: dict[tuple[str, str], int] = {}
        for _, row in labels_df.iterrows():
            key = (str(row["PLAYER_NAME"]), str(row["GAME_ID"]))
            label_map[key] = int(row["LABEL"])

        for player_name in PLAYERS:
            slug      = _player_slug(player_name)
            player_frame_dir = self.frames_root / slug
            pose_csv         = self.pose_root   / f"{slug}.csv"

            if not player_frame_dir.exists() or not pose_csv.exists():
                continue

            pose_df = pd.read_csv(pose_csv)
            # Build pose lookup: (game_id, event_id) → 135-dim vector
            pose_df["game_id"]  = pose_df["game_id"].astype(str).str.zfill(10)
            pose_df["event_id"] = pose_df["event_id"].astype(str)
            pose_lookup = self._build_pose_lookup(pose_df)

            for clip_dir in sorted(player_frame_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue

                # clip_dir.name: {game_id}_{event_id}
                parts = clip_dir.name.split("_", 1)
                if len(parts) != 2:
                    continue
                game_id, event_id = parts[0].zfill(10), parts[1]

                label = label_map.get((player_name, game_id))
                if label is None:
                    continue

                pose_vec = pose_lookup.get((game_id, event_id))
                if pose_vec is None:
                    continue

                frame_path = self._pick_middle_frame(clip_dir)
                if frame_path is None:
                    continue

                self.samples.append((frame_path, pose_vec, label))

    # ------------------------------------------------------------------
    @staticmethod
    def _build_pose_lookup(pose_df: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
        """
        Build a (game_id, event_id) → 135-dim numpy array dict.

        The 135-dim vector is: mean landmark columns (132) + 3 derived signals.
        """
        mean_cols = [c for c in pose_df.columns if c.endswith("_mean")]
        derived   = ["vertical_accel", "stride_length", "shoulder_droop"]

        # Ensure all expected columns are present
        feat_cols = [c for c in mean_cols if any(f"lm_{i}_" in c for i in range(33))]
        feat_cols = feat_cols[:132]   # guard against extra columns
        feat_cols += [c for c in derived if c in pose_df.columns]

        lookup: dict[tuple[str, str], np.ndarray] = {}
        for _, row in pose_df.iterrows():
            key = (str(row["game_id"]), str(row["event_id"]))
            vec = row[feat_cols].values.astype(np.float32)
            # Pad with zeros if fewer than POSE_DIM features found
            if len(vec) < POSE_DIM:
                vec = np.concatenate([vec, np.zeros(POSE_DIM - len(vec), dtype=np.float32)])
            else:
                vec = vec[:POSE_DIM]
            lookup[key] = vec
        return lookup

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_middle_frame(clip_dir: Path) -> Path | None:
        """Return the middle-indexed frame JPG in *clip_dir*, or None."""
        frames = sorted(clip_dir.glob("*.jpg"))
        if not frames:
            return None
        return frames[len(frames) // 2]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_path, pose_vec, label = self.samples[idx]

        img = Image.open(frame_path).convert("RGB")
        frame_tensor = self.transform(img)                          # (3, 224, 224)
        pose_tensor  = torch.tensor(pose_vec, dtype=torch.float32) # (135,)
        label_tensor = torch.tensor(label,    dtype=torch.float32) # scalar

        return frame_tensor, pose_tensor, label_tensor

    def class_weights(self) -> torch.Tensor:
        """Returns per-sample weights for WeightedRandomSampler."""
        labels = np.array([s[2] for s in self.samples])
        counts = np.bincount(labels.astype(int))
        weight_per_class = 1.0 / counts
        return torch.tensor(
            [weight_per_class[int(l)] for l in labels], dtype=torch.float32
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNNBranch(nn.Module):
    """
    MobileNetV2 visual encoder + lightweight pose MLP fused for binary
    over/under prediction.

    forward() returns:
      logit  : (batch, 1) — raw pre-sigmoid score
      hidden : (batch, 192) — fused visual+pose vector for cross-modal fusion
    """

    def __init__(self, pose_dim: int = POSE_DIM, dropout: float = DROPOUT):
        super().__init__()

        # ── Visual stream ────────────────────────────────────────────────
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # MobileNetV2.classifier is [Dropout, Linear(1280, 1000)]
        # Replace it with our projection head
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, VISUAL_DIM),   # → 128
        )
        self.visual_encoder = backbone

        # ── Pose stream ──────────────────────────────────────────────────
        self.pose_head = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(inplace=True),
        )

        # ── Standalone prediction head ───────────────────────────────────
        self.fusion_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(
        self,
        frame:    torch.Tensor,  # (batch, 3, 224, 224)
        pose_vec: torch.Tensor,  # (batch, 135)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logit  : (batch, 1)
        hidden : (batch, 192)
        """
        visual = self.visual_encoder(frame)        # (batch, 128)
        pose   = self.pose_head(pose_vec)          # (batch, 64)
        hidden = torch.cat([visual, pose], dim=-1) # (batch, 192)
        logit  = self.fusion_head(hidden)          # (batch, 1)
        return logit, hidden


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _make_loader(
    labels_csv: str,
    shuffle:    bool,
    weighted:   bool,
) -> DataLoader:
    ds = CNNPoseDataset(labels_csv)
    if weighted:
        weights = ds.class_weights()
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(
    train_path: Optional[str] = None,
    val_path:   Optional[str] = None,
    device:     Optional[str] = None,
) -> CNNBranch:
    """
    Trains the CNN branch and saves the best checkpoint.
    Mirrors train_lstm() in lstm.py (Adam / BCEWithLogitsLoss / early stopping
    via best-val-accuracy checkpointing).

    Parameters
    ----------
    train_path : path to train CSV (default: PROCESSED_DIR/train.csv)
    val_path   : path to val CSV   (default: PROCESSED_DIR/val.csv)
    device     : 'cuda' | 'mps' | 'cpu'  (auto-detected if None)

    Returns
    -------
    Trained CNNBranch model (on CPU).
    """
    train_path = train_path or os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = val_path   or os.path.join(PROCESSED_DIR, "val.csv")

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"[cnn] training on {device}")

    train_loader = _make_loader(train_path, shuffle=True,  weighted=True)
    val_loader   = _make_loader(val_path,   shuffle=False, weighted=False)

    model     = CNNBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for frames, poses, labels in train_loader:
            frames, poses, labels = frames.to(device), poses.to(device), labels.to(device)
            optimizer.zero_grad()
            logit, _ = model(frames, poses)
            loss = criterion(logit.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(frames)
        train_loss /= len(train_loader.dataset)

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for frames, poses, labels in val_loader:
                frames, poses, labels = frames.to(device), poses.to(device), labels.to(device)
                logit, _ = model(frames, poses)
                loss = criterion(logit.squeeze(1), labels)
                val_loss += loss.item() * len(frames)
                preds = (torch.sigmoid(logit.squeeze(1)) >= 0.5).long()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().long().tolist())

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
            ckpt = os.path.join(CHECKPOINTS_DIR, "cnn_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"    [checkpoint] saved (val_acc={val_acc:.4f})")

    print(f"\n[cnn] best val accuracy: {best_val_acc:.4f}")
    model.cpu()
    return model


if __name__ == "__main__":
    train_cnn()
