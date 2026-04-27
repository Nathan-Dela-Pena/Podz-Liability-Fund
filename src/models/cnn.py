"""
cnn.py

MobileNetV2-based CNN branch for the multimodal fusion model.

Why this is structured the way it is
------------------------------------
The dataset for this project contains 73 YouTube highlight clips spread across
22 players, with NO mapping from clips → real NBA game IDs.  That means the
CNN cannot be supervised against per-game over/under labels: those labels do
not correspond to anything visible in the clip.  Earlier versions of this file
fabricated labels by random-sampling each player's historical OVER/UNDER ratio,
which produced AUROC = 0.5000 on the test set (pure noise).

This file instead trains the CNN with a **player-identity** auxiliary task,
which has real labels (we know which player is in each clip).  The encoder
learns a player-kinetic fingerprint — the way a particular athlete's body
moves during a play — and that 192-dim hidden representation is what
FusionModel consumes alongside the LSTM stat encoder.  Fusion is the layer
that converts kinetic-fingerprint + recent-stats into an over/under decision.

Architecture
------------
Visual stream (multi-frame):
  N_FRAMES_PER_CLIP frames sampled evenly from each clip.
  MobileNetV2 (pretrained ImageNet) → 1280-d per frame → mean-pool over frames
  → Dropout(0.3) → Linear(1280, 256) → ReLU → Linear(256, 128)
  → 128-dim visual feature vector.

Pose stream (per clip):
  267-dim pose feature vector
    132 landmark means + 132 landmark stds + 3 derived signals.
    The stds carry motion-variability info — the closest proxy we have to
    "exhaustion / form drift" given the data is per-clip aggregated, not
    per-frame.
  Linear(267, 64) → ReLU
  → 64-dim pose feature vector.

Standalone training head:
  concat(visual_128, pose_64) → Linear(192, num_players)
  Cross-entropy on player identity.

Fusion-time forward path:
  Same encoder, returns (logit, hidden) with hidden = (batch, 192).
  FusionModel ignores `logit` and consumes `hidden`.

Augmentation (training only)
----------------------------
RandomResizedCrop, ColorJitter, HorizontalFlip — so the encoder learns
kinematic structure rather than jersey colours / arena lighting.

Usage
-----
  python -m src.models.cnn        # standalone training (player-ID task)
  from src.models.cnn import train_cnn ; train_cnn()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score
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
    POSE_DIR,
    PROCESSED_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 132 landmark means + 132 landmark stds + 3 derived signals
POSE_DIM:           int = 267
VISUAL_DIM:         int = 128
POSE_OUT:           int = 64
HIDDEN_DIM:         int = VISUAL_DIM + POSE_OUT          # 192
N_FRAMES_PER_CLIP:  int = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _player_slug(name: str) -> str:
    """'Stephen Curry' → 'stephen_curry'"""
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


# ---------------------------------------------------------------------------
# Pose feature extraction
# ---------------------------------------------------------------------------

def _pose_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the 267 pose feature column names in canonical order:
      [lm_0_x_mean, lm_0_x_std, lm_0_y_mean, lm_0_y_std, ..., lm_32_vis_std,
       vertical_accel, stride_length, shoulder_droop]

    Missing columns are tolerated (the caller pads with zeros).
    """
    cols: list[str] = []
    for i in range(33):
        for axis in ("x", "y", "z", "vis"):
            for stat in ("mean", "std"):
                c = f"lm_{i}_{axis}_{stat}"
                if c in df.columns:
                    cols.append(c)
    for c in ("vertical_accel", "stride_length", "shoulder_droop"):
        if c in df.columns:
            cols.append(c)
    return cols


def _row_to_pose_vec(row: pd.Series, cols: list[str]) -> np.ndarray:
    """Pull `cols` out of `row`, replace NaN with 0, pad/truncate to POSE_DIM."""
    vec = pd.to_numeric(row[cols], errors="coerce").fillna(0.0).values.astype(np.float32)
    if len(vec) < POSE_DIM:
        vec = np.concatenate([vec, np.zeros(POSE_DIM - len(vec), dtype=np.float32)])
    else:
        vec = vec[:POSE_DIM]
    return vec


# ---------------------------------------------------------------------------
# Dataset — player-identity supervision over clips
# ---------------------------------------------------------------------------

class ClipIdentityDataset(Dataset):
    """
    Each sample is one clip: (frames_tensor, pose_vec, player_idx).

    `frames_tensor` has shape (N_FRAMES_PER_CLIP, 3, 224, 224) — frames evenly
    spaced across the clip so the encoder gets multi-frame context instead of
    a single still.

    `pose_vec` is the 267-dim pose feature vector (means + stds + derived).

    `player_idx` is the integer index into PLAYERS (real labels).

    The clip pool is the union of every clip in FRAMES_DIR/<slug>/ that has a
    matching pose row.  YouTube clips and any future game clips are both used.
    """

    def __init__(
        self,
        frames_root: str | None = None,
        pose_root:   str | None = None,
        train:       bool = True,
        n_frames:    int = N_FRAMES_PER_CLIP,
    ):
        self.frames_root = Path(frames_root or FRAMES_DIR)
        self.pose_root   = Path(pose_root   or POSE_DIR)
        self.transform   = _train_transform if train else _eval_transform
        self.n_frames    = n_frames

        self.samples: list[tuple[Path, np.ndarray, int]] = []
        self._build_samples()

    # ------------------------------------------------------------------
    def _build_samples(self) -> None:
        for player_idx, player_name in enumerate(PLAYERS):
            slug             = _player_slug(player_name)
            player_frame_dir = self.frames_root / slug
            pose_csv         = self.pose_root   / f"{slug}.csv"

            if not player_frame_dir.exists() or not pose_csv.exists():
                continue

            pose_df = pd.read_csv(pose_csv)
            feat_cols = _pose_feature_columns(pose_df)

            # Build (event_id) → pose_vec lookup.  We key on event_id alone:
            # every clip is unambiguously identified by its event_id within a
            # given player's pose CSV (game_id is "yt" for all rows today).
            pose_lookup: dict[str, np.ndarray] = {}
            for _, row in pose_df.iterrows():
                event_id = str(row.get("event_id", ""))
                pose_lookup[event_id] = _row_to_pose_vec(row, feat_cols)

            for clip_dir in sorted(player_frame_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue

                # Clip dirs come in two flavours:
                #   yt_2025_26_<videoid>  → event_id = "2025_26_<videoid>"
                #   <gameid>_<eventid>    → event_id = "<eventid>"
                name = clip_dir.name
                event_id = name[3:] if name.startswith("yt_") else name.split("_", 1)[-1]

                pose_vec = pose_lookup.get(event_id)
                if pose_vec is None:
                    continue

                frames = sorted(clip_dir.glob("*.jpg"))
                if len(frames) < 1:
                    continue

                self.samples.append((clip_dir, pose_vec, player_idx))

    # ------------------------------------------------------------------
    def _sample_frames(self, clip_dir: Path) -> torch.Tensor:
        """Return (n_frames, 3, 224, 224) tensor with evenly-spaced frames."""
        frames = sorted(clip_dir.glob("*.jpg"))
        if not frames:
            return torch.zeros(self.n_frames, 3, 224, 224)

        # Evenly-spaced indices across the clip
        idxs = np.linspace(0, len(frames) - 1, self.n_frames).round().astype(int)
        out = []
        for idx in idxs:
            img = Image.open(frames[idx]).convert("RGB")
            out.append(self.transform(img))
        return torch.stack(out, dim=0)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_dir, pose_vec, player_idx = self.samples[idx]
        frames_t = self._sample_frames(clip_dir)
        pose_t   = torch.tensor(pose_vec, dtype=torch.float32)
        label_t  = torch.tensor(player_idx, dtype=torch.long)
        return frames_t, pose_t, label_t

    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        """Per-sample weights so rare players are seen as often as common ones."""
        labels = np.array([s[2] for s in self.samples])
        counts = np.bincount(labels, minlength=len(PLAYERS))
        # Avoid division-by-zero for absent classes
        weight_per_class = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
        return torch.tensor(
            [weight_per_class[int(l)] for l in labels], dtype=torch.float32
        )


# ---------------------------------------------------------------------------
# Pose-only lookup helpers (used by FusionDataset)
# ---------------------------------------------------------------------------

def build_player_pose_table(pose_root: str | None = None) -> dict[str, np.ndarray]:
    """
    For each player, return the **mean pose vector across all their clips**.
    Used as a per-player kinetic fingerprint when FusionDataset cannot resolve
    a specific (player, game_id) pose row (which is currently every fusion
    sample, since all clips are YouTube).

    Returns
    -------
    dict[player_slug] = np.ndarray of shape (POSE_DIM,)
    """
    root = Path(pose_root or POSE_DIR)
    table: dict[str, np.ndarray] = {}
    if not root.exists():
        return table

    for player_name in PLAYERS:
        slug     = _player_slug(player_name)
        pose_csv = root / f"{slug}.csv"
        if not pose_csv.exists():
            continue
        df = pd.read_csv(pose_csv)
        feat_cols = _pose_feature_columns(df)
        if not feat_cols:
            continue
        vecs = np.stack(
            [_row_to_pose_vec(row, feat_cols) for _, row in df.iterrows()], axis=0
        )
        table[slug] = vecs.mean(axis=0).astype(np.float32)
    return table


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNNBranch(nn.Module):
    """
    MobileNetV2 visual encoder + pose MLP, with two heads:

      * `id_head`     : Linear(192, num_players) — used for standalone training
      * `fusion_head` : Linear(192, 1)           — used when this branch is
                                                   fine-tuned inside FusionModel

    forward() returns (logit, hidden):
      logit  : (batch, 1) — fusion_head output (kept for API compatibility
                            with the previous file; FusionModel ignores it)
      hidden : (batch, 192) — concat(visual, pose) for cross-modal fusion

    Multi-frame input
    -----------------
    The visual encoder accepts EITHER (batch, 3, 224, 224) — single frame —
    OR (batch, T, 3, 224, 224) — multi-frame — and mean-pools across the T
    axis when present.  This lets CNNBranch be used with single-frame callers
    (FusionDataset) as well as the multi-frame standalone training loop.
    """

    def __init__(
        self,
        pose_dim:    int   = POSE_DIM,
        dropout:     float = DROPOUT,
        num_players: int   = len(PLAYERS),
    ):
        super().__init__()

        # ── Visual stream ────────────────────────────────────────────────
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Strip MobileNetV2's classifier; we use it as a 1280-d feature extractor.
        backbone.classifier = nn.Identity()
        self.visual_features = backbone           # input (B,3,224,224) → (B,1280)

        self.visual_proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, VISUAL_DIM),           # → 128
        )

        # ── Pose stream ──────────────────────────────────────────────────
        self.pose_head = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, POSE_OUT),             # → 64
            nn.ReLU(inplace=True),
        )

        # ── Heads ────────────────────────────────────────────────────────
        self.id_head     = nn.Linear(HIDDEN_DIM, num_players)
        self.fusion_head = nn.Linear(HIDDEN_DIM, 1)

    # ------------------------------------------------------------------
    def encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, 3, 224, 224) OR (B, T, 3, 224, 224)
        returns: (B, 128)
        """
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            x   = frames.reshape(B * T, C, H, W)
            f   = self.visual_features(x)                 # (B*T, 1280)
            f   = f.reshape(B, T, -1).mean(dim=1)         # mean-pool over T → (B, 1280)
        else:
            f = self.visual_features(frames)              # (B, 1280)
        return self.visual_proj(f)                        # (B, 128)

    # ------------------------------------------------------------------
    def forward(
        self,
        frames:   torch.Tensor,
        pose_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.encode_visual(frames)               # (B, 128)
        pose   = self.pose_head(pose_vec)                 # (B, 64)
        hidden = torch.cat([visual, pose], dim=-1)        # (B, 192)
        logit  = self.fusion_head(hidden)                 # (B, 1)
        return logit, hidden

    # ------------------------------------------------------------------
    def forward_id(
        self,
        frames:   torch.Tensor,
        pose_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward path used by the player-identity standalone training loop."""
        visual = self.encode_visual(frames)
        pose   = self.pose_head(pose_vec)
        hidden = torch.cat([visual, pose], dim=-1)
        return self.id_head(hidden), hidden


# ---------------------------------------------------------------------------
# Train / val split — by clip, stratified by player
# ---------------------------------------------------------------------------

def _stratified_split(
    ds: ClipIdentityDataset,
    val_frac: float = 0.2,
    seed: int       = 42,
) -> tuple[list[int], list[int]]:
    """
    Per-player stratified split: hold out `val_frac` of each player's clips.
    Players with only 1 clip put it in train (no val).
    """
    rng = np.random.default_rng(seed)
    by_player: dict[int, list[int]] = {}
    for i, (_, _, p) in enumerate(ds.samples):
        by_player.setdefault(p, []).append(i)

    train_idx, val_idx = [], []
    for p, idxs in by_player.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_frac))
        if len(idxs) <= 1 or n_val == 0:
            train_idx.extend(idxs)
        else:
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Training loop — player-identity classification
# ---------------------------------------------------------------------------

def train_cnn(
    train_path: Optional[str] = None,        # kept for API compatibility (unused)
    val_path:   Optional[str] = None,        # kept for API compatibility (unused)
    device:     Optional[str] = None,
) -> CNNBranch:
    """
    Trains the CNN branch with a player-identity auxiliary task.

    The standalone over/under labels for these clips are noise (no clip→game
    mapping exists), so we use real player-ID labels instead.  The encoder's
    192-dim hidden state — what FusionModel actually consumes — is what we
    care about; classifier accuracy is just a sanity check that the encoder
    is learning something.

    Saves checkpoints/cnn_best.pt on best val player-ID accuracy.
    """
    del train_path, val_path  # unused; kept for caller compatibility

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"[cnn] training on {device}  (objective: player-identity)")

    train_ds = ClipIdentityDataset(train=True)
    eval_ds  = ClipIdentityDataset(train=False)
    if len(train_ds) == 0:
        raise RuntimeError(
            f"No clips found.  frames_root={FRAMES_DIR}  pose_root={POSE_DIR}"
        )

    train_idx, val_idx = _stratified_split(train_ds, val_frac=0.2)
    print(f"[cnn] {len(train_idx)} train clips  /  {len(val_idx)} val clips  "
          f"across {len(PLAYERS)} players")

    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset   = torch.utils.data.Subset(eval_ds,  val_idx)

    full_weights  = train_ds.class_weights()
    train_weights = full_weights[train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_idx),
                                    replacement=True)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    model     = CNNBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS, 1)
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val_acc  = -1.0
    patience_left = 10
    chance        = 1.0 / max(len(PLAYERS), 1)

    for epoch in range(1, EPOCHS + 1):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_seen = 0
        train_preds, train_labels = [], []
        for frames, poses, labels in train_loader:
            frames = frames.to(device)
            poses  = poses.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model.forward_id(frames, poses)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(frames)
            n_seen += len(frames)
            train_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        train_loss /= max(n_seen, 1)
        train_acc   = accuracy_score(train_labels, train_preds) if train_labels else 0.0

        # ── validate ─────────────────────────────────────────────────────
        if len(val_loader) > 0:
            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for frames, poses, labels in val_loader:
                    frames = frames.to(device)
                    poses  = poses.to(device)
                    labels = labels.to(device)
                    logits, _ = model.forward_id(frames, poses)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(frames)
                    n_val += len(frames)
                    val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())
            val_loss /= max(n_val, 1)
            val_acc   = accuracy_score(val_labels, val_preds)
        else:
            # Fallback: when val set is empty (every player has ≤1 clip),
            # use train acc as the checkpoint signal.
            val_loss = train_loss
            val_acc  = train_acc

        scheduler.step()

        print(
            f"  Epoch {epoch:>3}/{EPOCHS} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"(chance={chance:.3f})"
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            patience_left = 10
            ckpt = os.path.join(CHECKPOINTS_DIR, "cnn_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"    [checkpoint] saved (val_acc={val_acc:.3f})")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch} (patience=10)")
                break

    print(f"\n[cnn] best val player-ID acc: {best_val_acc:.3f}  "
          f"(chance ≈ {chance:.3f})")
    model.cpu()
    return model


if __name__ == "__main__":
    train_cnn()
