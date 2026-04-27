"""
fusion.py

End-to-end multimodal fusion model combining the LSTM stats branch with the
CNN vision branch to predict player points prop over/under.

Architecture
------------
  LSTM branch  : pretrained LSTMBranch → hidden (batch, 128)
  CNN branch   : pretrained CNNBranch  → hidden (batch, 192)

  Learnable branch projections:
    lstm_proj  : Linear(128, 128)
    cnn_proj   : Linear(192, 128)

  Learnable branch weights (initialised to favour LSTM ~65/35):
    raw_weights : nn.Parameter([0.619, 0.381])  → softmax → [w_lstm, w_cnn]
    fused       : w_lstm * lstm_proj + w_cnn * cnn_proj   (batch, 128)

  Classifier head (two FC layers per proposal):
    Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 1)

  Optimizer: Adam
    - Fusion head + projections + branch weights: lr = 1e-3
    - Branch encoders (fine-tune):                lr = 1e-4

Branch weight reporting
-----------------------
softmax(raw_weights) is printed each epoch so the learned contribution of
each branch is visible throughout training, as required by the proposal.

Missing CNN data
----------------
If pose or frame data is absent for a sample, pose_tensor is all-zeros.
This lets the model train on stats-only samples — the branch weight for CNN
naturally learns to discount when its inputs carry no signal.

Usage
-----
  python scripts/run_fusion.py
  # or standalone:
  python -m src.models.fusion
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DROPOUT,
    EPOCHS,
    FEATURE_COLS,
    FRAMES_DIR,
    LEARNING_RATE,
    PLAYERS,
    POSE_DIR,
    PROCESSED_DIR,
    RAW_GAMELOGS_DIR,
    WINDOW_SIZE,
)
from src.models.lstm import LSTMBranch, STATIC_FEATURES, STATIC_DIM
from src.models.cnn import (
    CNNBranch,
    N_FRAMES_PER_CLIP,
    POSE_DIM as CNN_POSE_DIM,
    _pose_feature_columns,
    _player_slug as _cnn_player_slug,
    _row_to_pose_vec,
    build_player_pose_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LSTM_HIDDEN_DIM = 128   # hidden_size=64 bidirectional → 64*2
CNN_HIDDEN_DIM  = 192   # visual 128 + pose 64
PROJ_DIM        = 128   # both branches projected to this before weighted sum
POSE_DIM        = CNN_POSE_DIM   # 267 — must match cnn.py

# Softmax of these inits gives approximately [0.61, 0.39] → LSTM majority
_LSTM_INIT_LOGIT = 0.45
_CNN_INIT_LOGIT  = 0.00

_frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


def _build_game_id_lookup(gamelogs_dir: str) -> dict[tuple[str, str], str]:
    """
    Builds {(player_name, game_date_str): game_id} from raw game-log CSVs.
    Used to map stats-CSV rows to CNN pose features (which are keyed by game_id).
    """
    lookup: dict[tuple[str, str], str] = {}
    logs_path = Path(gamelogs_dir)
    if not logs_path.exists():
        return lookup
    for csv_file in logs_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, usecols=lambda c: c in {"PLAYER_NAME", "GAME_DATE", "Game_ID", "GAME_ID"})
        except Exception:
            continue
        # Normalise column name
        if "Game_ID" in df.columns:
            df = df.rename(columns={"Game_ID": "GAME_ID"})
        if "PLAYER_NAME" not in df.columns or "GAME_DATE" not in df.columns or "GAME_ID" not in df.columns:
            continue
        for _, row in df.iterrows():
            key = (str(row["PLAYER_NAME"]), str(row["GAME_DATE"])[:10])
            lookup[key] = str(row["GAME_ID"]).zfill(10)
    return lookup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """
    Each sample contains:
      seq_tensor   : (WINDOW_SIZE, num_features) — reshaped from flattened stats CSV
      pose_tensor  : (POSE_DIM,)  — pose features for the game when available,
                                    otherwise the player's average pose across
                                    all their clips (a kinetic fingerprint).
                                    All-zeros only if the player has no pose data.
      frames_tensor: (N_FRAMES_PER_CLIP, 3, 224, 224)
                                  — multi-frame stack for the game when available,
                                    otherwise from any of the player's clips.
                                    Zero-tensor only if the player has no clips.
      label        : float 0 or 1

    Per-player fallback rationale
    -----------------------------
    Today, every clip in FRAMES_DIR is a YouTube highlight (no real game IDs),
    so a strict (player, game_id) join leaves every sample with zero pose and
    zero frames — which is exactly what made the CNN branch useless.  The
    fallback gives the CNN a *player-stable* visual+pose signature for every
    sample.  The kinetic features still vary per player, which is what fusion
    can exploit alongside the LSTM stat history.
    """

    def __init__(
        self,
        labels_csv:   str,
        frames_root:  Optional[str] = None,
        pose_root:    Optional[str] = None,
        gamelogs_dir: Optional[str] = None,
        transform:    transforms.Compose = _frame_transform,
    ):
        self.transform   = transform
        self.frames_root = Path(frames_root  or FRAMES_DIR)
        self.pose_root   = Path(pose_root    or POSE_DIR)

        # Build game_date → game_id lookup from raw game logs
        self._game_id_map = _build_game_id_lookup(gamelogs_dir or RAW_GAMELOGS_DIR)

        df = pd.read_csv(labels_csv)
        self._n_features = len(FEATURE_COLS)

        # Per-game pose lookup (game-keyed) and per-player fallback pose vectors.
        self._pose_map     = self._load_pose_map()
        self._player_pose  = build_player_pose_table(str(self.pose_root))

        # Per-player clip-dir cache for frame fallback.
        self._player_clips = self._build_player_clip_index()

        self._missing_static = [c for c in STATIC_FEATURES if c not in df.columns]
        if self._missing_static:
            print(f"[fusion] WARNING: {labels_csv} missing static columns "
                  f"{self._missing_static}; filling with zeros. Re-run "
                  f"`python -m src.processing.build_sequences` to enable "
                  f"prop-line features.")

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Path], int]] = []
        for _, row in df.iterrows():
            seq    = self._extract_seq(row)
            static = np.array(
                [float(row[c]) if c in row else 0.0 for c in STATIC_FEATURES],
                dtype=np.float32,
            )
            player_name = str(row["PLAYER_NAME"])
            game_date   = str(row["GAME_DATE"])[:10]
            label       = int(row["LABEL"])

            game_id  = self._game_id_map.get((player_name, game_date))
            slug     = _cnn_player_slug(player_name)
            pose_vec = self._get_pose(slug, game_id)
            clip_dir = self._get_clip_dir(slug, game_id)

            self.samples.append((seq, static, pose_vec, clip_dir, label))

    # ------------------------------------------------------------------
    def _extract_seq(self, row: pd.Series) -> np.ndarray:
        """Reshape flattened window columns → (WINDOW_SIZE, num_features)."""
        vals = []
        for t in range(WINDOW_SIZE):
            step = [float(row.get(f"{feat}_t{t}", 0.0)) for feat in FEATURE_COLS]
            vals.append(step)
        return np.array(vals, dtype=np.float32)   # (W, F)

    def _load_pose_map(self) -> dict[str, dict[str, np.ndarray]]:
        """{player_slug: {game_id: pose_vec}}  — keyed on real game IDs only."""
        pose_map: dict[str, dict[str, np.ndarray]] = {}
        if not self.pose_root.exists():
            return pose_map
        for player in PLAYERS:
            slug     = _cnn_player_slug(player)
            pose_csv = self.pose_root / f"{slug}.csv"
            if not pose_csv.exists():
                continue
            pdf = pd.read_csv(pose_csv)
            if "game_id" not in pdf.columns:
                continue
            pdf["game_id"] = pdf["game_id"].astype(str)
            feat_cols = _pose_feature_columns(pdf)
            if not feat_cols:
                continue
            game_rows = pdf[pdf["game_id"].str.lower() != "yt"]
            if game_rows.empty:
                continue
            # Average all clips for the same real game
            sub: dict[str, np.ndarray] = {}
            for gid, group in game_rows.groupby("game_id"):
                vecs = np.stack(
                    [_row_to_pose_vec(r, feat_cols) for _, r in group.iterrows()],
                    axis=0,
                )
                sub[str(gid).zfill(10)] = vecs.mean(axis=0).astype(np.float32)
            pose_map[slug] = sub
        return pose_map

    def _build_player_clip_index(self) -> dict[str, list[Path]]:
        """{player_slug: [clip_dir, ...]} listing every clip dir for fallback."""
        idx: dict[str, list[Path]] = {}
        if not self.frames_root.exists():
            return idx
        for player_dir in self.frames_root.iterdir():
            if not player_dir.is_dir():
                continue
            clips = [d for d in player_dir.iterdir()
                     if d.is_dir() and any(d.glob("*.jpg"))]
            if clips:
                idx[player_dir.name] = sorted(clips)
        return idx

    def _get_pose(self, slug: str, game_id: Optional[str]) -> np.ndarray:
        # 1) Real game-id match
        if game_id and slug in self._pose_map:
            vec = self._pose_map[slug].get(str(game_id).zfill(10))
            if vec is not None:
                return self._fit(vec)
        # 2) Per-player kinetic-fingerprint fallback
        if slug in self._player_pose:
            return self._fit(self._player_pose[slug])
        # 3) Last resort
        return np.zeros(POSE_DIM, dtype=np.float32)

    @staticmethod
    def _fit(vec: np.ndarray) -> np.ndarray:
        if len(vec) >= POSE_DIM:
            return vec[:POSE_DIM].astype(np.float32)
        padded = np.zeros(POSE_DIM, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded

    def _get_clip_dir(self, slug: str, game_id: Optional[str]) -> Optional[Path]:
        # 1) Real game-id-prefixed clip
        if game_id:
            player_dir = self.frames_root / slug
            if player_dir.exists():
                for clip_dir in player_dir.iterdir():
                    if clip_dir.is_dir() and clip_dir.name.startswith(str(game_id)):
                        if any(clip_dir.glob("*.jpg")):
                            return clip_dir
        # 2) Any clip for this player (deterministic: pick first by name)
        clips = self._player_clips.get(slug)
        if clips:
            return clips[0]
        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, static, pose_vec, clip_dir, label = self.samples[idx]

        seq_t    = torch.tensor(seq,    dtype=torch.float32)    # (W, F)
        static_t = torch.tensor(static, dtype=torch.float32)    # (STATIC_DIM,)
        pose_t   = torch.tensor(pose_vec, dtype=torch.float32)  # (POSE_DIM,)

        if clip_dir is not None and clip_dir.exists():
            frames = sorted(clip_dir.glob("*.jpg"))
            if frames:
                idxs = np.linspace(0, len(frames) - 1, N_FRAMES_PER_CLIP).round().astype(int)
                imgs = [self.transform(Image.open(frames[i]).convert("RGB"))
                        for i in idxs]
                frames_t = torch.stack(imgs, dim=0)              # (T, 3, 224, 224)
            else:
                frames_t = torch.zeros(N_FRAMES_PER_CLIP, 3, 224, 224, dtype=torch.float32)
        else:
            frames_t = torch.zeros(N_FRAMES_PER_CLIP, 3, 224, 224, dtype=torch.float32)

        label_t = torch.tensor(float(label), dtype=torch.float32)
        return seq_t, static_t, pose_t, frames_t, label_t

    def class_weights(self) -> torch.Tensor:
        labels = np.array([s[4] for s in self.samples])
        counts = np.bincount(labels.astype(int))
        w_per_class = 1.0 / counts
        return torch.tensor([w_per_class[int(l)] for l in labels], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    Multimodal fusion of LSTM stats branch + CNN vision branch.

    Branch contributions are controlled by a learnable softmax weight pair
    initialised to favour LSTM (~61 %) over CNN (~39 %).  The weights are
    updated by the Adam optimizer end-to-end and reported after every epoch.

    forward() returns (logit, branch_weights) so callers can log the weights.
    """

    def __init__(
        self,
        lstm_ckpt: Optional[str] = None,
        cnn_ckpt:  Optional[str] = None,
        dropout:   float = DROPOUT,
    ):
        super().__init__()

        # ── Load pretrained branches ─────────────────────────────────────
        self.lstm_branch = LSTMBranch()
        self.cnn_branch  = CNNBranch()

        lstm_ckpt = lstm_ckpt or os.path.join(CHECKPOINTS_DIR, "lstm_best.pt")
        cnn_ckpt  = cnn_ckpt  or os.path.join(CHECKPOINTS_DIR, "cnn_best.pt")

        if os.path.exists(lstm_ckpt):
            ckpt_state = torch.load(lstm_ckpt, map_location="cpu")
            cur_state  = self.lstm_branch.state_dict()
            compatible = {k: v for k, v in ckpt_state.items()
                          if k in cur_state and cur_state[k].shape == v.shape}
            dropped = sorted(set(ckpt_state) - set(compatible))
            self.lstm_branch.load_state_dict(compatible, strict=False)
            if dropped:
                print(f"[fusion] loaded LSTM checkpoint with {len(dropped)} mismatched "
                      f"keys skipped (stale architecture): {lstm_ckpt}")
            else:
                print(f"[fusion] loaded LSTM checkpoint: {lstm_ckpt}")
        else:
            print(f"[fusion] warning — LSTM checkpoint not found at {lstm_ckpt}, using random init")

        if os.path.exists(cnn_ckpt):
            ckpt_state = torch.load(cnn_ckpt, map_location="cpu")
            # Drop any tensors whose shape doesn't match the current architecture
            # (e.g. stale checkpoints from before the pose-feature expansion).
            cur_state = self.cnn_branch.state_dict()
            compatible = {k: v for k, v in ckpt_state.items()
                          if k in cur_state and cur_state[k].shape == v.shape}
            dropped = sorted(set(ckpt_state) - set(compatible))
            self.cnn_branch.load_state_dict(compatible, strict=False)
            if dropped:
                print(f"[fusion] loaded CNN checkpoint with {len(dropped)} mismatched "
                      f"keys skipped (stale architecture): {cnn_ckpt}")
            else:
                print(f"[fusion] loaded CNN checkpoint: {cnn_ckpt}")
        else:
            print(f"[fusion] warning — CNN checkpoint not found at {cnn_ckpt}, using random init")

        # ── Branch projections (both → PROJ_DIM = 128) ───────────────────
        self.lstm_proj = nn.Linear(LSTM_HIDDEN_DIM, PROJ_DIM)
        self.cnn_proj  = nn.Linear(CNN_HIDDEN_DIM,  PROJ_DIM)

        # ── Learnable branch weights ──────────────────────────────────────
        # softmax([0.45, 0.00]) ≈ [0.611, 0.389] → LSTM majority (~61 %)
        self.raw_weights = nn.Parameter(
            torch.tensor([_LSTM_INIT_LOGIT, _CNN_INIT_LOGIT], dtype=torch.float32)
        )

        # ── Static-feature MLP (prop-line context) ───────────────────────
        # Shared with what LSTMBranch uses internally, but the fusion model
        # also gets a direct path to these features so the cross-modal
        # classifier can condition on the prop line explicitly.
        self.static_mlp = nn.Sequential(
            nn.Linear(STATIC_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # ── Classifier head ──────────────────────────────────────────────
        head_in = PROJ_DIM + 32  # fused + static
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @property
    def branch_weights(self) -> torch.Tensor:
        """Returns softmax-normalised [w_lstm, w_cnn] (detached for logging)."""
        return F.softmax(self.raw_weights, dim=0).detach()

    def forward(
        self,
        seq:    torch.Tensor,   # (batch, WINDOW_SIZE, num_features)
        static: torch.Tensor,   # (batch, STATIC_DIM)
        pose:   torch.Tensor,   # (batch, POSE_DIM)
        frame:  torch.Tensor,   # (batch, T, 3, 224, 224) or (batch, 3, 224, 224)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logit          : (batch, 1)
        branch_weights : (2,)  [w_lstm, w_cnn]
        """
        # Extract hidden representations from each branch
        _, lstm_hidden = self.lstm_branch(seq, static)   # (batch, 128) — pure LSTM hidden
        _, cnn_hidden  = self.cnn_branch(frame, pose)    # (batch, 192)

        # Project to common dimension
        h_lstm = self.lstm_proj(lstm_hidden)   # (batch, 128)
        h_cnn  = self.cnn_proj(cnn_hidden)     # (batch, 128)

        # Weighted combination (learned end-to-end)
        w = F.softmax(self.raw_weights, dim=0)
        fused = w[0] * h_lstm + w[1] * h_cnn   # (batch, 128)

        # Inject prop-line context into the classifier head
        s = self.static_mlp(static)            # (batch, 32)
        logit = self.classifier(torch.cat([fused, s], dim=-1))   # (batch, 1)
        return logit, w.detach()


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _make_loader(csv_path: str, shuffle: bool, weighted: bool) -> DataLoader:
    ds = FusionDataset(csv_path)
    if weighted:
        weights = ds.class_weights()
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fusion(
    train_path: Optional[str] = None,
    val_path:   Optional[str] = None,
    device:     Optional[str] = None,
) -> FusionModel:
    """
    Trains the fusion model end-to-end using Adam.

    Optimizer uses two parameter groups:
      - Fusion head, projections, branch weights : lr = LEARNING_RATE (1e-3)
      - Branch encoders (fine-tune gently)       : lr = LEARNING_RATE / 10

    Saves checkpoints/fusion_best.pt on val_acc improvement.
    Reports learned branch weights [w_lstm, w_cnn] after each epoch.
    """
    train_path = os.path.abspath(train_path or os.path.join(PROCESSED_DIR, "train.csv"))
    val_path   = os.path.abspath(val_path   or os.path.join(PROCESSED_DIR, "val.csv"))

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"[fusion] training on {device}")

    train_loader = _make_loader(train_path, shuffle=True,  weighted=True)
    val_loader   = _make_loader(val_path,   shuffle=False, weighted=False)

    model = FusionModel().to(device)

    # Two-speed Adam: fusion head fast, branch encoders slow (avoid forgetting)
    branch_params = (
        list(model.lstm_branch.parameters()) +
        list(model.cnn_branch.parameters())
    )
    head_params = (
        list(model.lstm_proj.parameters()) +
        list(model.cnn_proj.parameters()) +
        list(model.classifier.parameters()) +
        list(model.static_mlp.parameters()) +
        [model.raw_weights]
    )
    optimizer = torch.optim.Adam([
        {"params": head_params,   "lr": LEARNING_RATE},
        {"params": branch_params, "lr": LEARNING_RATE / 10},
    ])
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for seq, static, pose, frame, labels in train_loader:
            seq, static, pose, frame, labels = (
                seq.to(device), static.to(device), pose.to(device),
                frame.to(device), labels.to(device),
            )
            optimizer.zero_grad()
            logit, _ = model(seq, static, pose, frame)
            loss = criterion(logit.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(seq)
        train_loss /= len(train_loader.dataset)

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for seq, static, pose, frame, labels in val_loader:
                seq, static, pose, frame, labels = (
                    seq.to(device), static.to(device), pose.to(device),
                    frame.to(device), labels.to(device),
                )
                logit, _ = model(seq, static, pose, frame)
                loss = criterion(logit.squeeze(1), labels)
                val_loss += loss.item() * len(seq)
                probs = torch.sigmoid(logit.squeeze(1))
                preds = (probs >= 0.5).long()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().long().tolist())
                all_probs.extend(probs.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc  = accuracy_score(all_labels, all_preds)
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = float("nan")

        w = model.branch_weights.cpu()
        print(
            f"  Epoch {epoch:>3}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"w_lstm={w[0]:.3f}  w_cnn={w[1]:.3f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt = os.path.join(CHECKPOINTS_DIR, "fusion_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"    [checkpoint] saved (val_auc={val_auc:.4f})")

    # ── Final branch weight report ────────────────────────────────────────
    w = model.branch_weights.cpu()
    print(f"\n[fusion] best val AUC : {best_val_auc:.4f}")
    print(f"[fusion] final branch weights:")
    print(f"  LSTM  : {w[0]*100:.1f}%")
    print(f"  CNN   : {w[1]*100:.1f}%")

    return model


if __name__ == "__main__":
    train_fusion()
