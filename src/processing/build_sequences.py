"""
build_sequences.py

Converts all_records.parquet into three split parquets used by the LSTM:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet

Each row in the output represents ONE prediction sample and contains:
  - A flattened rolling window of WINDOW_SIZE games (for baseline/LSTM use)
  - The label for the NEXT game
  - Metadata (PLAYER_NAME, GAME_DATE, PROP_LINE, LABEL)

Normalization strategy
-----------------------
Z-score normalization is computed PER PLAYER, fitted only on the training
portion of that player's data, then applied to val/test.  This prevents
leakage and handles the large scoring range between Curry (25 ppg) and
Looney (4 ppg).

Features used (FEATURE_COLS from config)
-----------------------------------------
PTS, TS_PCT, EFG_PCT, USG_PCT, OPP_DRTG, MIN

If USG_PCT or OPP_DRTG are NaN, they are imputed with the per-player
training mean before normalization.

Usage:
  python -m src.processing.build_sequences
"""

import os
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR,
    FEATURE_COLS,
    WINDOW_SIZE,
    TRAIN_END,
    VAL_START, VAL_END,
    TEST_START,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "all_records.csv")


def _chronological_split(df):
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    train = df[df["GAME_DATE"] <= TRAIN_END].copy()
    val   = df[(df["GAME_DATE"] >= VAL_START) & (df["GAME_DATE"] <= VAL_END)].copy()
    test  = df[df["GAME_DATE"] >= TEST_START].copy()
    return train, val, test


def _available_features(df):
    available = []
    for col in FEATURE_COLS:
        if col in df.columns and df[col].notna().any():
            available.append(col)
    return available


def build_sequences(df, feat_cols, player_stats, line_mean, line_std):
    """
    Builds sliding-window sequences plus two static features per sample:
      PROP_LINE_Z     : z-score of PROP_LINE relative to the training distribution.
      LINE_VS_RECENT  : (PROP_LINE - raw rolling mean PTS over window) /
                        per-player raw PTS std.  Tells the model how far above /
                        below the player's recent form the line is set — the
                        most informative single signal for over/under.
    """
    records = []
    for player, grp in df.groupby("PLAYER_NAME"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        stats = player_stats.get(player, {})

        # Raw PTS series (used to compute LINE_VS_RECENT before normalization)
        pts_raw = grp["PTS"].astype(float).fillna(stats.get("PTS", (0.0, 1.0))[0]).values
        pts_player_std = max(stats.get("PTS", (0.0, 1.0))[1], 1.0)

        X = grp[feat_cols].copy()
        for col in feat_cols:
            mean, std = stats.get(col, (X[col].mean(), 1.0))
            X[col] = X[col].fillna(mean)
            X[col] = (X[col] - mean) / (std if std > 0 else 1.0)

        X_arr  = X.values.astype(np.float32)
        labels = grp["LABEL"].values
        dates  = grp["GAME_DATE"].values
        lines  = grp["PROP_LINE"].values

        for t in range(WINDOW_SIZE, len(grp)):
            window         = X_arr[t - WINDOW_SIZE: t]
            line           = float(lines[t])
            recent_mean    = float(pts_raw[t - WINDOW_SIZE: t].mean())
            line_vs_recent = (line - recent_mean) / pts_player_std
            line_z         = (line - line_mean) / (line_std if line_std > 0 else 1.0)

            rec = {
                "PLAYER_NAME":    player,
                "GAME_DATE":      dates[t],
                "PROP_LINE":      line,
                "PROP_LINE_Z":    float(line_z),
                "LINE_VS_RECENT": float(line_vs_recent),
                "LABEL":          int(labels[t]),
            }
            for i in range(WINDOW_SIZE):
                for j, col in enumerate(feat_cols):
                    rec[f"{col}_t{i}"] = float(window[i, j])
            records.append(rec)

    return pd.DataFrame(records)


def _fit_player_stats(train_df, feat_cols):
    stats = {}
    for player, grp in train_df.groupby("PLAYER_NAME"):
        stats[player] = {}
        for col in feat_cols:
            vals = grp[col].dropna()
            stats[player][col] = (float(vals.mean()), float(vals.std(ddof=0)))
    return stats


def build():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run build_game_records.py first."
        )

    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    before = len(df)
    df = df.drop_duplicates(subset=["PLAYER_NAME", "GAME_DATE"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} duplicate rows")
    print(f"  {len(df)} labeled records")

    feat_cols = _available_features(df)
    print(f"  Features used: {feat_cols}")

    train_raw, val_raw, test_raw = _chronological_split(df)
    print(f"  Raw splits — train: {len(train_raw)}  val: {len(val_raw)}  test: {len(test_raw)}")

    print("\nFitting normalization stats on training data...")
    player_stats = _fit_player_stats(train_raw, feat_cols)
    line_mean = float(train_raw["PROP_LINE"].mean())
    line_std  = float(train_raw["PROP_LINE"].std(ddof=0))
    print(f"  PROP_LINE z-score: mean={line_mean:.2f}  std={line_std:.2f}")

    print("Building sequence windows...")
    train_seq = build_sequences(train_raw, feat_cols, player_stats, line_mean, line_std)
    val_seq   = build_sequences(val_raw,   feat_cols, player_stats, line_mean, line_std)
    test_seq  = build_sequences(test_raw,  feat_cols, player_stats, line_mean, line_std)

    for name, seq in [("train", train_seq), ("val", val_seq), ("test", test_seq)]:
        over_pct = (seq["LABEL"] == 1).mean() * 100 if len(seq) > 0 else 0
        out = os.path.join(PROCESSED_DIR, f"{name}.csv")
        seq.to_csv(out, index=False)
        print(f"  {name}: {len(seq):4d} sequences  OVER={over_pct:.1f}%  -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    build()
