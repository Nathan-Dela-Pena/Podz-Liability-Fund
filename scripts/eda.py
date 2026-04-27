"""
CS 489 Phase 2 – Exploratory Data Analysis
Generates tables and plots for the Phase 2 report.
Run:  python scripts/eda.py
Outputs saved to data/eda_figures/
"""

import os
import glob
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

OUT_DIR = "data/eda_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config (matches config.py) ─────────────────────────────────────────────
TRAIN_START, TRAIN_END = "2023-10-24", "2025-02-28"
VAL_START,   VAL_END   = "2025-03-01", "2025-04-13"
TEST_START,  TEST_END  = "2025-10-21", "2026-04-06"

# ── 1. Load all game-log CSVs ──────────────────────────────────────────────
files = glob.glob("data/raw/game_logs/*.csv")
dfs   = [pd.read_csv(f) for f in files]
df    = pd.concat(dfs, ignore_index=True)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.drop_duplicates(subset=["PLAYER_NAME", "GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)

# ── 2. Dataset Summary ─────────────────────────────────────────────────────
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total records      : {len(df)}")
print(f"Unique players     : {df['PLAYER_NAME'].nunique()}")
print(f"Seasons            : {sorted(df['SEASON'].unique())}")
print(f"Date range         : {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
print(f"Unique game dates  : {df['GAME_DATE'].nunique()}")
print()
print("Records per season:")
print(df.groupby("SEASON").size().to_string())
print()
print("Records per player:")
print(df.groupby("PLAYER_NAME").size().sort_values(ascending=False).to_string())
print()

# ── 3. Missing values ──────────────────────────────────────────────────────
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df[["PTS", "TS_PCT", "EFG_PCT", "MIN", "USG_PCT"]].isnull().sum().to_string())
print()

# ── 4. Chronological split ─────────────────────────────────────────────────
train_raw = df[(df["GAME_DATE"] >= TRAIN_START) & (df["GAME_DATE"] <= TRAIN_END)]
val_raw   = df[(df["GAME_DATE"] >= VAL_START)   & (df["GAME_DATE"] <= VAL_END)]
test_raw  = df[(df["GAME_DATE"] >= TEST_START)  & (df["GAME_DATE"] <= TEST_END)]

print("=" * 60)
print("CHRONOLOGICAL DATA SPLIT (raw game logs)")
print("=" * 60)
for name, split in [("Train", train_raw), ("Val", val_raw), ("Test", test_raw)]:
    if len(split):
        print(f"{name:6s}: {len(split):5d} records  "
              f"({split['GAME_DATE'].min().date()} → {split['GAME_DATE'].max().date()})")
    else:
        print(f"{name:6s}: 0 records")
print()

# ── 5. Feature Statistics ──────────────────────────────────────────────────
print("=" * 60)
print("FEATURE STATISTICS (all records)")
print("=" * 60)
print(df[["PTS", "TS_PCT", "EFG_PCT", "USG_PCT", "MIN"]].describe().round(3).to_string())
print()

print("PTS per player (mean ± std):")
pts_stats = df.groupby("PLAYER_NAME")["PTS"].agg(["mean","std"]).round(2)
pts_stats.columns = ["Mean PTS", "Std PTS"]
print(pts_stats.sort_values("Mean PTS", ascending=False).to_string())
print()

# ── 6. Minutes subgroup ────────────────────────────────────────────────────
low_min  = df[df["MIN"] < 20]
high_min = df[df["MIN"] >= 20]
print("=" * 60)
print("SUBGROUP: Minutes Played")
print("=" * 60)
print(f"Games < 20 min  : {len(low_min):5d}  ({100*len(low_min)/len(df):.1f}%)")
print(f"Games >= 20 min : {len(high_min):5d}  ({100*len(high_min)/len(df):.1f}%)")
print()

# ── 7. FIGURE 1 – PTS distribution per player ─────────────────────────────
order = df.groupby("PLAYER_NAME")["PTS"].mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(16, 5))
sns.boxplot(data=df, x="PLAYER_NAME", y="PTS", order=order, ax=ax,
            hue="PLAYER_NAME", palette="Blues_d", legend=False, fliersize=2)
ax.set_title("Points Distribution per Player — All Seasons (2023-26)", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Points per Game")
ax.tick_params(axis="x", rotation=40)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_pts_distribution.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/fig1_pts_distribution.png")

# ── 8. FIGURE 2 – Records per player per season ───────────────────────────
pivot = df.groupby(["PLAYER_NAME", "SEASON"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 4))
pivot.plot(kind="bar", ax=ax, color=["#4c9be8", "#e87c4c", "#6bcb77"])
ax.set_title("Game Records per Player per Season", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Game Records")
ax.tick_params(axis="x", rotation=40)
ax.legend(title="Season")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_records_per_player_season.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/fig2_records_per_player_season.png")

# ── 9. FIGURE 3 – Feature correlation heatmap ─────────────────────────────
corr_cols = ["PTS", "TS_PCT", "EFG_PCT", "USG_PCT", "MIN"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/fig3_correlation_heatmap.png")

# ── 10. FIGURE 4 – Rolling mean PTS with train/val/test shading ───────────
top_players = ["Stephen Curry", "Victor Wembanyama", "De'Aaron Fox", "Jonathan Kuminga"]
fig, ax = plt.subplots(figsize=(15, 4))
for player in top_players:
    sub = df[df["PLAYER_NAME"] == player].sort_values("GAME_DATE")
    ax.plot(sub["GAME_DATE"], sub["PTS"].rolling(5, min_periods=1).mean(),
            label=player, linewidth=1.5)

ax.axvspan(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END), alpha=0.07, color="green")
ax.axvspan(pd.Timestamp(VAL_START),   pd.Timestamp(VAL_END),   alpha=0.12, color="orange")
ax.axvspan(pd.Timestamp(TEST_START),  pd.Timestamp(TEST_END),  alpha=0.07, color="red")

from matplotlib.patches import Patch as MPatch
legend_handles = [ax.get_lines()[i] for i in range(len(top_players))] + [
    MPatch(color="green",  alpha=0.3, label="Train"),
    MPatch(color="orange", alpha=0.4, label="Val"),
    MPatch(color="red",    alpha=0.3, label="Test"),
]
ax.legend(handles=legend_handles, fontsize=8, ncol=4)
ax.set_title("5-Game Rolling Mean PTS — Top Scorers", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Rolling Mean PTS")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_rolling_pts.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/fig4_rolling_pts.png")

# ── 11. FIGURE 5 – Minutes distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["MIN"], bins=30, color="#4c9be8", edgecolor="white", linewidth=0.4)
ax.axvline(20, color="red", linestyle="--", linewidth=1.4, label="20-min threshold")
ax.set_title("Distribution of Minutes Played", fontsize=13)
ax.set_xlabel("Minutes")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_minutes_distribution.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/fig5_minutes_distribution.png")

# ── 12. FIGURE 6 – Shooting efficiency distributions ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["TS_PCT"].clip(0, 1.05), bins=40, color="#4c9be8", edgecolor="white")
axes[0].set_title("True Shooting % Distribution")
axes[0].set_xlabel("TS%")
axes[1].hist(df["EFG_PCT"].clip(0, 1.05), bins=40, color="#e87c4c", edgecolor="white")
axes[1].set_title("Effective FG% Distribution")
axes[1].set_xlabel("eFG%")
plt.suptitle("Shooting Efficiency Distributions", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig6_shooting_distributions.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/fig6_shooting_distributions.png")

# ── 13. Summary table ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY TABLE (for report)")
print("=" * 60)
rows = [
    ("Total game-log records",    len(df)),
    ("Unique players",            df["PLAYER_NAME"].nunique()),
    ("Seasons covered",           "2023-24, 2024-25, 2025-26"),
    ("Date range",                f"{df['GAME_DATE'].min().date()} – {df['GAME_DATE'].max().date()}"),
    ("Training records",          len(train_raw)),
    ("Validation records",        len(val_raw)),
    ("Test records",              len(test_raw)),
    ("Features",                  "PTS, TS%, eFG%, USG%, OPP_DRTG, MIN"),
]
for k, v in rows:
    print(f"  {k:<35} {v}")
print()

# ── 14. Labeled dataset section ───────────────────────────────────────────
labeled_path = "data/processed/all_records.csv"
if not os.path.exists(labeled_path):
    print(f"[SKIP] {labeled_path} not found — run scripts/run_processing.py first")
else:
    print("=" * 60)
    print("LABELED DATASET (with prop lines)")
    print("=" * 60)

    labeled = pd.read_csv(labeled_path)
    labeled["GAME_DATE"] = pd.to_datetime(labeled["GAME_DATE"])

    print(f"Total labeled records  : {len(labeled)}")
    print(f"OVER  (label=1)        : {(labeled['LABEL']==1).sum()} ({(labeled['LABEL']==1).mean()*100:.1f}%)")
    print(f"UNDER (label=0)        : {(labeled['LABEL']==0).sum()} ({(labeled['LABEL']==0).mean()*100:.1f}%)")
    print(f"Class balance          : {(labeled['LABEL']==1).mean()*100:.1f}% OVER")
    print()
    print("Prop line coverage per player:")
    cov = labeled.groupby("PLAYER_NAME")["LABEL"].count().sort_values(ascending=False)
    print(cov.to_string())
    print()

    # ── FIGURE 7 – Label distribution ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    over_c, under_c = (labeled["LABEL"]==1).sum(), (labeled["LABEL"]==0).sum()
    axes[0].bar(["Under (0)", "Over (1)"], [under_c, over_c],
                color=["#e87c4c", "#4c9be8"], edgecolor="white")
    axes[0].set_title("Overall Label Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([under_c, over_c]):
        axes[0].text(i, v + 2, str(v), ha="center", fontsize=11)

    piv = labeled.groupby(["PLAYER_NAME","LABEL"]).size().unstack(fill_value=0)
    piv.columns = ["Under", "Over"]
    piv = piv.reindex(cov.index)
    piv.plot(kind="bar", ax=axes[1], color=["#e87c4c","#4c9be8"])
    axes[1].set_title("Over/Under Count per Player")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].legend(title="Label")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig7_label_distribution.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/fig7_label_distribution.png")

    # ── FIGURE 8 – Actual PTS vs prop line scatter ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = labeled["LABEL"].map({1: "#4c9be8", 0: "#e87c4c"})
    ax.scatter(labeled["PROP_LINE"], labeled["PTS"], c=colors, alpha=0.4, s=14)
    lo = min(labeled["PROP_LINE"].min(), labeled["PTS"].min()) - 1
    hi = max(labeled["PROP_LINE"].max(), labeled["PTS"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="PTS = Line")
    legend_els = [Patch(color="#4c9be8", label="Over"),
                  Patch(color="#e87c4c", label="Under"),
                  plt.Line2D([0],[0], color="k", linestyle="--", label="PTS=Line")]
    ax.legend(handles=legend_els)
    ax.set_xlabel("Prop Line (points)")
    ax.set_ylabel("Actual PTS")
    ax.set_title("Actual PTS vs. Prop Line", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig8_pts_vs_propline.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/fig8_pts_vs_propline.png")

    # ── FIGURE 9 – OPP_DRTG distribution ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(labeled["OPP_DRTG"], bins=25, color="#7c6be8", edgecolor="white")
    ax.set_title("Opponent Defensive Rating Distribution", fontsize=13)
    ax.set_xlabel("OPP_DRTG (points per 100 possessions)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig9_opp_drtg.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/fig9_opp_drtg.png")

    # ── FIGURE 10 – Sequence samples per split per player ─────────────────
    for split_file in ["train.csv", "val.csv", "test.csv"]:
        if not os.path.exists(f"data/processed/{split_file}"):
            print(f"[SKIP] data/processed/{split_file} not found")
            break
    else:
        train_seq = pd.read_csv("data/processed/train.csv")
        val_seq   = pd.read_csv("data/processed/val.csv")
        test_seq  = pd.read_csv("data/processed/test.csv")

        split_counts = pd.DataFrame({
            "Train": train_seq.groupby("PLAYER_NAME").size(),
            "Val":   val_seq.groupby("PLAYER_NAME").size(),
            "Test":  test_seq.groupby("PLAYER_NAME").size(),
        }).fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(14, 4))
        split_counts.plot(kind="bar", stacked=True, ax=ax,
                          color=["#4c9be8", "#e8a84c", "#e87c4c"])
        ax.set_title("Sequence Samples per Player per Split", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Sequences")
        ax.tick_params(axis="x", rotation=40)
        ax.legend(title="Split")
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig10_sequences_per_split.png", dpi=150)
        plt.close()
        print(f"Saved {OUT_DIR}/fig10_sequences_per_split.png")

        # ── FIGURE 11 – USG_PCT per player ────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        usg_order = labeled.groupby("PLAYER_NAME")["USG_PCT"].mean().sort_values(ascending=False).index
        sns.boxplot(data=labeled, x="PLAYER_NAME", y="USG_PCT", order=usg_order,
                    hue="PLAYER_NAME", palette="Greens_d", legend=False, ax=ax, fliersize=2)
        ax.set_title("Usage Rate (USG%) per Player", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("USG%")
        ax.tick_params(axis="x", rotation=40)
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig11_usg_distribution.png", dpi=150)
        plt.close()
        print(f"Saved {OUT_DIR}/fig11_usg_distribution.png")

        # ── Print final summary ────────────────────────────────────────────
        print()
        print("=" * 60)
        print("FINAL SUMMARY FOR PHASE 2 REPORT")
        print("=" * 60)
        print(f"  Raw game-log records     : {len(df)}")
        print(f"  Labeled records          : {len(labeled)} ({len(labeled)/len(df)*100:.1f}% coverage)")
        print(f"  Class balance            : {(labeled['LABEL']==1).mean()*100:.1f}% OVER / {(labeled['LABEL']==0).mean()*100:.1f}% UNDER")
        print(f"  Train sequences          : {len(train_seq)}")
        print(f"  Val sequences            : {len(val_seq)}")
        print(f"  Test sequences           : {len(test_seq)}")
        print()

        baseline = "checkpoints/baseline_results.json"
        if os.path.exists(baseline):
            import json as _json
            res = _json.load(open(baseline))
            print("  Baseline (Logistic Regression) Results:")
            for split in ["train", "val", "test"]:
                r = res.get(split, {})
                print(f"    {split:<6s}: Acc={r.get('accuracy',0)*100:.2f}%  "
                      f"F1={r.get('f1_macro',0):.4f}  AUROC={r.get('auroc',0):.4f}")

print()
print(f"All figures saved to: {OUT_DIR}")
