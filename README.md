# Podz Liability Fund


A multimodal NBA player prop prediction system that predicts **OVER / UNDER** on nightly points props for Golden State Warriors and San Antonio Spurs players. The system combines five models: a bidirectional LSTM (rolling game-log stats), a MobileNetV2 CNN (pose keypoints from YouTube highlights), a late-fusion model (LSTM + CNN), an XGBoost gradient boosting model, and a per-player logistic regression ensemble.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
│  nba_api (game logs)   The Odds API (prop lines)   YouTube     │
└────────────┬──────────────────┬────────────────────────┬────────┘
             │                  │                        │
             ▼                  ▼                        ▼
    build_game_records.py   fetch_odds.py    fetch_youtube_clips.py
    build_sequences.py                       extract_frames.py
             │                               extract_pose.py
             ▼                                     │
    data/processed/                          data/pose/
    train/val/test.csv                       <player>.csv (267-dim)
             │                                     │
    ┌────────▼────────┐               ┌────────────▼────────────┐
    │  Branch 1: LSTM │               │   Branch 2: CNN         │
    │  BiLSTM(64)     │               │   MobileNetV2 + Pose    │
    │  + static MLP   │               │   → hidden (192)        │
    │  → hidden (128) │               │                         │
    └────────┬────────┘               └────────────┬────────────┘
             └──────────────┬──────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │    Fusion Model         │
              │  Learned softmax weight │
              │  ~78% LSTM / ~22% CNN   │
              │  → LayerNorm → Linear   │
              └─────────────────────────┘

    ┌──────────────────────┐   ┌──────────────────────────────┐
    │  XGBoost             │   │  Per-Player Ensemble         │
    │  Flat window+static  │   │  Per-player LogisticReg      │
    │  gradient boosting   │   │  + global fallback           │
    └──────────────────────┘   └──────────────────────────────┘
```

---

## Players Tracked

**Golden State Warriors:**
Stephen Curry, Draymond Green, Jonathan Kuminga, Moses Moody, Gary Payton II, Kevon Looney, Brandin Podziemski, Gui Santos, Quinten Post, Will Richard, Pat Spencer, Jimmy Butler, Seth Curry, Al Horford, De'Anthony Melton, Kristaps Porzingis, Andrew Wiggins

**San Antonio Spurs:**
Victor Wembanyama, De'Aaron Fox, Keldon Johnson, Stephon Castle, Dylan Harper

---

## Dataset

| Split | Date Range          | Sequences |
|-------|---------------------|-----------|
| Train | Oct 2023 – Feb 2025 | 654       |
| Val   | Mar 2025 – Apr 2025 | 85        |
| Test  | Oct 2025 – Apr 2026 | 701       |

**Splits are chronological (no shuffling)** — test set is never touched during training to prevent look-ahead leakage.

**Input features (8 × 5-game rolling window = 40-dim):**
`PTS`, `TS%`, `eFG%`, `USG%`, `OPP_DRTG`, `MIN`, `HOME_AWAY`, `DAYS_REST`

- `HOME_AWAY` — 1 if home game, 0 if away (home/away splits measurably affect scoring)
- `DAYS_REST` — days since last game, capped at 7 (back-to-backs suppress performance)

All features are z-score normalized per player using training-split statistics only.

**Static prop-line features (2-dim, passed directly to model heads):**
- `PROP_LINE_Z` — prop line z-scored against the training distribution
- `LINE_VS_RECENT` — `(line − rolling_mean_PTS) / player_std` (form-relative context)

**Label:** `1 = OVER`, `0 = UNDER`.

---

## Results

| Model             | Test Accuracy | Test F1 (macro) | Test AUROC |
|-------------------|---------------|-----------------|------------|
| Baseline (LogReg) | 51.4%         | 0.503           | 0.516      |
| LSTM              | 52.9%         | 0.519           | 0.523      |
| Fusion            | 53.1%         | 0.521           | 0.535      |
| Per-Player        | 53.4%         | 0.531           | 0.540      |
| **XGBoost**       | **TBD**       | **TBD**         | **TBD**    |

Run `python scripts/evaluate.py` after Colab training to populate XGBoost. It is expected to outperform the LSTM by 3–8 AUROC points on this dataset size.

---

## Setup

```bash
git clone https://github.com/Nathan-Dela-Pena/Podz-Liability-Fund.git
cd Podz-Liability-Fund
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
ODDS_API_KEY=<your key from the-odds-api.com>
```

---

## Data Pipeline

### 1. Game logs + prop lines

```bash
python scripts/run_ingestion.py    # pulls nba_api game logs + The Odds API prop lines
python scripts/run_processing.py   # merges into labeled records + rolling-window CSVs
```

### 2. Video frames + pose

```bash
python -m src.ingestion.fetch_youtube_clips   # downloads highlights via yt-dlp
python -m src.processing.extract_frames       # samples frames from clips
python -m src.processing.extract_pose         # YOLO + MediaPipe → 267-dim pose CSVs
```

**Pose vector (267-dim):** 33 landmarks × (x, y, z) means + stds (motion variability) + 3 derived fatigue signals (vertical acceleration, stride length variance, shoulder droop).

Frames and pose CSVs are gitignored — they live on Google Drive for Colab training.

---

## Training

### Google Colab (recommended — T4 GPU)

1. Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Ensure `podz-liability-fund` is accessible in your Google Drive
4. Run all cells top to bottom:

| Section | What it does |
|---------|--------------|
| 1. Mount Drive | Connects Google Drive |
| 2. Clone repo + deps | Clones latest main, installs requirements |
| 3. Verify sequences | Checks prop-line feature columns in train.csv |
| 4. Train LSTM | BiLSTM with prop-line static features |
| 5. Train CNN | MobileNetV2 + pose player-identity encoder |
| 6. Train Fusion | End-to-end LSTM + CNN fusion |
| 7. Train XGBoost | Gradient boosting on flat rolling features (no GPU needed) |
| 8. Train Per-Player | Per-player logistic regression ensemble |
| 9. Evaluate | Unified results table across all 5 models |
| 10. Save to Drive | Copies all `.pt` and `.pkl` checkpoints to Drive |

### Locally (no GPU needed for LSTM, XGBoost, Per-Player)

```bash
python scripts/run_training.py       # LSTM → checkpoints/lstm_best.pt
python -m src.models.xgb_model       # XGBoost → checkpoints/xgb_best.pkl
python -m src.models.per_player      # Per-Player → checkpoints/per_player_best.pkl
python scripts/evaluate.py           # unified results table (all 5 models)
```

---

## Model Details

### LSTM

```
Input  (batch, 5, 8)
  → BiLSTM(hidden=64) → hidden (128)
  → static_mlp(2 → 32)   [PROP_LINE_Z, LINE_VS_RECENT]
  → concat (160) → Dropout(0.35) → Linear(160→64) → ReLU → Linear(64→1)
```

| Hyperparameter | Value |
|----------------|-------|
| Window size    | 5 games |
| Hidden size    | 64 (bidirectional) |
| Dropout        | 0.35 |
| Learning rate  | 1e-3 |
| Early stopping | patience=12 on val AUROC |
| Imbalance      | WeightedRandomSampler |

### CNN (Player-Identity Encoder)

```
Frames (B, T, 3, 224, 224) → MobileNetV2 → mean-pool → Linear → 128-d
Pose   (B, 267)             → Linear → LayerNorm → ReLU → 128-d
Concat → Linear(256 → 192)             — shared embedding
  → Linear(192 → num_players)          — player-ID head (training only)
```

Trained via a **player-identity auxiliary task** (22-class classification) rather than noisy per-clip over/under labels. The 192-d embedding feeds the fusion model.

### Fusion

```
LSTM hidden (128) + static (32) ─┐
                                  ├─ learned softmax weights → weighted sum (128)
CNN  hidden (192) → proj  (128) ─┘
  + static_mlp (32) → concat (160)
  → LayerNorm → Linear(160→64) → ReLU → Dropout → Linear(64→1)
```

### XGBoost

Gradient boosting on the flat 40-dim rolling window + 2 static features.
`max_depth=4`, `learning_rate=0.03`, `n_estimators=400`, early stopping on val AUROC.
Outperforms LSTM on small tabular data by handling non-linear feature interactions natively without sequence modeling overhead.

### Per-Player Ensemble

Separate `LogisticRegression(C=0.1)` per player (~30–60 training samples each, strong L2 regularization).
Global `LogisticRegression(C=0.05)` fallback for players with fewer than 15 training samples.
Each player-specific model specializes on that player's individual scoring patterns.

---

## API

```bash
ODDS_API_KEY=your_key python api/app.py
# → http://localhost:8080
```

**`GET /predict`** — fetches today's NBA prop lines, runs the best available model (fusion > LSTM), returns OVER/UNDER + confidence per player sorted by confidence. Falls back to CSV-based predictions if The Odds API returns no data.

**`GET /health`** — model mode, device, checkpoint sizes.

---

## Deployment

**API → Render:** root dir `api`, build `pip install -r requirements.txt`, start `gunicorn app:app`, add `ODDS_API_KEY` env var.

**Frontend → Vercel:** root dir `frontend`, update `API_BASE` in `frontend/index.html` to your Render URL.

---

## Project Structure

```
src/models/
  lstm.py           BiLSTM + static feature MLP
  cnn.py            MobileNetV2 + pose player-identity encoder
  fusion.py         late-fusion model (LSTM + CNN + prop-line)
  xgb_model.py      XGBoost gradient boosting
  per_player.py     per-player logistic regression ensemble
  baseline.py       logistic regression baseline
scripts/
  evaluate.py       unified results table (all 5 models)
  run_training.py   train LSTM locally
  run_evaluation.py detailed LSTM evaluation (all splits + per-player breakdown)
notebooks/
  train_colab.ipynb Colab notebook — all 5 models on T4 GPU
data/processed/     train/val/test CSVs (committed — 654/85/701 sequences)
checkpoints/        .pt and .pkl model checkpoints
config.py           hyperparameters and path constants
```

---

## Limitations

- **Small training set:** 654 sequences across 2 teams. Expanding to all 30 teams and additional seasons is the single biggest lever for improving AUROC beyond ~55%.
- **CNN data scarcity:** ~2–3 YouTube clips per player in validation — the identity encoder overfits. More clips per player would meaningfully help the fusion branch.
- **No live rolling stats:** The rolling-stats window uses the most recent row in `test.csv`, not a live nba_api call.
- **Prop line source:** Uses the first bookmaker returned by The Odds API rather than consensus lines (±0.5 pt variance across books).
