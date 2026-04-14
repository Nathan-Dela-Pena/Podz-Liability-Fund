# Podz-Liability-Fund

CS 489/689 Deep Learning — Multimodal NBA player prop prediction system.

Predicts whether Golden State Warriors and San Antonio Spurs players will go **OVER or UNDER** their bookmaker-set points prop line by fusing two deep learning branches:

- **Branch 1 (LSTM):** Rolling window of per-game advanced stats encoded by a bidirectional LSTM
- **Branch 2 (CNN):** MobileNetV2 + MediaPipe Pose keypoints extracted from play-by-play clips
- **Fusion:** Both hidden states concatenated and passed through dense layers with Adam end-to-end

Success criterion: Fusion model ≥ 58% accuracy and ≥ 3pp over the logistic regression baseline.

---

## Project Structure

```
src/
  ingestion/
    fetch_game_logs.py     # nba_api game logs + advanced stats per player-season
    fetch_odds.py          # The Odds API — prop lines via multi-snapshot strategy
    fetch_clips.py         # stats.nba.com play-by-play video clips (FGA events)
  processing/
    build_game_records.py  # merge game logs + odds → labeled records
    build_sequences.py     # rolling-window sequences + chronological CSV splits
    extract_frames.py      # sample frames from .mp4 clips via OpenCV
    extract_pose.py        # MediaPipe Pose keypoints + fatigue signals per clip
  models/
    baseline.py            # logistic regression baseline
    lstm.py                # bidirectional single-layer LSTM branch
    cnn.py                 # MobileNetV2 + pose head CNN branch
    fusion.py              # fusion model (stub — Week 6)
scripts/
  run_ingestion.py         # fetch game logs + odds
  run_processing.py        # build records + sequences
  run_baseline.py          # train + evaluate logistic regression baseline
  run_training.py          # train LSTM branch
  run_evaluation.py        # evaluate LSTM checkpoint on all splits
data/
  raw/
    game_logs/             # per-player-season CSVs from nba_api
    odds/                  # all_odds.json from The Odds API
    clips/                 # .mp4 clips per player per event
  processed/               # train.csv / val.csv / test.csv + all_records.csv
  frames/                  # sampled JPG frames per clip
  pose/                    # per-player pose feature CSVs
checkpoints/               # saved model weights + results JSON
```

---

## Data Pipeline

### Statistical Branch

```bash
# 1. Fetch game logs (nba_api) + prop lines (The Odds API)
python scripts/run_ingestion.py

# 2. Merge into labeled records and build rolling-window sequences
python scripts/run_processing.py
```

**Splits (chronological, no shuffling):**
| Split | Date Range | Sequences |
|-------|-----------|-----------|
| Train | Oct 2023 – Feb 2025 | 461 |
| Val   | Mar 2025 – Apr 2025 | 57  |
| Test  | Oct 2025 – Apr 2026 | 701 |

**Features:** PTS, TS%, eFG%, USG%, OPP_DRTG, MIN (6 features × 5-game window = 30-dim input)

### Vision Branch

```bash
# 3. Download play-by-play FGA clips per player
python -m src.ingestion.fetch_clips

# 4. Extract sampled frames from clips
python -m src.processing.extract_frames

# 5. Run MediaPipe Pose — extract keypoints + fatigue signals
python -m src.processing.extract_pose
```

Pose features per clip: 132 landmark stats (mean of 33 landmarks × (x,y,z,visibility)) + 3 derived signals (vertical acceleration, stride length variance, shoulder droop) = **135-dim vector**.

---

## Models

### Baseline — Logistic Regression

```bash
python scripts/run_baseline.py
```

Flattened 30-dim rolling window → `LogisticRegression(class_weight='balanced')`. Serves as the bar the LSTM must beat by ≥ 3pp.

| Split | Accuracy | F1 (macro) | AUROC |
|-------|----------|-----------|-------|
| Train | 57.3%    | 0.572     | 0.616 |
| Val   | 45.6%    | 0.439     | 0.355 |
| Test  | 49.9%    | 0.496     | 0.510 |

---

### Branch 1 — Bidirectional LSTM

```bash
python scripts/run_training.py      # trains → checkpoints/lstm_best.pt
python scripts/run_evaluation.py    # evaluates all splits + per-player breakdown
```

**Architecture:**
```
Input (batch, 5, 6)
  → BiLSTM(input=6, hidden=64, bidirectional=True)
  → hidden (batch, 128)  [fwd=64 + bwd=64]
  → Dropout(0.3)
  → Linear(128→64) + ReLU + Dropout(0.3)
  → Linear(64→1)
Output (batch, 1) — logit (probability of OVER)
```

**Hyperparameters:** `WINDOW_SIZE=5`, `HIDDEN_SIZE=64`, `NUM_LAYERS=1`, `DROPOUT=0.3`, `LR=1e-3`, `EPOCHS=50`, `BATCH_SIZE=32`

Per-player z-score normalization fitted on train split only. `WeightedRandomSampler` addresses class imbalance. Early stopping on val accuracy saves best checkpoint.

---

### Branch 2 — MobileNetV2 + Pose (CNN)

```bash
python -m src.models.cnn   # trains → checkpoints/cnn_best.pt
```

**Architecture:**
```
Frame (3, 224, 224) → MobileNetV2 backbone → Linear(1280→256) → ReLU → Linear(256→128)
Pose vector (135,)  → Linear(135→64) → ReLU
Concat (192,) → Dropout(0.3) → Linear(192→1)
Output (batch, 1) — logit (probability of OVER)
```

The 192-dim hidden vector is the CNN branch output passed to the fusion model.

---

### Fusion Model *(Week 6 — upcoming)*

Concatenates LSTM hidden (128) + CNN hidden (192) → dense layers → sigmoid output. Adam optimizer learns effective branch weighting end-to-end.

---

## Players Tracked

**Golden State Warriors:** Curry, Green, Kuminga, Moody, Payton II, Looney, Podziemski, Santos, Post, Richard, Spencer, Butler, S. Curry, Horford, Melton, Porzingis, Wiggins (pre-trade)

**San Antonio Spurs:** Wembanyama, Fox (pre-trade), K. Johnson, Castle, Harper

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `mediapipe`, `opencv-python`, `nba_api`, `scikit-learn`, `pandas`, `numpy`

Requires `.env` with `ODDS_API_KEY=<your_key>`.
