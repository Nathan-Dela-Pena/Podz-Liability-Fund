# Podz-Liability-Fund

CS 489/689 Deep Learning — Multimodal NBA player prop prediction system.

Predicts whether Golden State Warriors and San Antonio Spurs players will go **OVER or UNDER** their bookmaker-set points prop line by fusing two deep learning branches:

- **Branch 1 (LSTM):** Rolling window of per-game advanced stats encoded by a bidirectional LSTM
- **Branch 2 (CNN):** MobileNetV2 + MediaPipe Pose keypoints extracted from play-by-play clips
- **Fusion:** Both hidden states concatenated and passed through dense layers with Adam end-to-end

**Success criterion:** Fusion model ≥ 58% accuracy and ≥ 3pp over the logistic regression baseline.

---

## Project Structure

```
api/
  app.py                   # Flask REST API — /predict + /health endpoints
  requirements.txt         # API-specific dependencies
src/
  ingestion/
    fetch_game_logs.py     # nba_api game logs + advanced stats per player-season
    fetch_odds.py          # The Odds API — prop lines via multi-snapshot strategy
    fetch_clips.py         # stats.nba.com play-by-play video clips (FGA events)
    fetch_youtube_clips.py # YouTube highlights fallback via yt-dlp
  processing/
    build_game_records.py  # merge game logs + odds → labeled records
    build_sequences.py     # rolling-window sequences + chronological CSV splits
    extract_frames.py      # sample frames from .mp4 clips via OpenCV
    extract_pose.py        # YOLO person detection + MediaPipe Pose per clip
  models/
    baseline.py            # logistic regression baseline
    lstm.py                # bidirectional single-layer LSTM branch
    cnn.py                 # MobileNetV2 + pose head CNN branch
    fusion.py              # late-fusion model (LSTM + CNN)
scripts/
  run_ingestion.py         # fetch game logs + odds
  run_processing.py        # build records + sequences
  run_baseline.py          # train + evaluate logistic regression baseline
  run_training.py          # train LSTM branch
  run_evaluation.py        # evaluate LSTM checkpoint on all splits
  run_cnn_pipeline.py      # full CNN + fusion pipeline (fetch → pose → train)
notebooks/
  train_colab.ipynb        # Colab notebook — CNN + fusion training on T4 GPU
data/
  raw/
    game_logs/             # per-player-season CSVs from nba_api
    odds/                  # all_odds.json from The Odds API
    clips/                 # .mp4 clips per player per event
  processed/               # train.csv / val.csv / test.csv (committed)
  frames/                  # sampled JPG frames per clip (gitignored)
  pose/                    # per-player pose feature CSVs (gitignored)
checkpoints/               # saved model weights + results JSON
config.py                  # all hyperparameters and path constants
```

---

## Data Collection

### Statistical Branch

Game logs are pulled from **nba_api** (official NBA stats API, no key required). Prop lines are fetched from **The Odds API** and matched to each game by date and player name.

```bash
python scripts/run_ingestion.py   # fetch game logs + prop lines
python scripts/run_processing.py  # build labeled records + rolling-window sequences
```

**Splits (chronological, no shuffling — critical to prevent leakage):**

| Split | Date Range            | Sequences |
|-------|-----------------------|-----------|
| Train | Oct 2023 – Feb 2025   | 461       |
| Val   | Mar 2025 – Apr 2025   | 57        |
| Test  | Oct 2025 – Apr 2026   | 701       |

**Features (6 × 5-game window = 30-dim input):**
`PTS`, `TS%`, `eFG%`, `USG%`, `OPP_DRTG`, `MIN`

All features are z-score normalized per player using statistics computed on the training split only (no test leakage).

**Label:** `1 = OVER` (player scored more than their bookmaker prop line), `0 = UNDER`.

### Vision Branch

Play-by-play FGA clips are downloaded via the **stats.nba.com** video API using `PlayByPlayV3` endpoint. Where NBA API clips are unavailable, **YouTube highlights** are used as a fallback via `yt-dlp`.

Player identification in multi-person frames uses **YOLOv8n** person detection followed by HSV jersey color scoring (Warriors blue/gold, Spurs black/white) to select the correct athlete's crop before running **MediaPipe Pose**.

```bash
python scripts/run_cnn_pipeline.py   # full: fetch → frames → pose → train CNN → train fusion
```

Or step by step:

```bash
python -m src.ingestion.fetch_clips        # download FGA clips
python -m src.processing.extract_frames   # sample frames
python -m src.processing.extract_pose     # YOLO + MediaPipe → 135-dim pose vectors
python -m src.models.cnn                  # train CNN branch
```

**Pose features per clip:** 33 landmarks × (x, y, z, visibility) = 132 stats + 3 derived fatigue signals (vertical acceleration, stride length variance, shoulder droop) = **135-dim vector**.

Because the vision pipeline is compute-intensive, CNN and fusion training are designed to run on **Google Colab with a T4 GPU** (see Colab section below).

---

## Model Architecture

### Baseline — Logistic Regression

Flattened 30-dim rolling window → `LogisticRegression(class_weight='balanced')`.

| Split | Accuracy | F1 (macro) | AUROC |
|-------|----------|------------|-------|
| Train | 57.3%    | 0.572      | 0.616 |
| Val   | 45.6%    | 0.439      | 0.355 |
| Test  | 49.9%    | 0.496      | 0.510 |

```bash
python scripts/run_baseline.py   # saves checkpoints/baseline_results.json
```

---

### Branch 1 — Bidirectional LSTM

```
Input  (batch, 5, 6)
  → BiLSTM(input=6, hidden=64, bidirectional=True)
  → hidden (batch, 128)   [fwd=64 + bwd=64]
  → Dropout(0.3)
  → Linear(128→64) + ReLU + Dropout(0.3)
  → Linear(64→1)
Output (batch, 1)  — logit → probability of OVER
```

**Hyperparameters:**

| Parameter     | Value  |
|---------------|--------|
| WINDOW_SIZE   | 5      |
| HIDDEN_SIZE   | 64     |
| NUM_LAYERS    | 1      |
| DROPOUT       | 0.3    |
| LR            | 1e-3   |
| EPOCHS        | 50     |
| BATCH_SIZE    | 32     |

`WeightedRandomSampler` addresses class imbalance. Early stopping on validation accuracy saves the best checkpoint.

```bash
python scripts/run_training.py     # saves checkpoints/lstm_best.pt
python scripts/run_evaluation.py   # saves checkpoints/lstm_results.json
```

---

### Branch 2 — MobileNetV2 + Pose (CNN)

```
Frame (3, 224, 224) → MobileNetV2 backbone → Linear(1280→256) → ReLU → Linear(256→128)
Pose  (135,)        → Linear(135→64)        → ReLU
Concat (192,) → Dropout(0.3) → Linear(192→1)
Output (batch, 1)  — logit → probability of OVER
```

Training improvements: val-AUC checkpoint, patience=10 early stopping, `ReduceLROnPlateau(factor=0.5, patience=5)`, gradient clipping (`max_norm=1.0`).

Checkpoint saved to `checkpoints/cnn_best.pt`.

---

### Fusion Model

```
LSTM hidden  (128) → Linear(128→128) ─┐
                                       ├─ weighted sum (128) → LayerNorm
CNN  hidden  (192) → Linear(192→128) ─┘
  → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
Output (batch, 1)  — logit → probability of OVER
```

Branch contributions controlled by a learnable softmax weight pair (initialised ~61% LSTM / ~39% CNN). Adam uses two learning rates: fusion head `lr=1e-3`, branch encoders `lr=1e-4`.

Where CNN pose/frame data is absent for a sample, CNN inputs are zeroed out so the model can train on stats-only samples.

Checkpoint saved to `checkpoints/fusion_best.pt`.

---

## Training on Google Colab

The CNN and fusion models require a GPU. Use the provided Colab notebook:

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → GPU)
3. Add a shortcut to the `podz-liability-fund` folder in your Google Drive (Shared with me → right-click → Organize → Add shortcut → My Drive)
4. Run all cells top to bottom (~40 min CNN + ~15 min fusion)
5. Before running Step 6 (fusion), **upload `checkpoints/lstm_best.pt`** from your local machine to `My Drive/podz-liability-fund/checkpoints/`

Checkpoints are saved directly to Drive — no manual download needed.

---

## REST API

The Flask API serves real-time predictions using the latest available checkpoint (`fusion_best.pt` → `lstm_best.pt` fallback).

### Running locally

```bash
pip install -r api/requirements.txt
python api/app.py                  # dev server on :5000
# or
gunicorn api.app:app               # production
```

### Endpoints

**`GET /predict`**

Returns OVER/UNDER recommendation + confidence for all tracked players based on their last 5 regular-season games fetched live from nba_api.

```json
{
  "model_mode": "lstm",
  "players": [
    {
      "name": "Stephen Curry",
      "recommendation": "OVER",
      "confidence": 0.7231,
      "prop_line": 27.5,
      "stat_type": "points"
    }
  ]
}
```

**`GET /health`**

```json
{
  "status": "ok",
  "model_mode": "lstm",
  "device": "cpu",
  "checkpoints": {
    "lstm_best.pt": "1.2 MB"
  }
}
```

Rate limited to **10 requests / minute per IP**.

### Environment variables

| Variable          | Default        | Description                       |
|-------------------|----------------|-----------------------------------|
| `CHECKPOINTS_DIR` | `checkpoints/` | Override checkpoint directory     |
| `MODEL_DEVICE`    | auto-detect    | `cuda` or `cpu`                   |

---

## Understanding Predictions

- **OVER** — model predicts the player will score more than their bookmaker prop line
- **UNDER** — model predicts the player will score less
- **Confidence** — distance of the predicted probability from 0.5 (i.e. 0.72 means the model assigns 72% probability to the predicted side)

The model is trained on 2023–2026 Warriors and Spurs game data. Predictions are only meaningful for tracked players during active NBA regular seasons. This is a research project, not a betting tool.

---

## Players Tracked

**Golden State Warriors:** Curry, Green, Kuminga, Moody, Payton II, Looney, Podziemski, Santos, Post, Richard, Spencer, Butler, S. Curry, Horford, Melton, Porzingis, Wiggins

**San Antonio Spurs:** Wembanyama, Fox, K. Johnson, Castle, Harper

---

## Setup

```bash
git clone https://github.com/Nathan-Dela-Pena/Podz-Liability-Fund.git
cd Podz-Liability-Fund
pip install -r requirements.txt
```

Create a `.env` file:

```
ODDS_API_KEY=<your_key_from_the-odds-api.com>
```

**Key dependencies:** `torch`, `torchvision`, `mediapipe`, `opencv-python`, `ultralytics`, `nba_api`, `scikit-learn`, `pandas`, `yt-dlp`

---

## Limitations & Future Work

- Vision branch relies on weak supervision (YouTube clips with player-level labels, not clip-level ground truth) — true per-clip shot outcomes would improve CNN signal quality
- Jersey color heuristics are hardcoded per team; a learned re-identification model would generalise better
- Prop lines are matched by date and player name — missing lines fall back to a rolling average
- Model covers only two NBA teams; expanding to the full league would require more data and likely a player-embedding layer
- No live odds integration — prop lines in `/predict` come from cached historical snapshots; real deployment would need a live odds feed
