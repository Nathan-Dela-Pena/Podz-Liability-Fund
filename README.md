# Podz Liability Fund

CS 489/689 Deep Learning — Wilfrid Laurier University

A multimodal NBA player prop prediction system that predicts **OVER / UNDER** on nightly points props for Golden State Warriors and San Antonio Spurs players. The system fuses a bidirectional LSTM (rolling game-log stats) with a MobileNetV2 CNN (pose keypoints extracted from YouTube highlights) through a learned late-fusion model.

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
    train.csv / val.csv / test.csv           <player>.csv (135-dim)
             │                                     │
    ┌────────▼────────┐               ┌────────────▼────────────┐
    │  Branch 1: LSTM │               │   Branch 2: CNN         │
    │  BiLSTM(64)     │               │   MobileNetV2 + Pose    │
    │  → hidden (128) │               │   → hidden (192)        │
    └────────┬────────┘               └────────────┬────────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │    Fusion Model         │
              │  Learned softmax weight │
              │  ~61% LSTM / ~39% CNN   │
              │  → LayerNorm → Linear   │
              │  → logit (OVER/UNDER)   │
              └────────────┬────────────┘
                           │
                    api/app.py  ←── The Odds API (live prop lines)
                           │
                   frontend/index.html
                   (8-ball UI, picks cycle inside white circle)
```

---

## Players Tracked

**Golden State Warriors (22 players):**
Stephen Curry, Draymond Green, Jonathan Kuminga, Moses Moody, Gary Payton II, Kevon Looney, Brandin Podziemski, Gui Santos, Quinten Post, Will Richard, Pat Spencer, Jimmy Butler, Seth Curry, Al Horford, De'Anthony Melton, Kristaps Porzingis, Andrew Wiggins

**San Antonio Spurs (5 players):**
Victor Wembanyama, De'Aaron Fox, Keldon Johnson, Stephon Castle, Dylan Harper

---

## Dataset

| Split | Date Range          | Sequences |
|-------|---------------------|-----------|
| Train | Oct 2023 – Feb 2025 | 654       |
| Val   | Mar 2025 – Apr 2025 | 85        |
| Test  | Oct 2025 – Apr 2026 | 701       |

**Splits are chronological (no shuffling)** — test set is never touched during training to prevent look-ahead leakage.

**Input features (6 × 5-game rolling window = 30-dim):**
`PTS`, `TS%`, `eFG%`, `USG%`, `OPP_DRTG`, `MIN`

All features are z-score normalized per column using training-split statistics only.

**Label:** `1 = OVER` (player scored more than their prop line), `0 = UNDER`.

---

## Results

| Model          | Test Accuracy | Test F1 (macro) | Test AUROC |
|----------------|---------------|-----------------|------------|
| Baseline (LogReg) | 51.4%      | 0.503           | 0.516      |
| **LSTM**       | **51.6%**     | **0.387**       | **0.530**  |
| CNN + Fusion   | *see fusion_best.pt* | —           | —          |

> The LSTM modestly outperforms the logistic regression baseline on AUROC. F1 is low because the model currently predicts UNDER for nearly all samples — class imbalance in the test set. The fusion model runs end-to-end with both `cnn_best.pt` and `fusion_best.pt` now trained and committed.

---

## Setup

```bash
git clone https://github.com/Nathan-Dela-Pena/Podz-Liability-Fund.git
cd Podz-Liability-Fund
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

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

Outputs: `data/processed/train.csv`, `val.csv`, `test.csv`

### 2. Video frames + pose (vision branch)

Video clips were sourced from **YouTube highlights** via `yt-dlp` (NBA API video endpoint was unreliable). Player identification in multi-person frames uses a two-stage pipeline:

1. **YOLOv8n** detects all persons in the frame
2. **HSV jersey color scoring** picks the target player's bounding box (Warriors blue/gold, Spurs black/white)
3. **MediaPipe Pose** extracts 33 landmarks from the selected crop

```bash
python -m src.ingestion.fetch_youtube_clips   # downloads highlights (yt-dlp)
python -m src.processing.extract_frames       # samples frames from clips
python -m src.processing.extract_pose         # YOLO + MediaPipe → 135-dim pose CSVs
```

**Pose vector (135-dim per clip):**
33 landmarks × (x, y, z, visibility) = 132 stats + 3 derived fatigue signals (vertical acceleration, stride length variance, shoulder droop).

Frames and pose CSVs are **gitignored** (large binary/float data). They live on Google Drive for Colab training.

---

## Training

### Branch 1 — Bidirectional LSTM

**Architecture:**
```
Input  (batch, 5, 6)
  → BiLSTM(hidden=64, bidirectional=True)
  → hidden (batch, 128)
  → Dropout(0.35)
  → Linear(128 → 64) + ReLU + Dropout(0.35)
  → Linear(64 → 1)
Output (batch, 1) — logit
```

**Hyperparameters:**

| Parameter      | Value  |
|----------------|--------|
| Window size    | 5 games |
| Hidden size    | 64     |
| Layers         | 1      |
| Dropout        | 0.35   |
| Learning rate  | 1e-3   |
| Weight decay   | 5e-5   |
| Batch size     | 32     |
| Max epochs     | 60     |
| Early stopping | patience = 7 (val accuracy) |
| LR schedule    | ReduceLROnPlateau(factor=0.5, patience=5) |
| Optimizer      | Adam   |
| Loss           | BCEWithLogitsLoss |
| Imbalance      | WeightedRandomSampler |

**Run locally:**
```bash
source .venv/bin/activate
python scripts/run_training.py     # trains → checkpoints/lstm_best.pt
python scripts/run_evaluation.py   # evaluates all splits → checkpoints/lstm_results.json
```

Checkpoint is already trained and committed at `checkpoints/lstm_best.pt` (0.2 MB).

---

### Branch 2 — MobileNetV2 + Pose (CNN)

**Architecture:**
```
Frame (3, 224, 224) → MobileNetV2 backbone → Linear(1280 → 256) → ReLU → Linear(256 → 128)
Pose  (135,)        → Linear(135 → 64) → ReLU
Concat (192,) → Dropout(0.35) → Linear(192 → 1)
Output (batch, 1) — logit
```

**Hyperparameters:** Same as LSTM (Adam, lr=1e-3, BCEWithLogitsLoss, batch=32, max epochs=60, early stopping patience=10, ReduceLROnPlateau).

**Training on Google Colab (T4 GPU — required, ~40 min):**

1. Upload `checkpoints/lstm_best.pt` to `My Drive/podz-liability-fund/checkpoints/` on Google Drive
2. Open `notebooks/train_colab.ipynb` in Colab
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Add a Drive shortcut: in Drive, find `podz-liability-fund` under "Shared with me" → right-click → Organize → Add shortcut → My Drive
5. Run all cells top to bottom:
   - **Cell 1** — mounts Drive
   - **Cell 2** — clones `feat/fusion` branch, installs dependencies
   - **Cell 3** — overrides config paths to point at Drive
   - **Cell 4** — verifies GPU
   - **Cell 5** — trains CNN branch → saves `cnn_best.pt` to Drive (~40 min)
   - **Cell 6** — verifies `lstm_best.pt` is present, then trains fusion model → saves `fusion_best.pt` (~15 min)
   - **Cell 7** — lists saved checkpoints

Checkpoints land in `My Drive/podz-liability-fund/checkpoints/` — no manual download needed.

---

### Fusion Model

**Architecture:**
```
LSTM hidden (128) → Linear(128 → 128) ─┐
                                        ├─ softmax-weighted sum (128) → LayerNorm
CNN  hidden (192) → Linear(192 → 128) ─┘
  → Linear(128 → 64) → ReLU → Dropout(0.35) → Linear(64 → 1)
```

Learnable branch weights (softmax pair) initialised ~61% LSTM / ~39% CNN. Printed after each epoch.

**Optimizer:** Adam with two learning rates — fusion head `lr=1e-3`, branch encoders `lr=1e-4` (gentle fine-tune). Where CNN data is absent for a sample, CNN inputs are zeroed out.

---

## Running the API Locally

```bash
source .venv/bin/activate
pip install -r api/requirements.txt
ODDS_API_KEY=your_key python api/app.py
```

Then open **http://localhost:8080**

### Endpoints

**`GET /predict`**

1. Fetches today's NBA `player_points` props from The Odds API (per-event endpoint)
2. Fuzzy-matches each player name against the 22 trained players
3. Pulls their most recent rolling-stats window from `data/processed/test.csv`
4. Runs the LSTM (or fusion, if `fusion_best.pt` exists) → OVER/UNDER + confidence
5. Returns all matched players sorted by confidence descending

If `ODDS_API_KEY` is missing or The Odds API returns no data, the endpoint **falls back** to CSV-based predictions using all trained players and their stored prop lines.

Response includes `"source": "live" | "fallback"`.

**`GET /health`** — returns model mode, device, checkpoint file sizes.

### Environment variables

| Variable          | Required | Description                              |
|-------------------|----------|------------------------------------------|
| `ODDS_API_KEY`    | Recommended | Key from [the-odds-api.com](https://the-odds-api.com) — free tier has 500 requests/month |
| `CHECKPOINTS_DIR` | No       | Override checkpoint directory (default: `checkpoints/`) |
| `MODEL_DEVICE`    | No       | `cuda` or `cpu` (default: auto-detect)   |

---

## Frontend

The UI is a cinematic space-themed page with a 3D 8-ball as the hero element. Tap the ball (or shake on mobile) to fetch predictions. Picks cycle inside the white circle of the 8-ball — player last name, OVER/UNDER in green/red, and the prop line — rotating every 3 seconds.

**Configuring the API URL** (in `frontend/index.html`):

```js
const API_BASE = window.location.hostname === 'localhost'
  ? ''
  : 'https://your-render-app.onrender.com';  // ← update before deploying to Vercel
```

---

## Deployment

### API → Render

1. Connect the GitHub repo to a new Render **Web Service**
2. Set **Root directory:** `api`
3. Set **Build command:** `pip install -r requirements.txt`
4. Set **Start command:** `gunicorn app:app`
5. Add environment variable: `ODDS_API_KEY=<your key>`
6. Note the service URL (e.g. `https://podz-api.onrender.com`)

### Frontend → Vercel

1. Connect the GitHub repo to a new Vercel project
2. Set **Root directory:** `frontend`
3. No build step needed (pure HTML)
4. Update `API_BASE` in `frontend/index.html` to your Render URL before deploying

---

## Project Structure

```
api/
  app.py               Flask API — /predict (live + fallback), /health, GET /
  requirements.txt     API dependencies (flask, torch, requests, etc.)
frontend/
  index.html           8-ball UI — single file, no build step
src/
  ingestion/
    fetch_game_logs.py       nba_api game logs + advanced stats
    fetch_odds.py            The Odds API prop line snapshots
    fetch_clips.py           stats.nba.com FGA video clips
    fetch_youtube_clips.py   YouTube highlights via yt-dlp (fallback)
  processing/
    build_game_records.py    merge logs + odds → labeled records
    build_sequences.py       rolling-window sequences + chronological splits
    extract_frames.py        sample frames from .mp4 clips via OpenCV
    extract_pose.py          YOLO person detection + MediaPipe Pose
  models/
    baseline.py              logistic regression baseline
    lstm.py                  bidirectional LSTM branch
    cnn.py                   MobileNetV2 + pose head CNN branch
    fusion.py                late-fusion model (LSTM + CNN)
scripts/
  run_ingestion.py     fetch game logs + odds
  run_processing.py    build records + sequences
  run_baseline.py      train + evaluate logistic regression
  run_training.py      train LSTM
  run_evaluation.py    evaluate LSTM checkpoint on all splits
  run_cnn_pipeline.py  full CNN pipeline (fetch → frames → pose → train → fusion)
notebooks/
  train_colab.ipynb    Colab notebook — CNN + fusion on T4 GPU
data/
  raw/                 game logs, odds snapshots, clips (gitignored)
  processed/           train/val/test CSVs (committed — 654/85/701 sequences)
checkpoints/
  lstm_best.pt         ✅ trained (0.2 MB)
  cnn_best.pt          ✅ trained
  fusion_best.pt       ✅ trained
config.py              all hyperparameters and path constants
```

---

## Limitations

- **Small training set:** 1,440 total sequences across 22 players from two teams only. Real-world performance requires more teams and seasons.
- **Fusion model available:** The API uses the full fusion model when `fusion_best.pt` is present (it is). Both `cnn_best.pt` and `fusion_best.pt` are trained and committed.
- **Weak video supervision:** YouTube highlights provide clip-level labels derived from each player's historical OVER/UNDER distribution rather than game-specific outcomes. This introduces label noise in the CNN training data.
- **LSTM F1 is poor on UNDER class:** The model predicts UNDER very rarely (see confusion matrix in `lstm_results.json`). `WeightedRandomSampler` helps during training but test-set class distribution still skews predictions.
- **Prop line source:** The API uses the first bookmaker returned by The Odds API rather than consensus lines. Lines can vary ±0.5 pts across books.
- **No live stats:** The rolling-stats window comes from the most recent row in `test.csv`, not a live nba_api call. The window is from the last game in the CSV, which may be several days old.

---

## Requirements

```bash
pip install -r requirements.txt          # full pipeline
pip install -r api/requirements.txt      # API only
```

Key dependencies: `torch`, `torchvision`, `mediapipe`, `opencv-python`, `ultralytics` (YOLOv8), `nba_api`, `yt-dlp`, `scikit-learn`, `flask`, `flask-cors`, `requests`
