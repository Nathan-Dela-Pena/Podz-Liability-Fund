"""
run_fusion.py

Trains the multimodal fusion model (LSTM + CNN) and saves the best
checkpoint to checkpoints/fusion_best.pt.

Requires:
  - checkpoints/lstm_best.pt     (run scripts/run_training.py first)
  - checkpoints/cnn_best.pt      (run python -m src.models.cnn first)
  - data/processed/train.csv
  - data/processed/val.csv
  - data/pose/                   (run python -m src.processing.extract_pose)
  - data/frames/                 (optional — zero-image used if missing)

Usage:
  python scripts/run_fusion.py
"""

from src.models.fusion import train_fusion

if __name__ == "__main__":
    train_fusion()
