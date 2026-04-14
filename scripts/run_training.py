"""
run_training.py

Trains the LSTM branch on data/processed/train.parquet and saves the best
checkpoint to checkpoints/lstm_best.pt.

Usage:
  python scripts/run_training.py
"""

from src.models.lstm import train_lstm

if __name__ == "__main__":
    train_lstm()
