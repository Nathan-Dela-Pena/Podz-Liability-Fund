"""
run_baseline.py
Runs the logistic regression baseline model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.baseline import run

if __name__ == "__main__":
    run()
