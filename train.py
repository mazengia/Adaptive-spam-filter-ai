"""
Command-line training.   python train.py --data data/emails.csv --out models/v1
Add corrections from the web UI:  --feedback data/feedback.csv
(The web UI's Retrain button does the same thing without the command line.)
"""
import argparse

from spamfilter.training import FEEDBACK_WEIGHT, train_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feedback")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-precision", type=float, default=0.995)
    ap.add_argument("--feedback-weight", type=float, default=FEEDBACK_WEIGHT)
    a = ap.parse_args()
    try:
        train_model(a.data, a.feedback, a.out, a.target_precision, a.feedback_weight, verbose=True)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")