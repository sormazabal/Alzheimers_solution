#!/usr/bin/env python
"""CLI: train the phase-3 EEG classifier -> save models/eeg_model.joblib."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alz.eeg import DEFAULT_MODEL_PATH, train_eeg
from alz.metrics import save_metrics

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "ds004504", "derivatives")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    metrics = train_eeg(args.data, out_path=args.out)
    save_metrics("eeg", metrics)

    print(f"Trained on data from {args.data} (5-fold CV, out-of-fold metrics):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
