#!/usr/bin/env python
"""CLI: load OASIS data -> train -> save models/model.joblib."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alz.data import clean, load_raw
from alz.metrics import save_metrics
from alz.model import save_model, train_model

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "oasis_longitudinal.csv")
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "model.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_MODEL)
    args = parser.parse_args()

    df = load_raw(args.data)
    X, y = clean(df)
    pipeline, metrics = train_model(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(pipeline, args.out)
    save_metrics("clinical", metrics)

    print(f"Trained on {len(X)} rows (5-fold CV, out-of-fold metrics):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
