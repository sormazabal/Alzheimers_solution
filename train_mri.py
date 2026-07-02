#!/usr/bin/env python
"""CLI: train the phase-2 MRI severity classifier -> save models/mri_model.pt."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alz.imaging import DEFAULT_MODEL_PATH, train_mri

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "data", "imagesoasis", "versions", "1", "Data"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    accuracy = train_mri(
        args.data, out_path=args.out, epochs=args.epochs, limit=args.limit, device=args.device
    )

    print(f"Trained on data from {args.data}, held-out accuracy: {accuracy:.2f}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
