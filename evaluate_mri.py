#!/usr/bin/env python
"""CLI: evaluate the phase-2 MRI classifier on train/val/test splits -> print metrics, save plots."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alz.imaging import DEFAULT_MODEL_PATH, evaluate_mri

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "data", "imagesoasis", "versions", "1", "Data"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot-dir", default=os.path.join(os.path.dirname(__file__), "eval_plots"))
    args = parser.parse_args()

    results = evaluate_mri(
        args.data, model_path=args.model, device=args.device, limit=args.limit, plot_dir=args.plot_dir
    )

    for split, m in results.items():
        print(f"{split}:")
        for k, v in m.items():
            print(f"  {k}: {v}")

    print(f"Plots saved to {args.plot_dir}")


if __name__ == "__main__":
    main()
