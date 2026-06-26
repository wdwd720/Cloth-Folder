#!/usr/bin/env python3
"""Train the keypoint/grasp-place heatmap detector (Model 1).

Wraps `terafold train-keypoints`. Point --data at a synthetic dataset
produced by `scripts/02_generate_synthetic.py`. Requires the ML extra
(torch): `pip install -e '.[ml]'`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/synthetic", help="Synthetic dataset directory.")
    p.add_argument("--out", default="runs/keypoints", help="Run output directory (model.pt).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples (debug).")
    args = p.parse_args()

    cmd = [
        "terafold", "train-keypoints",
        "--data", args.data,
        "--out", args.out,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
