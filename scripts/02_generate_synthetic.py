#!/usr/bin/env python3
"""Generate a synthetic top-down towel dataset with keypoint/mask labels.

Wraps `terafold generate-synthetic`. Use --pairs to also emit before/after
fold pairs for training the success model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=10000, help="Number of samples.")
    p.add_argument("--out", default="data/synthetic", help="Output dataset directory.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markers", action="store_true", help="Marker-assisted mode.")
    p.add_argument("--pairs", action="store_true", help="Also generate before/after fold pairs.")
    args = p.parse_args()

    cmd = [
        "terafold", "generate-synthetic",
        "--num", str(args.num),
        "--out", args.out,
        "--image-size", str(args.image_size),
        "--seed", str(args.seed),
        "--markers" if args.markers else "--no-markers",
    ]
    if args.pairs:
        cmd.append("--pairs")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
