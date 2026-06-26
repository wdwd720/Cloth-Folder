#!/usr/bin/env python3
"""Export recorded episodes to a LeRobot-compatible dataset.

Wraps `terafold export-lerobot`. Reads episodes recorded by
`scripts/07_record_demo.py` and writes a LeRobot-style dataset (meta/,
data/, images/) plus suggested ACT/SmolVLA training commands.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episodes",
        default="data/episodes/fold_towel_half",
        help="Episodes root directory.",
    )
    p.add_argument("--out", default="data/lerobot/fold_towel_half", help="Dataset output directory.")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    cmd = [
        "terafold", "export-lerobot",
        "--episodes", args.episodes,
        "--out", args.out,
        "--fps", str(args.fps),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
