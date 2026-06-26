#!/usr/bin/env python3
"""Smoke-test a camera by capturing a single frame.

Wraps `terafold camera-test`. Use --camera mock to verify the pipeline
without any hardware, or point --camera-index at a real webcam.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--camera", default="opencv", help="opencv | mock")
    p.add_argument("--out", default=None, help="Optional path to save the captured frame.")
    args = p.parse_args()

    cmd = [
        "terafold", "camera-test",
        "--camera-index", str(args.camera_index),
        "--camera", args.camera,
    ]
    if args.out:
        cmd += ["--out", args.out]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
