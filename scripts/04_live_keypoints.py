#!/usr/bin/env python3
"""Live keypoint overlay from a webcam.

Wraps `terafold live-keypoints`. Requires OpenCV (`pip install -e '.[vision]'`).
Pass --checkpoint to use a trained detector; otherwise the classical
fallback predictor is used. Press 'q' (or Ctrl+C) to stop.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--checkpoint", default=None, help="Keypoint model checkpoint (optional).")
    args = p.parse_args()

    cmd = ["terafold", "live-keypoints", "--camera-index", str(args.camera_index)]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
