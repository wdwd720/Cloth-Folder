#!/usr/bin/env python3
"""Capture a table image and compute the image->table homography.

Two-step convenience flow:
  1. `terafold capture-calibration` grabs a still of the table.
  2. `terafold calibrate-homography` computes the homography from corner
     correspondences (interactive clicker if OpenCV is available, otherwise
     supply --points as a JSON file of {image_pts, table_pts}).

Pass --skip-capture to reuse an existing image.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--image", default="data/calib/table.jpg", help="Calibration image path.")
    p.add_argument("--out", default="data/calib/homography.json")
    p.add_argument("--table-width", type=float, default=0.70, help="Known rectangle width (m).")
    p.add_argument("--table-height", type=float, default=0.50, help="Known rectangle height (m).")
    p.add_argument("--points", default=None, help="JSON of {image_pts, table_pts} correspondences.")
    p.add_argument("--skip-capture", action="store_true", help="Reuse an existing --image.")
    args = p.parse_args()

    if not args.skip_capture:
        cap = ["terafold", "capture-calibration",
               "--camera-index", str(args.camera_index), "--out", args.image]
        print("Running:", " ".join(cap))
        rc = subprocess.call(cap)
        if rc != 0:
            print("Capture failed; pass --skip-capture to reuse an existing image.", file=sys.stderr)
            return rc

    cal = ["terafold", "calibrate-homography",
           "--image", args.image, "--out", args.out,
           "--table-width", str(args.table_width),
           "--table-height", str(args.table_height)]
    if args.points:
        cal += ["--points", args.points]
    print("Running:", " ".join(cal))
    return subprocess.call(cal)


if __name__ == "__main__":
    sys.exit(main())
