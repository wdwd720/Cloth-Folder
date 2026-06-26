#!/usr/bin/env python3
"""Compute a fold plan from a camera image (no robot motion).

Wraps `terafold plan-fold`. Supply --image to plan from a still, or omit it
to grab a frame from --camera (mock by default). Pass --calibration to plan
in metric/table coordinates, and --out / --viz to save the plan and overlay.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default=None, help="Input image (else use --camera).")
    p.add_argument("--task", default="configs/task_fold_towel_half.yaml")
    p.add_argument("--camera", default="mock", help="mock | opencv (used if --image omitted).")
    p.add_argument("--checkpoint", default=None, help="Keypoint model checkpoint.")
    p.add_argument("--use-residual", default=None, help="Residual model checkpoint.")
    p.add_argument("--calibration", default=None, help="Homography JSON for metric planning.")
    p.add_argument("--out", default=None, help="Save plan.json here.")
    p.add_argument("--viz", default=None, help="Save plan overlay image here.")
    args = p.parse_args()

    cmd = ["terafold", "plan-fold", "--task", args.task, "--camera", args.camera]
    if args.image:
        cmd += ["--image", args.image]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.use_residual:
        cmd += ["--use-residual", args.use_residual]
    if args.calibration:
        cmd += ["--calibration", args.calibration]
    if args.out:
        cmd += ["--out", args.out]
    if args.viz:
        cmd += ["--viz", args.viz]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
