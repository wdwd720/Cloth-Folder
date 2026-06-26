#!/usr/bin/env python3
"""Plan a fold and execute it on the mock robot in DRY-RUN (no motion).

Wraps `terafold dry-run-fold --robot mock --camera mock`. This is the
end-to-end Stage-0 smoke test: synthetic camera -> planner -> mock robot.
No physical hardware is touched.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="configs/task_fold_towel_half.yaml")
    p.add_argument("--robot", default="mock", help="mock | learm | so101")
    p.add_argument("--camera", default="mock", help="mock | opencv")
    p.add_argument("--checkpoint", default=None, help="Keypoint model checkpoint.")
    p.add_argument("--calibration", default=None, help="Homography JSON for metric planning.")
    p.add_argument("--viz", default=None, help="Save plan overlay image here.")
    args = p.parse_args()

    cmd = [
        "terafold", "dry-run-fold",
        "--task", args.task,
        "--robot", args.robot,
        "--camera", args.camera,
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.calibration:
        cmd += ["--calibration", args.calibration]
    if args.viz:
        cmd += ["--viz", args.viz]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
