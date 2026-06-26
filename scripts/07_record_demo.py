#!/usr/bin/env python3
"""Record a folding demonstration episode.

Wraps `terafold record-demo`. Defaults to a fully DRY-RUN mock setup
(mock robot + mock camera). Physical motion is blocked unless you pass BOTH
--enable-motion and --i-understand-this-moves-hardware.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="configs/task_fold_towel_half.yaml")
    p.add_argument("--robot", default="mock", help="mock | learm | so101")
    p.add_argument("--camera", default="mock", help="webcam | mock")
    p.add_argument("--out", default=None, help="Episodes root (default data/episodes/<task>).")
    p.add_argument("--operator", default="unknown")
    p.add_argument("--enable-motion", action="store_true", help="Allow physical motion.")
    p.add_argument(
        "--i-understand-this-moves-hardware",
        action="store_true",
        help="Required second motion flag.",
    )
    args = p.parse_args()

    cmd = [
        "terafold", "record-demo",
        "--task", args.task,
        "--robot", args.robot,
        "--camera", args.camera,
        "--operator", args.operator,
    ]
    if args.out:
        cmd += ["--out", args.out]
    if args.enable_motion:
        cmd.append("--enable-motion")
    if args.i_understand_this_moves_hardware:
        cmd.append("--i-understand-this-moves-hardware")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
