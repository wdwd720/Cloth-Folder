"""Thin wrapper around LeRobot's ACT training entry point.

We do **not** reimplement ACT — we build the exact ``lerobot`` training command
for our exported dataset and either print it (the default, hardware/GPU-free) or
run it as a subprocess when ``run=True`` and lerobot is importable. Importing
this module never requires lerobot, torch, or a GPU.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any, Dict, List, Optional

__all__ = ["train_act", "build_act_command"]

_INSTALL_GUIDANCE = (
    "lerobot is not installed. Install it with: pip install -e '.[lerobot]' "
    "(or `pip install lerobot`). Then re-run with run=True, or copy the printed "
    "command into a machine with a GPU."
)


def build_act_command(
    dataset_dir: str,
    out_dir: str,
    steps: int = 20000,
    batch_size: int = 8,
    device: Optional[str] = None,
    job_name: str = "terafold_act",
    extra: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build the argv for ``lerobot.scripts.train`` configured for ACT."""
    cmd: List[str] = [
        "python",
        "-m",
        "lerobot.scripts.train",
        f"--dataset.repo_id={dataset_dir}",
        "--policy.type=act",
        f"--output_dir={out_dir}",
        f"--job_name={job_name}",
        f"--batch_size={int(batch_size)}",
        f"--steps={int(steps)}",
    ]
    if device:
        cmd.append(f"--policy.device={device}")
    for key, value in (extra or {}).items():
        cmd.append(f"--{key}={value}")
    return cmd


def train_act(
    dataset_dir: str,
    out_dir: str,
    steps: int = 20000,
    batch_size: int = 8,
    run: bool = False,
    device: Optional[str] = None,
    **kw: Any,
) -> Dict[str, Any]:
    """Prepare (and optionally launch) an ACT training run via lerobot.

    Returns a dict with ``status`` one of:

    * ``"dry_run"``       — command built and printed, not executed (default);
    * ``"lerobot_missing"`` — ``run=True`` but lerobot is not importable;
    * ``"completed"`` / ``"failed"`` — subprocess finished (run=True).
    """
    from terafold.robot.so101_adapter import check_lerobot_available

    available = check_lerobot_available()
    argv = build_act_command(
        dataset_dir, out_dir, steps=steps, batch_size=batch_size, device=device,
        extra=kw or None,
    )
    command = " ".join(shlex.quote(a) for a in argv)

    result: Dict[str, Any] = {
        "policy": "act",
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "lerobot_available": bool(available),
        "command": command,
    }

    print("ACT training command:\n  " + command)
    if not available:
        print(_INSTALL_GUIDANCE)

    if not run:
        result["status"] = "dry_run"
        return result

    if not available:
        result["status"] = "lerobot_missing"
        result["guidance"] = _INSTALL_GUIDANCE
        return result

    proc = subprocess.run(argv)
    result["returncode"] = proc.returncode
    result["status"] = "completed" if proc.returncode == 0 else "failed"
    return result
