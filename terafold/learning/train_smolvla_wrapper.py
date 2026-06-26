"""Thin wrapper around LeRobot's SmolVLA fine-tuning entry point.

Analogous to :mod:`terafold.learning.train_act_wrapper`: we build the exact
``lerobot`` command to fine-tune the SmolVLA base policy on our exported dataset
and either print it (default) or run it (``run=True`` and lerobot available).
Importing this module never requires lerobot, torch, or a GPU.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any, Dict, List, Optional

__all__ = ["train_smolvla", "build_smolvla_command"]

_INSTALL_GUIDANCE = (
    "lerobot (with the SmolVLA extra) is not installed. Install it with: "
    "pip install -e '.[lerobot]' (or `pip install lerobot`), then re-run with "
    "run=True or copy the printed command onto a GPU machine. SmolVLA fine-tuning "
    "starts from the `lerobot/smolvla_base` checkpoint."
)


def build_smolvla_command(
    dataset_dir: str,
    out_dir: str,
    steps: int = 20000,
    batch_size: int = 8,
    device: Optional[str] = None,
    base_policy: str = "lerobot/smolvla_base",
    job_name: str = "terafold_smolvla",
    extra: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build the argv for ``lerobot.scripts.train`` configured for SmolVLA."""
    cmd: List[str] = [
        "python",
        "-m",
        "lerobot.scripts.train",
        f"--dataset.repo_id={dataset_dir}",
        f"--policy.path={base_policy}",
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


def train_smolvla(
    dataset_dir: str,
    out_dir: str,
    steps: int = 20000,
    batch_size: int = 8,
    run: bool = False,
    device: Optional[str] = None,
    base_policy: str = "lerobot/smolvla_base",
    **kw: Any,
) -> Dict[str, Any]:
    """Prepare (and optionally launch) a SmolVLA fine-tune via lerobot.

    Returns a dict whose ``status`` is one of ``"dry_run"``,
    ``"lerobot_missing"``, ``"completed"``, or ``"failed"``.
    """
    from terafold.robot.so101_adapter import check_lerobot_available

    available = check_lerobot_available()
    argv = build_smolvla_command(
        dataset_dir, out_dir, steps=steps, batch_size=batch_size, device=device,
        base_policy=base_policy, extra=kw or None,
    )
    command = " ".join(shlex.quote(a) for a in argv)

    result: Dict[str, Any] = {
        "policy": "smolvla",
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "base_policy": base_policy,
        "lerobot_available": bool(available),
        "command": command,
    }

    print("SmolVLA fine-tune command:\n  " + command)
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
