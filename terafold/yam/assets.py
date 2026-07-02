"""Asset/config validation for the YAM/MolmoAct2 stack (``yam-check-assets``).

Pure filesystem + YAML checks — never touches hardware, never imports torch.
Each check yields ``{name, status, detail, fix?}`` with status ``PASS`` /
``WARN`` / ``FAIL``:

* ``PASS`` — present and looks right.
* ``WARN`` — optional asset not configured yet (``fix`` holds the exact
  clone/download command). Missing hardware/assets never hard-fail the check.
* ``FAIL`` — the config itself is missing/invalid, or a *configured* path does
  not exist (the config claims something that is not true).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from terafold.yam.safety import ShadowModeViolation

__all__ = ["I2RT_CLONE_CMD", "check_assets", "summarize_status"]

I2RT_REPO_URL = "https://github.com/i2rt-robotics/i2rt.git"
I2RT_CLONE_CMD = f"git clone {I2RT_REPO_URL} ~/i2rt   # then set paths.i2rt_repo in the config"


def _download_cmd(model_name: str, target: str = "~/models/molmoact2_bimanual_yam") -> str:
    return (
        "pip install -U 'huggingface_hub[cli]' && "
        f"huggingface-cli download {model_name} --local-dir {target}"
        "   # then set paths.molmoact2_checkpoint (optional: the HF cache also works)"
    )


def _check(name: str, status: str, detail: str, fix: Optional[str] = None) -> Dict[str, Any]:
    c: Dict[str, Any] = {"name": name, "status": status, "detail": detail}
    if fix:
        c["fix"] = fix
    return c


def _expand(p: Any) -> Optional[Path]:
    if p is None or p == "":
        return None
    return Path(str(p)).expanduser()


def _check_path(name: str, raw: Any, kind: str, fix_if_unset: str) -> Dict[str, Any]:
    """Shared logic: unset -> WARN with fix; set+missing -> FAIL; set+present -> PASS."""
    p = _expand(raw)
    if p is None:
        return _check(name, "WARN", f"{kind} not configured yet", fix_if_unset)
    if not p.exists():
        return _check(name, "FAIL", f"configured {kind} does not exist: {p}", fix_if_unset)
    return _check(name, "PASS", f"{kind} found: {p}")


def summarize_status(checks: List[Dict[str, Any]]) -> str:
    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def check_assets(config_path: str) -> Dict[str, Any]:
    """Validate the YAM config + configured assets. Never raises; never needs hardware.

    Returns ``{status, checks, config?, camera_order?, norm_tag?, model?}``.
    """
    from terafold.yam.config import load_yam_config  # local import keeps module cheap

    checks: List[Dict[str, Any]] = []

    try:
        cfg = load_yam_config(config_path)
    except FileNotFoundError as e:
        checks.append(_check("config", "FAIL", str(e),
                             "expected configs/yam_dual_reference.yaml — pass --config"))
        return {"status": "FAIL", "checks": checks}
    except ShadowModeViolation as e:
        checks.append(_check("config", "FAIL", f"shadow-mode violation: {e}",
                             "set autonomous_execution_enabled: false"))
        return {"status": "FAIL", "checks": checks}
    except Exception as e:  # malformed YAML / bad values
        checks.append(_check("config", "FAIL", f"config does not parse: {e}"))
        return {"status": "FAIL", "checks": checks}

    checks.append(_check("config", "PASS", f"parsed {cfg.source} (robot={cfg.robot})"))
    checks.append(_check("shadow_mode", "PASS",
                         "autonomous_execution_enabled=false (real execution forbidden)"))
    checks.append(_check("camera_order", "PASS", f"camera_order={cfg.camera_order}"))
    checks.append(_check("norm_tag", "PASS", f"norm_tag={cfg.norm_tag}"))
    checks.append(_check("model", "PASS", f"model={cfg.model_name} (dtype={cfg.model_dtype})"))
    checks.append(_check("action_space", "PASS",
                         f"action_dim={cfg.action_dim} "
                         f"({len(cfg.arms)} arms x ({cfg.joints_per_arm} joints "
                         f"+ {cfg.grippers_per_arm} gripper)) — placeholder, verify vs I2RT specs"))

    paths = cfg.paths or {}
    checks.append(_check_path("i2rt_repo", paths.get("i2rt_repo"), "I2RT repo", I2RT_CLONE_CMD))
    checks.append(_check_path("molmoact2_checkpoint", paths.get("molmoact2_checkpoint"),
                              "MolmoAct2 checkpoint dir", _download_cmd(cfg.model_name)))
    checks.append(_check_path(
        "urdf", paths.get("urdf"), "URDF",
        "URDFs usually live inside the I2RT repo — clone it first, then set paths.urdf"))
    checks.append(_check_path(
        "usd", paths.get("usd"), "USD",
        "USD is optional (Isaac scene falls back to placeholder arms); set paths.usd "
        "after converting the YAM URDF"))
    checks.append(_check_path(
        "mjcf", paths.get("mjcf"), "MJCF",
        "MJCF is optional this sprint (digital twin later); set paths.mjcf when you have one"))

    # Camera device paths: purely informational — hardware absence is never a failure.
    def _cam_path(entry: Any) -> Any:
        # tolerate both `top: {path: /dev/video0}` and shorthand `top: /dev/video0`
        return entry.get("path") if isinstance(entry, dict) else entry

    unset = [name for name in cfg.camera_order if not _cam_path(cfg.cameras.get(name))]
    if unset:
        checks.append(_check("camera_paths", "WARN",
                             f"camera device paths not set for: {unset} "
                             "(fine off-rig; fill in tomorrow on the real setup)"))
    else:
        checks.append(_check("camera_paths", "PASS", "all camera device paths configured"))

    return {
        "status": summarize_status(checks),
        "checks": checks,
        "config": str(cfg.source),
        "camera_order": cfg.camera_order,
        "norm_tag": cfg.norm_tag,
        "model": cfg.model_name,
    }
