"""Shadow policy: run MolmoAct2 over a *recorded* episode, command nothing.

"Shadow mode" means the policy sees exactly what a live controller would see
(the three camera frames + robot state per timestep) and its predictions are
saved and compared against the recorded actions — but no output ever leaves
the process as a hardware command. This is the safe way to evaluate the model
on the real rig tomorrow: record teleop, then shadow it offline.

Outputs (all JSON, plus an optional Rerun ``.rrd``)::

    <out>/predicted_actions.json   # per-frame predictions
    <out>/comparison.json          # predicted-vs-recorded diffs when comparable
    <out>/metadata.json            # model/dtype/norm_tag/... hardware_commanded: false
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from terafold.yam.config import DEFAULT_MODEL, DEFAULT_TASK
from terafold.yam.dataset import load_episode
from terafold.yam.molmoact2_smoke import (
    AdapterError,
    MockAdapter,
    _default_cfg,
    load_model,
)
from terafold.yam.safety import ShadowModeViolation, assert_no_hardware_execution

__all__ = ["run_shadow_policy"]

SAFETY_STATUS = "shadow: no hardware commanded (execution disabled this sprint)"


def _compare(predicted: Any, actual: Any) -> Dict[str, Any]:
    """Elementwise diff when shapes are comparable; explicit note when not."""
    import numpy as np

    try:
        p = np.asarray(predicted, dtype=float)
        a = np.asarray(actual, dtype=float)
    except Exception:
        return {"comparable": False, "note": "non-numeric action(s)"}
    if p.ndim == 2 and a.ndim == 1 and p.shape[-1] == a.shape[0]:
        p = p[0]  # action-chunk models: compare the first predicted step
    if p.shape != a.shape:
        return {"comparable": False,
                "note": f"shape mismatch: predicted {list(p.shape)} vs recorded {list(a.shape)}"}
    diff = p - a
    return {"comparable": True,
            "l2": round(float(np.linalg.norm(diff)), 6),
            "max_abs": round(float(np.max(np.abs(diff))), 6),
            "mean_abs": round(float(np.mean(np.abs(diff))), 6)}


def run_shadow_policy(
    episode: str,
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    out: str = "runs/yam_shadow_smoke",
    first_frame_only: bool = False,
    max_frames: Optional[int] = None,
    mock: bool = False,
    adapter: Optional[Any] = None,
    use_rerun: bool = True,
    max_new_tokens: int = 256,
    revision: Optional[str] = None,
    log=print,
) -> Dict[str, Any]:
    """Predict actions for a recorded episode. Never commands hardware.

    ``adapter`` lets tests inject a fake; ``mock=True`` uses the built-in
    deterministic :class:`~terafold.yam.molmoact2_smoke.MockAdapter` (no torch).
    Statuses: ``ok`` | ``bad_episode`` | ``bad_config`` | ``deps_missing`` |
    ``model_load_failed`` | ``inference_failed``.
    """
    assert_no_hardware_execution(context="yam-shadow-policy")

    try:
        ep = load_episode(episode)
    except (FileNotFoundError, ValueError) as e:
        return {"status": "bad_episode", "message": str(e),
                "hint": "create one with: terafold yam-create-dummy-episode "
                        "--out data/yam_episodes/towel_fold_smoke/episode_000001 --frames 10"}

    try:
        cfg = _default_cfg()
    except ShadowModeViolation as e:
        return {"status": "bad_config", "message": str(e)}
    meta = ep["metadata"]
    task = meta.get("task") or DEFAULT_TASK
    # The episode's own recording layout wins over the reference config.
    camera_order = meta.get("camera_order") or list(cfg.camera_order)
    norm_tag = meta.get("norm_tag") or cfg.norm_tag
    cfg.camera_order = list(camera_order)
    cfg.norm_tag = norm_tag

    if adapter is None:
        if mock:
            adapter = MockAdapter(cfg, model_name=model_name, dtype_name=dtype)
        else:
            try:
                adapter = load_model(model_name, dtype, cfg=cfg, revision=revision, log=log)
            except ImportError as e:
                return {"status": "deps_missing", "instructions": str(e).splitlines(),
                        "hint": "or re-run with --mock for a dependency-free dry run"}
            except Exception as e:
                return {"status": "model_load_failed", "message": f"{type(e).__name__}: {e}"}

    from terafold.vision.imageio import imread
    from terafold.yam.rerun_logger import YamRerunLogger

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rerun_logger = YamRerunLogger(out_dir=str(out_dir), enabled=use_rerun)
    log(rerun_logger.note())

    n_total = int(meta.get("num_frames") or len(ep["states"]) or 0)
    n = min(1, n_total) if first_frame_only else n_total
    if max_frames is not None:
        n = min(n, int(max_frames))

    predictions: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    for i in range(n):
        images = {}
        for cam in camera_order:
            paths = ep["frames_by_camera"].get(cam) or []
            if i < len(paths):
                try:
                    images[cam] = imread(paths[i])
                except Exception as e:
                    return {"status": "bad_image", "frames_completed": i,
                            "message": f"could not read {cam} frame {i} "
                                       f"({paths[i]}): {e}",
                            "hint": "the episode recording may be truncated/corrupt; "
                                    "re-record or remove the broken frame"}
        state = ep["states"][i] if i < len(ep["states"]) else None
        try:
            pred = adapter.predict(images, robot_state=state, task=task,
                                   max_new_tokens=max_new_tokens)
        except AdapterError as e:
            return {"status": "inference_failed", "message": str(e), "frames_completed": i}
        predictions.append({"frame": i, "action": pred["action"],
                            "raw_text": pred.get("raw_text")})

        recorded = ep["actions"][i]["action"] if i < len(ep["actions"]) else None
        if recorded is not None:
            comparisons.append({"frame": i, **_compare(pred["action"], recorded)})

        pa = pred["action"]
        if isinstance(pa, list) and pa and isinstance(pa[0], list):
            pa = pa[0]  # action-chunk models: visualize the first predicted step
        rerun_logger.log_frame(
            i, images=images, robot_state=state, task=task,
            predicted_action=pa if isinstance(pa, list) else None,
            actual_action=recorded if isinstance(recorded, list) else None,
            safety_status=SAFETY_STATUS)

    (out_dir / "predicted_actions.json").write_text(json.dumps(
        {"episode": ep["dir"], "task": task, "predictions": predictions,
         "hardware_commanded": False}, indent=2))
    (out_dir / "comparison.json").write_text(json.dumps(
        {"episode": ep["dir"], "frames_compared": len(comparisons),
         "comparisons": comparisons}, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps({
        "model": model_name,
        "revision": revision,
        "dtype": dtype,
        "camera_order": camera_order,
        "norm_tag": norm_tag,
        "episode": ep["dir"],
        "frames_predicted": len(predictions),
        "first_frame_only": bool(first_frame_only),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hardware_commanded": False,
        "safety_status": SAFETY_STATUS,
        "adapter": adapter.describe() if hasattr(adapter, "describe") else str(type(adapter)),
        "rerun": rerun_logger.note(),
    }, indent=2))

    log(f"[safety] {SAFETY_STATUS}")
    return {"status": "ok", "out": str(out_dir), "frames_predicted": len(predictions),
            "frames_compared": len(comparisons), "hardware_commanded": False,
            "rerun": rerun_logger.note()}
