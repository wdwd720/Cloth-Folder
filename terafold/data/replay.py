"""Load, replay, and summarize recorded episodes (numpy only).

:func:`load_episode` reads a whole episode directory into a plain dict.
:func:`replay_episode` re-executes the recorded actions (optionally on a robot,
dry-run by default) and reports a summary. :func:`summarize_episode` formats a
one-screen human-readable digest. No torch / pandas required.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from terafold.data.episode_schema import EpisodePaths, read_json, read_jsonl
from terafold.robot.base import BaseRobot, RobotAction

__all__ = ["load_episode", "replay_episode", "summarize_episode"]


def _frame_paths(paths: EpisodePaths) -> List[str]:
    if not os.path.isdir(paths.frames_dir):
        return []
    names = sorted(n for n in os.listdir(paths.frames_dir) if n.endswith(".png"))
    return [os.path.join(paths.frames_dir, n) for n in names]


def load_episode(episode_dir: str) -> Dict[str, Any]:
    """Read an episode directory into a dict of its components.

    Keys: ``root``, ``metadata``, ``observations`` (list of dicts),
    ``actions`` (list of dicts), ``keypoints`` (list of dicts), ``plan``,
    ``result``, ``safety``, ``notes`` and ``frame_paths``.
    """
    paths = EpisodePaths(episode_dir)
    if not os.path.isdir(episode_dir):
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    metadata = read_json(paths.metadata) if os.path.exists(paths.metadata) else None
    plan = read_json(paths.plan) if os.path.exists(paths.plan) else None
    result = read_json(paths.result) if os.path.exists(paths.result) else None
    safety = read_json(paths.safety) if os.path.exists(paths.safety) else None
    notes = ""
    if os.path.exists(paths.notes):
        with open(paths.notes) as f:
            notes = f.read()

    return {
        "root": episode_dir,
        "metadata": metadata,
        "observations": list(read_jsonl(paths.observations)),
        "actions": list(read_jsonl(paths.actions)),
        "keypoints": list(read_jsonl(paths.keypoints)),
        "plan": plan,
        "result": result,
        "safety": safety,
        "notes": notes,
        "frame_paths": _frame_paths(paths),
    }


def replay_episode(
    episode_dir: str,
    robot: Optional[BaseRobot] = None,
    visualize: bool = False,
) -> Dict[str, Any]:
    """Replay an episode's recorded actions and return a summary.

    If ``robot`` is given, each non-empty action is re-sent to it (the robot's
    own dry-run / safety settings govern motion). If ``visualize`` is set, a best
    -effort plan overlay is rendered onto the first frame under
    ``<episode>/replay/`` (degrades silently if visualization is unavailable).
    """
    data = load_episode(episode_dir)
    actions = [RobotAction.from_dict(a) for a in data["actions"] if a]

    executed = 0
    errors: List[str] = []
    if robot is not None:
        connected_here = not robot.is_connected
        if connected_here:
            robot.connect()
        try:
            for action in actions:
                try:
                    robot.send_action(action)
                    executed += 1
                except Exception as exc:  # surface but keep going.
                    errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if connected_here:
                robot.disconnect()

    vis_path: Optional[str] = None
    if visualize:
        vis_path = _maybe_visualize(episode_dir, data)

    result = data.get("result") or {}
    return {
        "root": episode_dir,
        "num_frames": len(data["frame_paths"]),
        "num_actions": len(actions),
        "actions_executed": executed,
        "errors": errors,
        "success": result.get("success") if isinstance(result, dict) else None,
        "failure_mode": result.get("failure_mode") if isinstance(result, dict) else None,
        "visualization": vis_path,
    }


def _maybe_visualize(episode_dir: str, data: Dict[str, Any]) -> Optional[str]:
    """Render a plan overlay onto the first frame; return path or ``None``."""
    frame_paths = data["frame_paths"]
    if not frame_paths or not data.get("plan"):
        return None
    try:
        from terafold.planning.fold_plan import FoldPlan
        from terafold.vision.imageio import imread
        from terafold.vision.visualization import draw_fold_plan, save_visualization

        plan = FoldPlan.from_dict(data["plan"])
        img = imread(frame_paths[0])
        overlay = draw_fold_plan(img, plan)
        out_dir = os.path.join(episode_dir, "replay")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "plan_overlay.png")
        save_visualization(out_path, overlay)
        return out_path
    except Exception:
        return None


def summarize_episode(episode_dir: str) -> str:
    """Return a concise human-readable summary of an episode."""
    data = load_episode(episode_dir)
    meta = data.get("metadata") or {}
    result = data.get("result") or {}
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

    lines = [
        f"Episode: {data['root']}",
        f"  task        : {meta.get('task_name', '?')} — {meta.get('task_instruction', '')}",
        f"  robot/cam   : {meta.get('robot_type', '?')} / {meta.get('camera_type', '?')}"
        f"  (dry_run={meta.get('dry_run')})",
        f"  frames      : {len(data['frame_paths'])}"
        f"  actions: {sum(1 for a in data['actions'] if a)}",
        f"  start/end   : {meta.get('start_time', '?')} -> {meta.get('end_time', '?')}",
        f"  success     : {result.get('success') if isinstance(result, dict) else None}"
        f"  failure: {result.get('failure_mode') if isinstance(result, dict) else None}",
    ]
    if metrics:
        metric_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in metrics.items()
        )
        lines.append(f"  metrics     : {metric_str}")
    safety = data.get("safety")
    if isinstance(safety, dict) and safety.get("num_violations"):
        lines.append(f"  safety      : {safety['num_violations']} workspace violation(s)")
    return "\n".join(lines)
