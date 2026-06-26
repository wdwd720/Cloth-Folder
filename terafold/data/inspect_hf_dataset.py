"""Inspect a public Hugging Face / LeRobot dataset and report its modalities.

The goal is to answer, for a candidate public dataset: what observation/action
modalities does it have, how many episodes, what is the state/action
dimensionality, which embodiment is it, and does its action space plausibly match
our robot? This guides whether the dataset is usable for direct control (rare —
different embodiment) vs. pretraining / representation / scoring (common).

Heavy deps (``huggingface_hub``, ``datasets``, ``lerobot``) are lazy-imported and
the function degrades gracefully with actionable guidance when they are missing
or the machine is offline. The module always imports with numpy only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["inspect_hf_dataset"]


_INSTALL_HINT = (
    "Install dataset tooling with: pip install huggingface_hub datasets "
    "(and 'pip install -e \".[lerobot]\"' for LeRobot datasets)."
)


def _empty_report(repo_id: str) -> Dict[str, Any]:
    return {
        "repo_id": repo_id,
        "available": False,
        "online": None,
        "modalities": [],
        "num_episodes": None,
        "num_frames": None,
        "image_keys": [],
        "video_keys": [],
        "state_dim": None,
        "action_dim": None,
        "embodiment": "unknown",
        "action_space_matches_our_robot": "uncertain",
        "recommended_use": "",
        "error": None,
        "guidance": "",
    }


def inspect_hf_dataset(repo_id: str, our_robot: Optional[object] = None) -> Dict[str, Any]:
    """Inspect ``repo_id`` and return a structured report dict.

    Parameters
    ----------
    repo_id:
        Hugging Face dataset id, e.g. ``"lerobot/aloha_sim_insertion_human"``.
    our_robot:
        Optional :class:`~terafold.config.schema.RobotConfig`. When given, its
        ``dof`` is used to judge whether the dataset's action space could match.
    """
    report = _empty_report(repo_id)

    # 1. Try to read the LeRobot/HF info.json via the hub API (lazy import).
    info = _try_fetch_info(repo_id, report)
    if info is not None:
        _fill_from_lerobot_info(report, info)

    # 2. Embodiment + action-space judgement.
    _judge_action_space(report, our_robot)

    # 3. Recommendation string.
    report["recommended_use"] = _recommend(report, our_robot)

    return report


def _try_fetch_info(repo_id: str, report: Dict[str, Any]) -> Optional[dict]:
    """Attempt to download ``meta/info.json`` for a LeRobot dataset.

    Returns the parsed info dict, or ``None`` (and annotates ``report``) when the
    libs are missing / the machine is offline / the file is absent.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:
        report["error"] = f"huggingface_hub unavailable: {exc}"
        report["guidance"] = _INSTALL_HINT
        return None

    import json

    last: Optional[Exception] = None
    for candidate in ("meta/info.json", "info.json"):
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=candidate, repo_type="dataset"
            )
            report["available"] = True
            report["online"] = True
            with open(path) as f:
                return json.load(f)
        except Exception as exc:  # offline, 404, auth, etc.
            last = exc
    report["online"] = not _looks_offline(last) if last is not None else None
    report["error"] = f"Could not fetch info.json: {last}"
    report["guidance"] = (
        "Dataset metadata not reachable. Check the repo id, network access, and "
        "that it is a LeRobot-format dataset. " + _INSTALL_HINT
    )
    return None


def _looks_offline(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("connection", "offline", "timeout", "resolve", "network"))


def _fill_from_lerobot_info(report: Dict[str, Any], info: dict) -> None:
    """Populate the report from a LeRobot ``info.json`` features schema."""
    report["available"] = True
    report["num_episodes"] = info.get("total_episodes")
    report["num_frames"] = info.get("total_frames")
    report["embodiment"] = info.get("robot_type", "unknown") or "unknown"
    report["fps"] = info.get("fps")

    features = info.get("features", {}) or {}
    modalities: List[str] = []
    image_keys: List[str] = []
    video_keys: List[str] = []
    for key, spec in features.items():
        dtype = (spec or {}).get("dtype", "")
        if dtype == "image" or ".images." in key:
            image_keys.append(key)
        elif dtype == "video" or ".videos." in key or key.startswith("observation.video"):
            video_keys.append(key)
        modalities.append(key)
        shape = (spec or {}).get("shape")
        if key == "observation.state" and shape:
            report["state_dim"] = int(shape[0]) if shape else None
        if key == "action" and shape:
            report["action_dim"] = int(shape[0]) if shape else None

    report["modalities"] = modalities
    report["image_keys"] = image_keys
    report["video_keys"] = video_keys


def _judge_action_space(report: Dict[str, Any], our_robot: Optional[object]) -> None:
    """Set ``action_space_matches_our_robot`` to True/False/'uncertain'."""
    action_dim = report.get("action_dim")
    our_dof = getattr(our_robot, "dof", None) if our_robot is not None else None
    if action_dim is None or our_dof is None:
        report["action_space_matches_our_robot"] = "uncertain"
        return
    # A loose match: action dim within our DoF (+ a gripper channel) range.
    report["action_space_matches_our_robot"] = bool(
        action_dim in (our_dof, our_dof + 1)
    )


def _recommend(report: Dict[str, Any], our_robot: Optional[object]) -> str:
    if not report.get("available"):
        return (
            "Could not inspect the dataset. If reachable and LeRobot-format, it is "
            "most likely useful for pretraining / representation learning, not for "
            "direct control of the TeraFold robot (different embodiment)."
        )
    match = report.get("action_space_matches_our_robot")
    embod = report.get("embodiment", "unknown")
    if match is True:
        return (
            f"Action space ({report.get('action_dim')}) plausibly matches our robot "
            f"(embodiment={embod}). Candidate for fine-tuning, but still VERIFY frame "
            "conventions and units before any execution."
        )
    if match is False:
        return (
            f"Action space ({report.get('action_dim')}) differs from our robot "
            f"(embodiment={embod}). Use for visual/representation pretraining and "
            "success scoring — do NOT replay its actions on our hardware."
        )
    return (
        "Embodiment/action match is uncertain. Default to pretraining / "
        "representation use; never replay foreign actions on our robot without "
        "explicit re-mapping and validation."
    )
