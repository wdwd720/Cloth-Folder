"""YAM episode structure: create dummy episodes, load them back.

An episode directory is the on-disk contract between recording (real teleop
later, dummy now), the shadow policy, and future LeRobot export::

    episode_XXXXXX/
      metadata.json          # task, camera_order, norm_tag, dims, hardware_commanded
      states.jsonl           # one line per frame: per-arm joints + gripper + state_vector
      actions.jsonl          # one line per frame: action vector (absolute joint pose)
      cameras/top/frame_000000.png ...
      cameras/left/ ...
      cameras/right/ ...

Vector layout is ``YamDualConfig.state_layout()`` (left joints, left gripper,
right joints, right gripper). Dummy data is deterministic (seeded sinusoids) so
tests and the shadow-policy comparison are reproducible. Images are written
with :mod:`terafold.vision.imageio`, which falls back to a pure-Python PNG
codec — no cv2/Pillow needed.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from terafold.yam.config import DEFAULT_TASK, YamDualConfig, load_yam_config

__all__ = ["create_dummy_episode", "load_episode"]

PathLike = Union[str, Path]


def _dummy_image(camera: str, frame: int, num_frames: int, hw=(240, 320)):
    """Deterministic per-camera gradient + a marker square that moves per frame."""
    import numpy as np

    h, w = hw
    base = {"top": (40, 90, 160), "left": (160, 90, 40), "right": (60, 150, 60)}.get(
        camera, (120, 120, 120)
    )
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = base[c] * (0.5 + 0.5 * (yy + xx) / 2.0)
    # moving white square = visible frame index for eyeballing in Rerun
    t = frame / max(1, num_frames - 1)
    cx, cy = int((0.15 + 0.7 * t) * w), int(0.5 * h)
    s = max(4, h // 16)
    img[max(0, cy - s):cy + s, max(0, cx - s):cx + s, :] = 255.0
    return img.clip(0, 255).astype(np.uint8)


def _dummy_arm_state(joints: int, frame: int, num_frames: int, phase: float):
    """Smooth small sinusoids around a fixed home pose (radians)."""
    t = frame / max(1, num_frames - 1)
    home = [0.0, -0.4, 0.6, 0.0, 0.5, 0.0][:joints]
    home += [0.0] * (joints - len(home))
    jp = [round(h + 0.05 * math.sin(2 * math.pi * (t + phase) + 0.7 * i), 6)
          for i, h in enumerate(home)]
    grip = round(0.5 + 0.4 * math.sin(2 * math.pi * (t + phase)), 6)
    return jp, grip


def create_dummy_episode(
    out_dir: PathLike,
    frames: int = 10,
    task: str = DEFAULT_TASK,
    seed: int = 0,
    fps: float = 10.0,
    image_hw=(240, 320),
    config: Optional[YamDualConfig] = None,
) -> Dict[str, Any]:
    """Write a complete dummy episode (no hardware anywhere near this)."""
    from terafold.vision.imageio import imwrite

    cfg = config or load_yam_config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    states: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    for f in range(frames):
        ts = round(t0 + f / fps, 6)
        arms: Dict[str, Any] = {}
        state_vec: List[float] = []
        action_vec: List[float] = []
        for i, arm in enumerate(cfg.arms):
            jp, grip = _dummy_arm_state(cfg.joints_per_arm, f, frames, phase=0.25 * i)
            arms[arm] = {"joint_pos_rad": jp, "gripper": grip}
            state_vec += jp + [grip] * cfg.grippers_per_arm
            # dummy "action" = next-step absolute pose (same trajectory, small lead)
            jp_n, grip_n = _dummy_arm_state(cfg.joints_per_arm,
                                            min(f + 1, frames - 1), frames, phase=0.25 * i)
            action_vec += jp_n + [grip_n] * cfg.grippers_per_arm
        states.append({"frame": f, "timestamp": ts, "arms": arms, "state_vector": state_vec})
        actions.append({"frame": f, "timestamp": ts, "action": action_vec,
                        "hardware_commanded": False})

    with (out / "states.jsonl").open("w") as fh:
        for s in states:
            fh.write(json.dumps(s) + "\n")
    with (out / "actions.jsonl").open("w") as fh:
        for a in actions:
            fh.write(json.dumps(a) + "\n")

    for cam in cfg.camera_order:
        cam_dir = out / "cameras" / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f in range(frames):
            imwrite(str(cam_dir / f"frame_{f:06d}.png"),
                    _dummy_image(cam, f, frames, hw=image_hw))

    metadata = {
        "episode_id": out.name,
        "task": task,
        "success": None,
        "hardware_commanded": False,
        "dummy": True,
        "robot": cfg.robot,
        "arms": cfg.arms,
        "camera_order": cfg.camera_order,
        "norm_tag": cfg.norm_tag,
        "control_mode": cfg.control_mode,
        "action_mode": cfg.action_mode,
        "num_frames": frames,
        "fps": fps,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "state_layout": cfg.state_layout(),
        "image_hw": list(image_hw),
        "seed": seed,
        "created_unix": round(t0, 3),
    }
    with (out / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    return {"status": "ok", "out": str(out), "frames": frames,
            "cameras": list(cfg.camera_order), "action_dim": cfg.action_dim}


def load_episode(episode_dir: PathLike) -> Dict[str, Any]:
    """Load an episode directory back into dicts + per-camera frame paths.

    Raises ``FileNotFoundError``/``ValueError`` with a clear message if the
    directory is not a valid episode.
    """
    ep = Path(episode_dir)
    meta_path = ep / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"not an episode dir (no metadata.json): {ep}")
    metadata = json.loads(meta_path.read_text())

    def _jsonl(name: str) -> List[Dict[str, Any]]:
        p = ep / name
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    states = _jsonl("states.jsonl")
    actions = _jsonl("actions.jsonl")

    frames_by_camera: Dict[str, List[str]] = {}
    for cam in metadata.get("camera_order", []):
        cam_dir = ep / "cameras" / cam
        frames_by_camera[cam] = (
            sorted(str(p) for p in cam_dir.glob("frame_*.png")) if cam_dir.is_dir() else []
        )

    n = metadata.get("num_frames")
    counts = {"states": len(states), "actions": len(actions),
              **{f"cameras/{c}": len(v) for c, v in frames_by_camera.items()}}
    if n is not None and any(v not in (0, n) for v in counts.values()):
        raise ValueError(f"episode {ep} is inconsistent: num_frames={n} but {counts}")

    return {"dir": str(ep), "metadata": metadata, "states": states,
            "actions": actions, "frames_by_camera": frames_by_camera}
