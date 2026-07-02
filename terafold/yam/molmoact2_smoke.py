"""MolmoAct2-BimanualYAM smoke test: load model, predict once, save JSON.

Design rules:

* torch / transformers are imported **only** inside :func:`load_model` — the
  module imports clean on a laptop with nothing but numpy installed.
* All model access goes through one clearly isolated adapter
  (:class:`MolmoAct2Adapter.predict`), so tests (and the ``--mock`` CLI flag)
  can swap in :class:`MockAdapter` without touching any of the plumbing.
* Nothing here can command hardware: outputs are JSON files, and the metadata
  always records ``hardware_commanded: false``.

The real inference path follows the MolmoAct2-BimanualYAM model-card recipe
(AutoProcessor/AutoModelForImageTextToText with ``trust_remote_code``, then
``model.predict_action(...)``; older MolmoAct generate+``parse_action`` is the
fallback); if the checkpoint's remote code matches neither, the adapter fails
with a pointer to the model card instead of a stack trace.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from terafold.yam.config import (
    DEFAULT_MODEL,
    DEFAULT_TASK,
    YamDualConfig,
    load_yam_config,
)
from terafold.yam.safety import ShadowModeViolation, assert_no_hardware_execution

__all__ = [
    "SUPPORTED_DTYPES",
    "AdapterError",
    "MolmoAct2Adapter",
    "MockAdapter",
    "install_instructions",
    "load_model",
    "make_dummy_images",
    "make_dummy_robot_state",
    "run_smoke_test",
]

SUPPORTED_DTYPES = ("bfloat16", "float32")

MODEL_CARD_URL = "https://huggingface.co/{model}"


class AdapterError(RuntimeError):
    """Model loaded but the inference recipe did not fit — see the model card."""


def install_instructions() -> List[str]:
    """Exact commands to make :func:`load_model` work."""
    return [
        "MolmoAct2 dependencies are missing. Install ONE of:",
        "  pip install -e '.[molmoact]'",
        "  pip install torch transformers accelerate einops pillow",
        "On a CUDA RunPod pod (torch preinstalled):",
        "  pip install transformers accelerate einops pillow",
        "Then re-run this command. No GPU? Use --dtype float32 (slow) or --mock.",
    ]


# ---------------------------------------------------------------------------
# Dummy inputs
# ---------------------------------------------------------------------------


def make_dummy_images(camera_order, hw=(224, 224)) -> Dict[str, Any]:
    """Deterministic RGB test cards, one per camera, keyed by camera name."""
    import numpy as np

    h, w = hw
    images: Dict[str, Any] = {}
    for k, cam in enumerate(camera_order):
        yy = np.linspace(0, 255, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 255, w, dtype=np.float32)[None, :]
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[:, :, k % 3] = 0.7 * yy + 0.3 * xx
        img[:, :, (k + 1) % 3] = 0.3 * yy
        images[cam] = img.clip(0, 255).astype(np.uint8)
    return images


def make_dummy_robot_state(cfg: YamDualConfig) -> Dict[str, Any]:
    """Zero-pose robot state with the same structure/shape episodes use."""
    arms = {
        arm: {"joint_pos_rad": [0.0] * cfg.joints_per_arm,
              "gripper": 0.5 if cfg.grippers_per_arm else None}
        for arm in cfg.arms
    }
    vec: List[float] = []
    for arm in cfg.arms:
        vec += arms[arm]["joint_pos_rad"] + ([arms[arm]["gripper"]] if cfg.grippers_per_arm else [])
    return {"arms": arms, "state_vector": vec, "layout": cfg.state_layout()}


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class MolmoAct2Adapter:
    """Isolated wrapper around a loaded MolmoAct2 model + processor."""

    def __init__(self, model, processor, dtype_name: str, camera_order, norm_tag: str,
                 model_name: str) -> None:
        self.model = model
        self.processor = processor
        self.dtype_name = dtype_name
        self.camera_order = list(camera_order)
        self.norm_tag = norm_tag
        self.model_name = model_name
        self.kind = "molmoact2"

    def _to_pil(self, images: Dict[str, Any]):
        from PIL import Image
        import numpy as np

        return [Image.fromarray(np.asarray(images[c]).astype("uint8")) for c in self.camera_order]

    def predict(self, images: Dict[str, Any], robot_state: Optional[Dict[str, Any]] = None,
                task: str = DEFAULT_TASK, max_new_tokens: int = 256) -> Dict[str, Any]:
        """One forward pass: (top/left/right images, state, task) -> action dict.

        Returns ``{"action": nested list, "raw_text": str|None}``. Never touches
        hardware. Primary path is the MolmoAct2-BimanualYAM model-card API
        (``model.predict_action(processor=..., images=..., task=..., state=...,
        norm_tag=...)``); falls back to the older MolmoAct generate +
        ``parse_action`` recipe. Raises :class:`AdapterError` if the checkpoint's
        remote-code API matches neither.
        """
        import contextlib

        import torch

        try:
            pil_images = self._to_pil(images)
            state_vec = (robot_state or {}).get("state_vector") if isinstance(robot_state, dict) \
                else robot_state
            amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if self.dtype_name == "bfloat16" and torch.cuda.is_available()
                   else contextlib.nullcontext())
            if hasattr(self.model, "predict_action"):
                # MolmoAct2-BimanualYAM model-card recipe.
                with torch.inference_mode(), amp:
                    action = self.model.predict_action(
                        processor=self.processor, images=pil_images, task=task,
                        state=state_vec, norm_tag=self.norm_tag)
                return {"action": _tolist(action), "raw_text": None}
            if hasattr(self.model, "parse_action"):
                # Older MolmoAct recipe: prompt -> generate -> parse_action.
                prompt = f"The task is {task}. What is the action the robot should take?"
                inputs = self.processor(images=pil_images, text=prompt, return_tensors="pt")
                inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                with torch.inference_mode(), amp:
                    generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
                action = self.model.parse_action(text, unnorm_key=self.norm_tag)
                return {"action": _tolist(action), "raw_text": text}
            raise AdapterError(
                f"loaded model has neither predict_action() nor parse_action(); the "
                f"{self.model_name} remote code differs from the known MolmoAct recipes — "
                f"check {MODEL_CARD_URL.format(model=self.model_name)} for the official "
                "inference snippet and adapt MolmoAct2Adapter.predict()."
            )
        except AdapterError:
            raise
        except Exception as e:
            raise AdapterError(
                f"MolmoAct2 inference failed ({type(e).__name__}: {e}). The exact "
                f"inference recipe lives in the model card: "
                f"{MODEL_CARD_URL.format(model=self.model_name)} — adapt "
                "MolmoAct2Adapter.predict() if the remote-code API changed."
            ) from e

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "model": self.model_name, "dtype": self.dtype_name,
                "camera_order": self.camera_order, "norm_tag": self.norm_tag}


class MockAdapter:
    """Deterministic stand-in used by ``--mock`` and the tests (no torch)."""

    def __init__(self, cfg: YamDualConfig, model_name: str = "mock", dtype_name: str = "mock"):
        self.cfg = cfg
        self.model_name = model_name
        self.dtype_name = dtype_name
        self.camera_order = list(cfg.camera_order)
        self.norm_tag = cfg.norm_tag
        self.kind = "mock"
        self.calls: List[Dict[str, Any]] = []  # inspectable by tests

    def predict(self, images, robot_state=None, task=DEFAULT_TASK, max_new_tokens=0):
        self.calls.append({"cameras": sorted(images), "task": task})
        base = robot_state.get("state_vector") if isinstance(robot_state, dict) else None
        vec = list(base) if base else [0.0] * self.cfg.action_dim
        action = [round(v + 0.01 * ((i % 3) - 1), 6) for i, v in enumerate(vec)]
        return {"action": action, "raw_text": f"mock action for task: {task}"}

    def describe(self):
        return {"kind": self.kind, "model": self.model_name, "dtype": self.dtype_name,
                "camera_order": self.camera_order, "norm_tag": self.norm_tag}


def _tolist(x: Any):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_tolist(v) for v in x]
    return x


def _action_shape(action: Any) -> List[int]:
    import numpy as np

    try:
        return list(np.asarray(action, dtype=float).shape)
    except Exception:
        return [len(action)] if hasattr(action, "__len__") else []


# ---------------------------------------------------------------------------
# Model loading (the only place torch/transformers are imported)
# ---------------------------------------------------------------------------


def load_model(model_name: str = DEFAULT_MODEL, dtype: str = "bfloat16",
               device: Optional[str] = None, cfg: Optional[YamDualConfig] = None,
               revision: Optional[str] = None, log=print) -> MolmoAct2Adapter:
    """Load MolmoAct2 + processor. Raises ImportError (with exact install
    commands in the message) when torch/transformers are absent.

    ``trust_remote_code=True`` is required by the MolmoAct recipe, which means
    hub code runs in-process. Containment: pass ``revision=<commit sha>`` to pin
    it, or point ``model_name`` at a local snapshot dir — local paths are loaded
    with ``local_files_only=True`` so nothing is fetched. On the real rig, do
    one of those two (see docs/YAM_MOLMOACT2_QUICKSTART.md).
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"--dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")
    cfg = cfg or _default_cfg()
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError("\n".join(install_instructions())) from e

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    device_map = device or ("auto" if torch.cuda.is_available() else None)
    shared: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if revision:
        shared["revision"] = revision
    if Path(model_name).expanduser().is_dir():
        shared["local_files_only"] = True  # local snapshot: never touch the hub
    log(f"Loading {model_name} (dtype={dtype}, device_map={device_map}, "
        f"revision={revision or 'unpinned!'}) ...")
    processor = AutoProcessor.from_pretrained(model_name, padding_side="left", **shared)
    kwargs = dict(shared)
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model.eval()
    return MolmoAct2Adapter(model, processor, dtype, cfg.camera_order, cfg.norm_tag, model_name)


def _default_cfg() -> YamDualConfig:
    try:
        return load_yam_config()
    except ShadowModeViolation:
        raise  # a tampered reference config must be refused, never masked
    except Exception:
        return YamDualConfig()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test(
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    out: str = "runs/yam_molmoact2_smoke/actions.json",
    top_image: Optional[str] = None,
    left_image: Optional[str] = None,
    right_image: Optional[str] = None,
    task: str = DEFAULT_TASK,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 256,
    mock: bool = False,
    revision: Optional[str] = None,
    log=print,
) -> Dict[str, Any]:
    """Run one MolmoAct2 prediction on real-or-dummy images and save JSON.

    Returns a status dict; never raises for missing deps and never commands
    hardware. Statuses: ``ok`` | ``deps_missing`` | ``bad_dtype`` |
    ``bad_config`` | ``model_load_failed`` | ``inference_failed`` | ``bad_image``.
    """
    assert_no_hardware_execution(context="yam-molmoact2-smoke-test")
    if dtype not in SUPPORTED_DTYPES and not mock:
        return {"status": "bad_dtype",
                "message": f"--dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}"}
    try:
        cfg = load_yam_config(config_path) if config_path else _default_cfg()
    except ShadowModeViolation as e:
        return {"status": "bad_config", "message": str(e)}
    except (FileNotFoundError, ValueError) as e:
        return {"status": "bad_config", "message": str(e)}

    # Images: provided paths win; anything unspecified gets a dummy test card.
    provided = {"top": top_image, "left": left_image, "right": right_image}
    images = make_dummy_images(cfg.camera_order)
    image_sources = {cam: "dummy" for cam in cfg.camera_order}
    for cam, p in provided.items():
        if p is None or cam not in images:
            continue
        try:
            from terafold.vision.imageio import imread

            images[cam] = imread(p)
            image_sources[cam] = str(p)
        except Exception as e:
            return {"status": "bad_image", "message": f"could not read --{cam}-image {p}: {e}"}

    robot_state = make_dummy_robot_state(cfg)

    if mock:
        adapter = MockAdapter(cfg, model_name=model_name, dtype_name=dtype)
    else:
        try:
            adapter = load_model(model_name, dtype, device=device, cfg=cfg,
                                 revision=revision, log=log)
        except ImportError as e:
            return {"status": "deps_missing", "instructions": str(e).splitlines()}
        except Exception as e:
            # dtype was validated above, so any error here (incl. transformers'
            # ValueError for config/remote-code mismatches) is a load failure.
            return {"status": "model_load_failed",
                    "message": f"{type(e).__name__}: {e}",
                    "hint": f"see {MODEL_CARD_URL.format(model=model_name)}"}

    try:
        pred = adapter.predict(images, robot_state=robot_state, task=task,
                               max_new_tokens=max_new_tokens)
    except AdapterError as e:
        return {"status": "inference_failed", "message": str(e)}

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    actions_doc = {
        "task": task,
        "action": pred["action"],
        "raw_text": pred.get("raw_text"),
        "robot_state": robot_state,
        "image_sources": image_sources,
        "hardware_commanded": False,
    }
    out_path.write_text(json.dumps(actions_doc, indent=2))

    metadata = {
        "model": model_name,
        "revision": revision,
        "dtype": dtype,
        "camera_order": cfg.camera_order,
        "norm_tag": cfg.norm_tag,
        "action_shape": _action_shape(pred["action"]),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hardware_commanded": False,
        "adapter": adapter.describe(),
        "task": task,
        "mock": bool(mock),
    }
    meta_path = out_path.with_name("metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {"status": "ok", "actions_path": str(out_path), "metadata_path": str(meta_path),
            "action_shape": metadata["action_shape"], "mock": bool(mock),
            "hardware_commanded": False}
