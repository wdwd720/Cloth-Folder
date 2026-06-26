"""Streaming-first access to public Hugging Face datasets.

Philosophy: **never download a huge dataset by default.** Everything here either
streams (HF ``IterableDataset``, pulling only what we iterate) or is bounded by
an explicit ``max_samples`` / ``max_episodes`` cap. A genuine *full* download
only happens when the caller passes ``allow_large_download=True``.

If ``datasets`` / ``huggingface_hub`` are not installed, or the machine is
offline, every function fails *gracefully* — returning a status dict with the
exact install command — instead of raising. The module imports with numpy only.

Sampled images are converted into TeraFold's perception dataset layout (the same
``images/`` + ``masks/`` + ``labels/`` + ``index.json`` produced by the
synthetic generator) using the classical fallback predictor, so they can feed
keypoint/segmentation pretraining. Such labels are *weak* (auto-generated) and
flagged as ``weak_label: true``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from terafold.data.sources import get_data_source

__all__ = [
    "INSTALL_HINT",
    "VIDEO_INSTALL_HINT",
    "SAFE_SAMPLE_CAP",
    "SAFE_EPISODE_CAP",
    "requires_large_download_flag",
    "sample_hf_dataset",
    "cache_hf_subset",
    "import_hf_lerobot",
    "debug_hf_row",
    "VideoDecodeUnavailable",
]

VIDEO_INSTALL_HINT = (
    "Native LeRobot video decoding is needed. Install a backend:  pip install av   "
    "(or: pip install imageio imageio-ffmpeg ;  or: pip install -e \".[lerobot]\")"
)
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


class VideoDecodeUnavailable(RuntimeError):
    """Raised when an mp4 must be decoded but no video backend is installed."""

INSTALL_HINT = (
    "Install dataset tooling with:  pip install -e \".[lerobot]\"   "
    "(or: pip install datasets huggingface_hub pillow)"
)

# Bounded streaming below these caps is always considered safe (no flag needed).
SAFE_SAMPLE_CAP = 5000
SAFE_EPISODE_CAP = 100
_MAX_ROWS_PER_EPISODE = 5000  # safety bound when an episode index is absent/odd


# --------------------------------------------------------------------------
# Dependency / safety guards
# --------------------------------------------------------------------------


def _import_datasets():
    """Return the ``datasets`` module, or ``None`` if unavailable."""
    try:
        import datasets  # type: ignore

        return datasets
    except Exception:
        return None


def _missing_dep_result(repo_id: str, op: str) -> Dict[str, Any]:
    return {
        "status": "missing_dependency",
        "repo_id": repo_id,
        "op": op,
        "install_command": "pip install -e \".[lerobot]\"",
        "message": (
            f"{op} needs the Hugging Face `datasets` library, which is not "
            f"installed. {INSTALL_HINT}"
        ),
    }


def _looks_offline(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        k in text
        for k in ("connection", "offline", "timeout", "resolve", "network", "getaddrinfo")
    )


def requires_large_download_flag(
    repo_id: str, *, bounded: bool, allow_large_download: bool
) -> bool:
    """Decide whether an operation needs the explicit ``--allow-large-download``.

    * If the caller already authorized it -> never required.
    * A *bounded* request (finite max_samples / max_episodes) streams only what
      it needs -> safe, never required.
    * An *unbounded* request needs the flag when the source is registry-flagged
      ``large`` — or when the source is unknown (we default to caution).
    """
    if allow_large_download:
        return False
    if bounded:
        return False
    source = get_data_source(repo_id)
    return True if source is None else bool(source.large)


def _is_bounded(n: Optional[int]) -> bool:
    """A positive finite cap is 'bounded'; None / <=0 means 'all'."""
    return n is not None and n > 0


# --------------------------------------------------------------------------
# Image extraction + perception conversion
# --------------------------------------------------------------------------


def _from_numpy_imagelike(arr: Any) -> Optional[np.ndarray]:
    """Normalize an array (HWC/CHW, uint8/float, with optional leading frame dim)."""
    from terafold.vision.imageio import ensure_uint8_rgb

    a = np.asarray(arr)
    if a.ndim == 4:  # (T, H, W, C) or (T, C, H, W) -> first frame
        return _from_numpy_imagelike(a[0])
    if a.ndim == 2:
        return ensure_uint8_rgb(a)
    if a.ndim == 3:
        # CHW -> HWC when the first axis is the channel axis.
        if a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
            a = np.transpose(a, (1, 2, 0))
        if a.dtype.kind == "f":
            a = a * 255.0 if float(np.nanmax(a) if a.size else 0) <= 1.0 + 1e-6 else a
            a = np.clip(a, 0, 255).astype(np.uint8)
        try:
            return ensure_uint8_rgb(a)
        except Exception:
            return None
    return None


def _decode_image_bytes(b: bytes) -> Optional[np.ndarray]:
    """Decode PNG/JPEG image bytes via Pillow -> cv2 -> pure PNG fallback."""
    from terafold.vision.imageio import ensure_uint8_rgb

    try:
        from PIL import Image

        return ensure_uint8_rgb(np.asarray(Image.open(io.BytesIO(b)).convert("RGB")))
    except Exception:
        pass
    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            return np.ascontiguousarray(arr[:, :, ::-1])
    except Exception:
        pass
    try:
        from terafold.vision.imageio import _png_decode

        if b[:8] == b"\x89PNG\r\n\x1a\n":
            return _png_decode(b)
    except Exception:
        pass
    return None


def _coerce_image(value: Any) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Convert an HF example value to ``(H,W,3) uint8`` and explain the result.

    Handles: numpy array (HWC/CHW/float/with-frame-dim), PIL.Image, torch
    tensor, raw image bytes, HF Image dict ``{'bytes','path'}``, ``{'array': ...}``,
    a bare path string, and video-reference dicts/paths (flagged as needing
    native video decoding). Returns ``(array_or_None, info)`` where ``info`` has
    ``kind`` and (on failure) ``reason``.
    """
    # torch tensor
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                a = _from_numpy_imagelike(value.detach().cpu().numpy())
                return a, {"kind": "torch_tensor", "reason": None if a is not None else "bad tensor shape"}
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        a = _from_numpy_imagelike(value)
        return a, {"kind": "ndarray", "shape": list(value.shape),
                   "reason": None if a is not None else "unsupported ndarray shape"}
    if hasattr(value, "convert") and hasattr(value, "size"):  # PIL.Image
        try:
            from terafold.vision.imageio import ensure_uint8_rgb

            return ensure_uint8_rgb(np.asarray(value.convert("RGB"))), {"kind": "pil"}
        except Exception as exc:
            return None, {"kind": "pil", "reason": str(exc)}
    if isinstance(value, (bytes, bytearray)):
        a = _decode_image_bytes(bytes(value))
        return a, {"kind": "bytes", "reason": None if a is not None else "undecodable image bytes"}
    if isinstance(value, str):
        if os.path.exists(value):
            ext = os.path.splitext(value)[1].lower()
            if ext in VIDEO_EXTS:
                return None, {"kind": "video_path", "reason": "video file — needs frame decoding", "path": value}
            try:
                from terafold.vision.imageio import imread

                return imread(value), {"kind": "path", "path": value}
            except Exception as exc:
                return None, {"kind": "path", "reason": str(exc), "path": value}
        return None, {"kind": "str", "reason": "string is not an existing file path"}
    if isinstance(value, dict):
        if isinstance(value.get("array"), np.ndarray):
            a, info = _coerce_image(value["array"])
            info["kind"] = "dict_array"
            return a, info
        b = value.get("bytes")
        path = value.get("path")
        if isinstance(b, (bytes, bytearray)) and b:
            a = _decode_image_bytes(bytes(b))
            return a, {"kind": "image_dict_bytes", "reason": None if a is not None else "bytes not a decodable image"}
        if isinstance(path, str) and path:
            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTS:
                return None, {"kind": "video_reference", "path": path,
                              "reason": "video reference (mp4); native LeRobot video decoding needed"}
            if os.path.exists(path):
                try:
                    from terafold.vision.imageio import imread

                    return imread(path), {"kind": "image_dict_path", "path": path}
                except Exception as exc:
                    return None, {"kind": "image_dict_path", "reason": str(exc), "path": path}
            return None, {"kind": "image_dict_path", "path": path,
                          "reason": f"path not present locally: {path}"}
        return None, {"kind": "dict", "reason": f"unrecognized image dict keys: {sorted(value.keys())}"}
    # decord/av VideoReader-like objects
    if hasattr(value, "asnumpy"):
        try:
            return _from_numpy_imagelike(value.asnumpy()), {"kind": "video_object"}
        except Exception as exc:
            return None, {"kind": "video_object", "reason": str(exc)}
    if hasattr(value, "__array__"):
        try:
            return _from_numpy_imagelike(np.asarray(value)), {"kind": "array_like"}
        except Exception as exc:
            return None, {"kind": "array_like", "reason": str(exc)}
    return None, {"kind": type(value).__name__, "reason": "unrecognized image value type"}


def _to_image_array(value: Any) -> Optional[np.ndarray]:
    """Back-compat thin wrapper returning just the array (or None)."""
    return _coerce_image(value)[0]


def _looks_like_image_key(key: str) -> bool:
    kl = key.lower()
    return (
        ".images." in kl
        or "image" in kl
        or kl.endswith((".top", ".base", ".endeffector", ".wrist", ".cam", ".rgb"))
        or "observation.image" in kl
    )


def _extract_images(example: Dict[str, Any], image_key: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Extract image arrays from an HF example, keyed by field name."""
    out: Dict[str, np.ndarray] = {}
    if image_key is not None:
        arr = _to_image_array(example.get(image_key))
        if arr is not None:
            out[image_key] = arr
        return out
    for k, v in example.items():  # prefer fields that look like images
        if _looks_like_image_key(k):
            arr = _to_image_array(v)
            if arr is not None:
                out[k] = arr
    if not out:  # fall back to any image-like value
        for k, v in example.items():
            arr = _to_image_array(v)
            if arr is not None:
                out[k] = arr
    return out


def _pick_top_image(images: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if not images:
        return None
    for k in images:  # requirement: prefer observation.images.top
        if k.lower().endswith(".top") or "images.top" in k.lower() or "top" in k.lower():
            return images[k]
    for k in images:
        if ".images." in k.lower():
            return images[k]
    return next(iter(images.values()))


# --------------------------------------------------------------------------
# LeRobot video support (frames live in separate mp4 files, not the parquet)
# --------------------------------------------------------------------------


def _fetch_lerobot_info(repo_id: str) -> Optional[dict]:
    """Download ``meta/info.json`` for a LeRobot dataset (or ``None``)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    for candidate in ("meta/info.json", "info.json"):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="dataset")
            with open(path) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _video_keys_from_info(info: dict) -> List[str]:
    feats = (info or {}).get("features", {}) or {}
    return [
        k for k, v in feats.items()
        if (v or {}).get("dtype") in ("video", "image") or ".images." in k
    ]


def _pick_camera_key(keys: List[str], prefer: str = "observation.images.top") -> Optional[str]:
    if not keys:
        return None
    if prefer in keys:
        return prefer
    for k in keys:
        if k.lower().endswith(".top") or "images.top" in k.lower():
            return k
    return keys[0]


def _video_backend() -> Optional[str]:
    """Name of the first available video backend, or ``None``."""
    if importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"):
        return "imageio"
    if importlib.util.find_spec("av"):
        return "av"
    if importlib.util.find_spec("decord"):
        return "decord"
    return None


def _iter_video_frames(path: str, backend: Optional[str] = None):
    """Yield decoded RGB ``(H,W,3) uint8`` frames from a video file."""
    backend = backend or _video_backend()
    if backend is None:
        raise VideoDecodeUnavailable(VIDEO_INSTALL_HINT)
    if backend == "imageio":
        import imageio.v3 as iio

        for frame in iio.imiter(path, plugin="FFMPEG"):
            yield np.asarray(frame)
    elif backend == "av":
        import av

        with av.open(path) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
    elif backend == "decord":
        import decord

        vr = decord.VideoReader(path)
        for i in range(len(vr)):
            yield vr[i].asnumpy()
    else:  # pragma: no cover
        raise VideoDecodeUnavailable(VIDEO_INSTALL_HINT)


def _download_episode_video(repo_id: str, info: dict, episode_index: int, video_key: str) -> str:
    """Download one episode's mp4 for ``video_key`` and return the local path."""
    from huggingface_hub import hf_hub_download

    chunks_size = int(info.get("chunks_size", 1000) or 1000)
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    rel = template.format(
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )
    return hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset")


def _convert_to_perception(out_dir: str, index_items: List[dict], image_size: int) -> None:
    """Write an index.json so the sampled images load as a perception dataset."""
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(
            {
                "num": len(index_items),
                "image_size": image_size,
                "marker_mode": False,
                "items": index_items,
                "meta": {
                    "generator": "terafold.data.hf_streaming",
                    "weak_label": True,
                    "note": "Auto labels from the classical fallback predictor; verify before training.",
                },
            },
            f,
            indent=2,
        )


# --------------------------------------------------------------------------
# Public operations
# --------------------------------------------------------------------------


def _setup_sample_dirs(out: str, convert: bool) -> None:
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    if convert:
        os.makedirs(os.path.join(out, "masks"), exist_ok=True)
        os.makedirs(os.path.join(out, "labels"), exist_ok=True)


def _write_perception_sample(out, idx, img, repo_id, predictor, convert, items) -> int:
    """Save one image (+ optional weak perception label) and append to ``items``."""
    from terafold.vision.imageio import imwrite

    rel_img = f"images/{idx:06d}.png"
    imwrite(os.path.join(out, rel_img), img)
    item = {"image": rel_img}
    if convert and predictor is not None:
        try:
            fs = predictor.predict(img)
            rel_label = f"labels/{idx:06d}.json"
            with open(os.path.join(out, rel_label), "w") as f:
                json.dump(
                    {"fold_state": fs.to_dict(), "source": repo_id, "weak_label": True, "index": idx},
                    f,
                )
            item["label"] = rel_label
            if fs.mask is not None and fs.mask.mask is not None:
                rel_mask = f"masks/{idx:06d}.png"
                imwrite(os.path.join(out, rel_mask), (fs.mask.mask > 0).astype("uint8") * 255)
                item["mask"] = rel_mask
        except Exception:
            pass  # perception is best-effort on arbitrary real images
    items.append(item)
    return int(img.shape[0])


def _sample_lerobot_videos(
    repo_id, info, max_samples, out, convert, frames_per_episode, frame_stride, camera_key=None
) -> Optional[Dict[str, Any]]:
    """Sample frames from a LeRobot video dataset (frames live in episode mp4s)."""
    video_keys = _video_keys_from_info(info)
    if not video_keys:
        return None
    camera = camera_key if (camera_key in video_keys) else _pick_camera_key(video_keys)
    backend = _video_backend()
    if backend is None:
        return {
            "status": "needs_video_decoder",
            "repo_id": repo_id,
            "camera": camera,
            "source_format": "lerobot_video",
            "available_cameras": video_keys,
            "message": (
                f"{repo_id} stores camera frames as MP4 video (e.g. {camera}); the "
                f"streamed parquet rows contain no pixels. {VIDEO_INSTALL_HINT}"
            ),
            "install_command": "pip install av",
        }

    total_eps = int(info.get("total_episodes") or 0) or 1
    cam_info = ((info.get("features", {}) or {}).get(camera, {}) or {}).get("info", {}) or {}
    fps = int(cam_info.get("video.fps", info.get("fps", 30)) or 30)
    if not frame_stride or frame_stride <= 0:
        frame_stride = max(1, fps)  # ~1 frame/sec for temporal diversity
    if not frames_per_episode or frames_per_episode <= 0:
        frames_per_episode = max(1, math.ceil(max_samples / min(total_eps, 8)))

    _setup_sample_dirs(out, convert)
    predictor = None
    if convert:
        from terafold.vision.infer_keypoints import get_keypoint_predictor

        predictor = get_keypoint_predictor(None)

    items: List[dict] = []
    saved = 0
    image_size = 0
    episodes_used = 0
    episodes_downloaded = 0
    ep = 0
    while saved < max_samples and ep < total_eps:
        try:
            mp4 = _download_episode_video(repo_id, info, ep, camera)
            episodes_downloaded += 1
        except Exception:
            ep += 1
            continue
        take = min(max_samples - saved, frames_per_episode)
        got = 0
        try:
            for i, frame in enumerate(_iter_video_frames(mp4, backend)):
                if i % frame_stride:
                    continue
                img = _from_numpy_imagelike(frame)
                if img is None:
                    continue
                image_size = max(
                    image_size,
                    _write_perception_sample(out, saved, img, repo_id, predictor, convert, items),
                )
                saved += 1
                got += 1
                if got >= take or saved >= max_samples:
                    break
        except VideoDecodeUnavailable:
            return {
                "status": "needs_video_decoder",
                "repo_id": repo_id,
                "camera": camera,
                "message": VIDEO_INSTALL_HINT,
                "install_command": "pip install av",
            }
        if got:
            episodes_used += 1
        ep += 1

    if convert:
        _convert_to_perception(out, items, image_size or 256)
    return {
        "status": "ok",
        "repo_id": repo_id,
        "saved": saved,
        "out": out,
        "camera": camera,
        "source_format": "lerobot_video",
        "backend": backend,
        "episodes_used": episodes_used,
        "episodes_downloaded": episodes_downloaded,
        "converted_to_perception": bool(convert),
        "note": (
            "Frames decoded from a BOUNDED set of episode MP4s (LeRobot stores "
            "pixels in mp4, not parquet). Bounded download, not pure streaming."
        ),
    }


def sample_hf_dataset(
    repo_id: str,
    max_samples: int = 500,
    out: Optional[str] = None,
    split: str = "train",
    convert: bool = True,
    image_key: Optional[str] = None,
    allow_large_download: bool = False,
    frames_per_episode: int = 0,
    frame_stride: int = 0,
) -> Dict[str, Any]:
    """Sample up to ``max_samples`` images from a public dataset (perception fmt).

    Two paths, chosen by inspecting the real row schema:

    * **Image-in-parquet** datasets: stream rows and save the image field
      (preferring ``observation.images.top``). Bounded streaming, no download.
    * **LeRobot video** datasets (frames stored as mp4, e.g.
      ``observabot/so101_cloth_folding1``): the parquet has no pixels, so we
      download a BOUNDED set of episode mp4s and decode frames (needs a video
      backend; if absent, returns ``needs_video_decoder`` with install guidance).

    Returns a status dict; never raises for the common missing-dep/offline cases.
    """
    bounded = _is_bounded(max_samples)
    if not bounded and requires_large_download_flag(
        repo_id, bounded=False, allow_large_download=allow_large_download
    ):
        return {
            "status": "requires_allow_large_download",
            "repo_id": repo_id,
            "message": (
                f"{repo_id} is large / unbounded. Re-run with --max-samples N for a "
                "bounded sample, or pass --allow-large-download to permit a full pull."
            ),
        }

    datasets = _import_datasets()
    if datasets is None:
        return _missing_dep_result(repo_id, "sample-hf-dataset")

    out = out or f"data/public_samples/{repo_id.replace('/', '__')}"

    # Peek the real row schema to decide image-in-parquet vs LeRobot video.
    stream = None
    first = None
    try:
        stream = datasets.load_dataset(repo_id, split=split, streaming=True)
        first = next(iter(stream))
    except StopIteration:
        first = None
    except Exception as exc:
        if _looks_offline(exc):
            return {"status": "offline", "repo_id": repo_id, "error": str(exc), "hint": INSTALL_HINT}
        # Streaming unsupported: try the LeRobot video path via info.json.
        info = _fetch_lerobot_info(repo_id)
        if info and _video_keys_from_info(info):
            return _sample_lerobot_videos(
                repo_id, info, max_samples, out, convert, frames_per_episode, frame_stride, image_key
            )
        if not allow_large_download:
            return {
                "status": "requires_allow_large_download",
                "repo_id": repo_id,
                "error": str(exc),
                "message": f"Streaming failed for {repo_id} ({exc}); pass --allow-large-download.",
            }
        return {"status": "error", "repo_id": repo_id, "error": str(exc)}

    has_row_images = bool(first and _extract_images(first, image_key))

    if has_row_images:
        # PATH A: images are in the parquet rows — stream them.
        _setup_sample_dirs(out, convert)
        predictor = None
        if convert:
            from terafold.vision.infer_keypoints import get_keypoint_predictor

            predictor = get_keypoint_predictor(None)
        items: List[dict] = []
        saved = 0
        image_size = 0
        for example in stream:  # IterableDataset re-iterates from the start
            if saved >= max_samples:
                break
            img = _pick_top_image(_extract_images(example, image_key))
            if img is None:
                continue
            image_size = max(
                image_size,
                _write_perception_sample(out, saved, img, repo_id, predictor, convert, items),
            )
            saved += 1
        if convert:
            _convert_to_perception(out, items, image_size or 256)
        return {
            "status": "ok",
            "repo_id": repo_id,
            "saved": saved,
            "out": out,
            "source_format": "parquet_image",
            "converted_to_perception": bool(convert),
            "streaming": True,
            "note": "Bounded stream — no full dataset download.",
        }

    # PATH B: no image fields in the parquet — try the LeRobot video layout.
    info = _fetch_lerobot_info(repo_id)
    if info and _video_keys_from_info(info):
        return _sample_lerobot_videos(
            repo_id, info, max_samples, out, convert, frames_per_episode, frame_stride, image_key
        )

    return {
        "status": "no_images_found",
        "repo_id": repo_id,
        "row_keys": list(first.keys()) if first else [],
        "message": (
            "No image fields in the streamed rows and no video features in "
            "info.json. Run `terafold debug-hf-row --repo-id "
            f"{repo_id} --streaming` to inspect the schema."
        ),
    }


def _stream_episode_rows(
    stream, max_episodes: int, image_dir: Optional[str]
):
    """Yield ``(episode_index, rows)`` grouping a streamed dataset by episode.

    Falls back to a single pseudo-episode (capped) when no ``episode_index``
    field is present. Saves any images encountered under ``image_dir``.
    """
    from terafold.vision.imageio import imwrite

    current_ep: Optional[int] = None
    rows: List[dict] = []
    episodes_done = 0
    frame_in_ep = 0
    total_rows = 0

    for example in stream:
        ep_idx = example.get("episode_index")
        if ep_idx is None:
            ep_idx = 0  # pseudo-episode
        if current_ep is None:
            current_ep = ep_idx
            frame_in_ep = 0
        if ep_idx != current_ep:
            yield current_ep, rows
            episodes_done += 1
            if episodes_done >= max_episodes:
                return
            current_ep = ep_idx
            rows = []
            frame_in_ep = 0

        row = {k: v for k, v in example.items() if _json_safe(v)}
        if image_dir is not None:
            images = _extract_images(example)
            top = _pick_top_image(images)
            if top is not None:
                ep_dir = os.path.join(image_dir, f"episode_{int(current_ep):06d}")
                os.makedirs(ep_dir, exist_ok=True)
                rel = os.path.join(f"episode_{int(current_ep):06d}", f"top_{frame_in_ep:06d}.png")
                imwrite(os.path.join(image_dir, rel), top)
                row["observation.images.top"] = rel
        rows.append(row)
        frame_in_ep += 1
        total_rows += 1
        if frame_in_ep > _MAX_ROWS_PER_EPISODE:  # pathological / no episode index
            yield current_ep, rows
            episodes_done += 1
            if episodes_done >= max_episodes:
                return
            current_ep = None
            rows = []

    if rows and episodes_done < max_episodes:
        yield current_ep if current_ep is not None else 0, rows


def _json_safe(v: Any) -> bool:
    """Keep only JSON-serializable scalar/list fields (drop PIL/bytes/arrays)."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return True
    if isinstance(v, (list, tuple)):
        try:
            json.dumps(list(v))
            return True
        except Exception:
            return False
    return False


def cache_hf_subset(
    repo_id: str,
    max_episodes: int = 20,
    out: Optional[str] = None,
    split: str = "train",
    allow_large_download: bool = False,
) -> Dict[str, Any]:
    """Cache a *bounded* subset (first ``max_episodes`` episodes) locally.

    Streams and stops after ``max_episodes`` — no full download. Writes raw rows
    to ``<out>/data/episode_NNNNNN.jsonl`` and any frames to ``<out>/images/``.
    """
    bounded = _is_bounded(max_episodes)
    if requires_large_download_flag(repo_id, bounded=bounded, allow_large_download=allow_large_download):
        return {
            "status": "requires_allow_large_download",
            "repo_id": repo_id,
            "message": (
                f"{repo_id} is large / unbounded. Set --max-episodes N (bounded) or "
                "pass --allow-large-download."
            ),
        }

    datasets = _import_datasets()
    if datasets is None:
        return _missing_dep_result(repo_id, "cache-hf-subset")

    try:
        stream = datasets.load_dataset(repo_id, split=split, streaming=True)
    except Exception as exc:
        if _looks_offline(exc):
            return {"status": "offline", "repo_id": repo_id, "error": str(exc), "hint": INSTALL_HINT}
        return {
            "status": "requires_allow_large_download" if not allow_large_download else "error",
            "repo_id": repo_id,
            "error": str(exc),
            "message": f"Streaming failed for {repo_id}; pass --allow-large-download to permit a full pull.",
        }

    out = out or f"data/cache/{repo_id.replace('/', '__')}"
    data_dir = os.path.join(out, "data")
    image_dir = os.path.join(out, "images")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    n_eps = 0
    n_rows = 0
    for ep_idx, rows in _stream_episode_rows(stream, max_episodes, image_dir):
        with open(os.path.join(data_dir, f"episode_{n_eps:06d}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_eps += 1
        n_rows += len(rows)
        if n_eps >= max_episodes:
            break

    with open(os.path.join(out, "cache_info.json"), "w") as f:
        json.dump(
            {"repo_id": repo_id, "episodes": n_eps, "rows": n_rows, "streaming": True}, f, indent=2
        )
    return {
        "status": "ok",
        "repo_id": repo_id,
        "episodes": n_eps,
        "rows": n_rows,
        "out": out,
        "note": "Bounded stream — no full dataset download.",
    }


def import_hf_lerobot(
    repo_id: str,
    max_episodes: int = 20,
    out: Optional[str] = None,
    split: str = "train",
    allow_large_download: bool = False,
) -> Dict[str, Any]:
    """Import a bounded subset of a LeRobot dataset into TeraFold's intermediate format.

    Produces ``<out>/meta/info.json`` (from inspection), ``<out>/data/episode_*.jsonl``
    with the canonical fields (``observation.images.top``, ``observation.state``,
    ``action``, ``timestamp``, ``frame_index``, ``episode_index``, ``task``), and
    frames under ``<out>/images/``. Bounded by ``max_episodes`` (no full download).

    Foreign actions are imported for analysis/representation ONLY — they are not
    directly executable on the LeArm.
    """
    bounded = _is_bounded(max_episodes)
    if requires_large_download_flag(repo_id, bounded=bounded, allow_large_download=allow_large_download):
        return {
            "status": "requires_allow_large_download",
            "repo_id": repo_id,
            "message": f"{repo_id} is large / unbounded. Set --max-episodes N or pass --allow-large-download.",
        }

    datasets = _import_datasets()
    if datasets is None:
        return _missing_dep_result(repo_id, "import-hf-lerobot")

    out = out or f"data/public/{repo_id.replace('/', '__')}"
    meta_dir = os.path.join(out, "meta")
    data_dir = os.path.join(out, "data")
    image_dir = os.path.join(out, "images")
    for d in (meta_dir, data_dir, image_dir):
        os.makedirs(d, exist_ok=True)

    # Best-effort metadata (does not require a download).
    try:
        from terafold.data.inspect_hf_dataset import inspect_hf_dataset

        report = inspect_hf_dataset(repo_id)
    except Exception:
        report = {"repo_id": repo_id}
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(
            {
                "source_repo_id": repo_id,
                "codebase_version": "terafold-import",
                "imported_subset_episodes": max_episodes,
                "source_report": report,
                "warning": (
                    "Foreign-embodiment dataset. Imported for perception / "
                    "representation / reference. Do NOT replay actions on the LeArm."
                ),
            },
            f,
            indent=2,
        )

    try:
        stream = datasets.load_dataset(repo_id, split=split, streaming=True)
    except Exception as exc:
        if _looks_offline(exc):
            return {"status": "offline", "repo_id": repo_id, "error": str(exc), "hint": INSTALL_HINT}
        return {
            "status": "requires_allow_large_download" if not allow_large_download else "error",
            "repo_id": repo_id,
            "error": str(exc),
        }

    n_eps = 0
    n_rows = 0
    for ep_idx, rows in _stream_episode_rows(stream, max_episodes, image_dir):
        mapped = []
        for i, r in enumerate(rows):
            mapped.append(
                {
                    "observation.images.top": r.get("observation.images.top"),
                    "observation.state": r.get("observation.state"),
                    "action": r.get("action"),
                    "timestamp": r.get("timestamp", i),
                    "frame_index": i,
                    "episode_index": n_eps,
                    "task": r.get("task", repo_id),
                }
            )
        with open(os.path.join(data_dir, f"episode_{n_eps:06d}.jsonl"), "w") as f:
            for m in mapped:
                f.write(json.dumps(m) + "\n")
        n_eps += 1
        n_rows += len(mapped)
        if n_eps >= max_episodes:
            break

    return {
        "status": "ok",
        "repo_id": repo_id,
        "episodes": n_eps,
        "rows": n_rows,
        "out": out,
        "note": "Bounded import — foreign actions are for analysis only, not LeArm rollout.",
    }


# --------------------------------------------------------------------------
# Debugging: explain a dataset's row schema and why rows are/aren't sampled.
# --------------------------------------------------------------------------


def _describe_value(value: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, dict):
        d["nested_keys"] = list(value.keys())
    elif isinstance(value, (list, tuple)):
        d["len"] = len(value)
    elif hasattr(value, "size") and hasattr(value, "mode"):  # PIL
        d["pil"] = {"size": list(getattr(value, "size", [])), "mode": getattr(value, "mode", None)}
    elif hasattr(value, "shape"):
        d["shape"] = list(getattr(value, "shape"))
        d["dtype"] = str(getattr(value, "dtype", ""))
    return d


def debug_hf_row(
    repo_id: str,
    num_rows: int = 3,
    streaming: bool = True,
    split: str = "train",
) -> Dict[str, Any]:
    """Inspect the actual streamed rows and explain image extraction decisions.

    Reports, per row: keys, value types/nested keys/shapes, image-field
    candidates (with coercion result + reason), and state/action dims — plus
    whether the row is sampleable directly or needs the LeRobot video path.
    """
    datasets = _import_datasets()
    if datasets is None:
        return _missing_dep_result(repo_id, "debug-hf-row")

    info = _fetch_lerobot_info(repo_id)
    info_video_keys = _video_keys_from_info(info) if info else []
    report: Dict[str, Any] = {
        "status": "ok",
        "repo_id": repo_id,
        "info_image_or_video_keys": info_video_keys,
        "video_path_template": (info or {}).get("video_path"),
        "video_backend": _video_backend(),
        "total_episodes": (info or {}).get("total_episodes"),
        "rows": [],
    }

    try:
        stream = datasets.load_dataset(repo_id, split=split, streaming=True)
        it = iter(stream)
    except Exception as exc:
        report["status"] = "offline" if _looks_offline(exc) else "error"
        report["error"] = str(exc)
        return report

    for r_i in range(max(1, num_rows)):
        try:
            row = next(it)
        except StopIteration:
            break
        fields: Dict[str, Any] = {}
        image_candidates: List[str] = []
        for k, v in row.items():
            desc = _describe_value(v)
            name_hit = _looks_like_image_key(k)
            arr, cinfo = _coerce_image(v)
            if name_hit or arr is not None:
                image_candidates.append(k)
                desc["image_candidate"] = True
                desc["coerce_ok"] = arr is not None
                desc["coerce_kind"] = cinfo.get("kind")
                if arr is not None:
                    desc["coerced_shape"] = list(arr.shape)
                elif cinfo.get("reason"):
                    desc["coerce_reason"] = cinfo["reason"]
            fields[k] = desc

        img = _pick_top_image(_extract_images(row))
        state = row.get("observation.state")
        action = row.get("action")
        if img is not None:
            decision = "SAMPLE: a usable image (preferring observation.images.top) is in the row."
        elif info_video_keys:
            decision = (
                "VIDEO PATH: no pixels in the parquet row; frames are stored as MP4 "
                f"({', '.join(info_video_keys)}). sample-hf-dataset downloads bounded "
                "episode mp4s and decodes them"
                + ("." if report["video_backend"] else " — but NO video backend is installed (" + VIDEO_INSTALL_HINT + ").")
            )
        else:
            decision = "SKIP: no image field in the row and no video features in info.json."

        report["rows"].append(
            {
                "index": r_i,
                "keys": list(row.keys()),
                "fields": fields,
                "image_field_candidates": image_candidates,
                "image_usable_in_row": img is not None,
                "state_dim": len(state) if hasattr(state, "__len__") else None,
                "action_dim": len(action) if hasattr(action, "__len__") else None,
                "sampling_decision": decision,
            }
        )
    return report
