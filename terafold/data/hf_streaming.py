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

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from terafold.data.sources import get_data_source

__all__ = [
    "INSTALL_HINT",
    "SAFE_SAMPLE_CAP",
    "SAFE_EPISODE_CAP",
    "requires_large_download_flag",
    "sample_hf_dataset",
    "cache_hf_subset",
    "import_hf_lerobot",
]

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


def _to_image_array(value: Any) -> Optional[np.ndarray]:
    """Best-effort conversion of an HF example value to an ``(H, W, 3)`` uint8."""
    from terafold.vision.imageio import ensure_uint8_rgb

    # numpy array
    if isinstance(value, np.ndarray) and value.ndim in (2, 3):
        try:
            return ensure_uint8_rgb(value)
        except Exception:
            return None
    # PIL.Image (datasets decodes Image features to PIL; PIL ships with datasets)
    if hasattr(value, "convert") and hasattr(value, "size"):
        try:
            return ensure_uint8_rgb(np.asarray(value.convert("RGB")))
        except Exception:
            return None
    # dict-like {'array': ...} or {'path': ...}
    if isinstance(value, dict):
        if isinstance(value.get("array"), np.ndarray):
            return _to_image_array(value["array"])
        path = value.get("path")
        if path and isinstance(path, str) and os.path.exists(path):
            try:
                from terafold.vision.imageio import imread

                return imread(path)
            except Exception:
                return None
    return None


def _extract_images(example: Dict[str, Any], image_key: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Extract image arrays from an HF example, keyed by field name."""
    out: Dict[str, np.ndarray] = {}
    if image_key is not None:
        arr = _to_image_array(example.get(image_key))
        if arr is not None:
            out[image_key] = arr
        return out
    # Prefer fields that look like images.
    for k, v in example.items():
        kl = k.lower()
        if ".images." in kl or "image" in kl or "observation.image" in kl:
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
    for k in images:
        if "top" in k.lower():
            return images[k]
    for k in images:
        if ".images." in k.lower():
            return images[k]
    return next(iter(images.values()))


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


def sample_hf_dataset(
    repo_id: str,
    max_samples: int = 500,
    out: Optional[str] = None,
    split: str = "train",
    convert: bool = True,
    image_key: Optional[str] = None,
    allow_large_download: bool = False,
) -> Dict[str, Any]:
    """Stream up to ``max_samples`` examples and save their images (perception fmt).

    Streaming-only: bounded by ``max_samples``, so no full download. Returns a
    status dict; never raises for the common missing-dep / offline cases.
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
                "bounded stream, or pass --allow-large-download to permit a full pull."
            ),
        }

    datasets = _import_datasets()
    if datasets is None:
        return _missing_dep_result(repo_id, "sample-hf-dataset")

    try:
        stream = datasets.load_dataset(repo_id, split=split, streaming=True)
    except Exception as exc:
        if _looks_offline(exc):
            return {"status": "offline", "repo_id": repo_id, "error": str(exc), "hint": INSTALL_HINT}
        if not allow_large_download:
            return {
                "status": "requires_allow_large_download",
                "repo_id": repo_id,
                "error": str(exc),
                "message": (
                    f"Streaming failed for {repo_id} ({exc}); a full download may be "
                    "required. Pass --allow-large-download to permit it."
                ),
            }
        try:
            stream = datasets.load_dataset(repo_id, split=f"{split}[:{max_samples}]")
        except Exception as exc2:
            return {"status": "error", "repo_id": repo_id, "error": str(exc2)}

    out = out or f"data/public_samples/{repo_id.replace('/', '__')}"
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    if convert:
        os.makedirs(os.path.join(out, "masks"), exist_ok=True)
        os.makedirs(os.path.join(out, "labels"), exist_ok=True)

    from terafold.vision.imageio import imwrite

    predictor = None
    if convert:
        from terafold.vision.infer_keypoints import get_keypoint_predictor

        predictor = get_keypoint_predictor(None)  # classical fallback (numpy)

    items: List[dict] = []
    saved = 0
    image_size = 0
    for example in stream:
        if saved >= max_samples:
            break
        images = _extract_images(example, image_key)
        img = _pick_top_image(images)
        if img is None:
            continue
        image_size = max(image_size, int(img.shape[0]))
        rel_img = f"images/{saved:06d}.png"
        imwrite(os.path.join(out, rel_img), img)
        item = {"image": rel_img}
        if convert and predictor is not None:
            try:
                fs = predictor.predict(img)
                label = {
                    "fold_state": fs.to_dict(),
                    "source": repo_id,
                    "weak_label": True,
                    "index": saved,
                }
                rel_label = f"labels/{saved:06d}.json"
                with open(os.path.join(out, rel_label), "w") as f:
                    json.dump(label, f)
                item["label"] = rel_label
                if fs.mask is not None and fs.mask.mask is not None:
                    rel_mask = f"masks/{saved:06d}.png"
                    imwrite(os.path.join(out, rel_mask), (fs.mask.mask > 0).astype("uint8") * 255)
                    item["mask"] = rel_mask
            except Exception:
                pass  # perception is best-effort on arbitrary real images
        items.append(item)
        saved += 1

    if convert:
        _convert_to_perception(out, items, image_size or 256)

    return {
        "status": "ok",
        "repo_id": repo_id,
        "saved": saved,
        "out": out,
        "converted_to_perception": bool(convert),
        "streaming": True,
        "note": "Bounded stream — no full dataset download.",
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
