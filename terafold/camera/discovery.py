"""Camera discovery — scan OpenCV camera indices and report which are usable.

Works with whatever a phone-as-webcam app, macOS Continuity Camera, or a plain
USB webcam exposes as an OpenCV ``VideoCapture`` index. Each index is probed in
an ISOLATED subprocess so that a denied-permission abort (macOS TCC raises
``abort()`` inside AVFoundation, which Python cannot catch) takes down only that
probe — never the whole scan. If OpenCV is missing it degrades gracefully.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["CameraScan", "scan_cameras", "_probe_index"]

CV2_HINT = "OpenCV is required to scan cameras. Install it with:  pip install opencv-python"


@dataclass
class CameraScan:
    index: int
    opened: bool
    width: Optional[int] = None
    height: Optional[int] = None
    brightness: Optional[float] = None  # mean grayscale 0-255
    blur: Optional[float] = None  # variance-of-Laplacian (higher = sharper)
    usable: bool = False
    snapshot_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _gray(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float64)
    if a.ndim == 2:
        return a
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def _blur_metric(rgb: np.ndarray) -> float:
    g = _gray(rgb)
    try:
        from scipy import ndimage

        lap = ndimage.laplace(g)
    except Exception:
        lap = (
            -4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
            + np.roll(g, 1, 1) + np.roll(g, -1, 1)
        )
    return float(lap.var())


def _probe_index(index: int, out: Optional[str], warmup: int = 3, min_blur: float = 25.0) -> dict:
    """Open one camera index, capture, compute stats, save a snapshot. (cv2 work.)

    Designed to be called inside an isolated subprocess. Returns a plain dict.
    """
    import cv2

    scan = {"index": int(index), "opened": False, "usable": False}
    cap = cv2.VideoCapture(int(index))
    try:
        if not cap or not cap.isOpened():
            scan["error"] = "could not open"
            return scan
        scan["opened"] = True
        frame = None
        for _ in range(max(1, warmup)):
            ok, bgr = cap.read()
            if ok and bgr is not None:
                frame = bgr
        if frame is None:
            scan["error"] = "opened but no frame"
            return scan
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scan["height"], scan["width"] = int(rgb.shape[0]), int(rgb.shape[1])
        scan["brightness"] = float(_gray(rgb).mean())
        scan["blur"] = _blur_metric(rgb)
        scan["usable"] = bool(10.0 <= scan["brightness"] <= 248.0 and scan["blur"] >= min_blur)
        if out:
            from terafold.vision.imageio import imwrite

            os.makedirs(out, exist_ok=True)
            path = os.path.join(out, f"camera_{int(index)}.png")
            imwrite(path, rgb)
            scan["snapshot_path"] = path
    finally:
        try:
            cap.release()
        except Exception:
            pass
    return scan


_PROBE_CODE = (
    "import json,sys; from terafold.camera.discovery import _probe_index; "
    "print(json.dumps(_probe_index(int(sys.argv[1]), sys.argv[2] or None, "
    "int(sys.argv[3]), float(sys.argv[4]))))"
)


def _probe_isolated(index: int, out: Optional[str], warmup: int, min_blur: float, timeout: float) -> CameraScan:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE, str(index), out or "", str(warmup), str(min_blur)],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return CameraScan(**json.loads(proc.stdout.strip().splitlines()[-1]))
        # Non-zero / abort (e.g. denied camera permission) -> this index is unusable.
        reason = (proc.stderr or "").strip().splitlines()[-1:] or ["probe failed"]
        return CameraScan(index=index, opened=False, error=reason[0][:160])
    except subprocess.TimeoutExpired:
        return CameraScan(index=index, opened=False, error="probe timed out")
    except Exception as exc:
        return CameraScan(index=index, opened=False, error=str(exc))


def scan_cameras(
    max_index: int = 8,
    out: Optional[str] = None,
    preview_seconds: float = 0.0,
    warmup_frames: int = 3,
    min_blur: float = 25.0,
    isolate: bool = True,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Scan camera indices ``0..max_index`` and return a structured report."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return {"available": False, "cv2": False, "hint": CV2_HINT, "cameras": [], "out": out}

    if out:
        os.makedirs(out, exist_ok=True)

    cameras: List[CameraScan] = []
    for idx in range(0, int(max_index) + 1):
        if isolate:
            scan = _probe_isolated(idx, out, warmup_frames, min_blur, timeout)
        else:  # pragma: no cover - direct path used only when caller opts out
            try:
                scan = CameraScan(**_probe_index(idx, out, warmup_frames, min_blur))
            except Exception as exc:
                scan = CameraScan(index=idx, opened=False, error=str(exc))
        cameras.append(scan)
        if preview_seconds and preview_seconds > 0 and scan.usable:
            _preview(idx, float(preview_seconds))

    return {
        "available": True, "cv2": True,
        "cameras": [c.to_dict() for c in cameras],
        "usable_indices": [c.index for c in cameras if c.usable],
        "out": out,
    }


def _preview(index: int, seconds: float) -> None:
    """Briefly show a live feed (best-effort; silently skipped when headless)."""
    try:
        import cv2

        cap = cv2.VideoCapture(int(index))
        window = f"terafold camera {index} (preview)"
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            cv2.imshow(window, bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyWindow(window)
    except Exception:
        pass
