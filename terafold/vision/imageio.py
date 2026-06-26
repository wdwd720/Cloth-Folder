"""Dependency-light image I/O.

TeraFold's Stage-0 pipeline must run with numpy only, but it still needs to
read/write images (synthetic frames, episode frames, calibration captures).
This module provides ``imread`` / ``imwrite`` with a three-tier backend:

1. OpenCV (``cv2``) if installed — fastest, handles JPG/PNG/etc.
2. Pillow (``PIL``) if installed.
3. A built-in pure-Python PNG codec (zlib only, stdlib) as a guaranteed
   fallback for 8-bit grayscale / RGB / RGBA PNGs.

So the synthetic generator can emit real ``.png`` files and the keypoint
dataset can load them back with nothing beyond numpy + the standard library.
JPEG read/write requires cv2 or Pillow (the pure fallback is PNG-only); writing
a ``.jpg`` without those backends transparently falls back to writing PNG bytes
with a clear warning, so nothing crashes during dry-runs.
"""

from __future__ import annotations

import struct
import warnings
import zlib
from typing import Optional

import numpy as np

__all__ = ["imread", "imwrite", "imread_gray", "ensure_uint8_rgb", "have_cv2", "have_pil"]

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def have_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def have_pil() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except Exception:
        return False


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce an array to contiguous ``(H, W, 3)`` uint8 RGB."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 + 1e-6 else arr, 0, 255)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError(f"unsupported image shape {arr.shape}")
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------
# Pure-python PNG codec (fallback).
# --------------------------------------------------------------------------


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(
        ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
    )


def _png_encode(img: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB (or ``(H, W)`` gray) array to PNG bytes."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        color_type = 0
        h, w = arr.shape
        raw = arr.astype(np.uint8)
        channels = 1
    else:
        arr = ensure_uint8_rgb(arr)
        color_type = 2
        h, w, channels = arr.shape
        raw = arr

    # Prepend a 0 (None) filter byte to each scanline.
    stride = w * channels
    flat = raw.reshape(h, stride)
    filtered = np.empty((h, stride + 1), dtype=np.uint8)
    filtered[:, 0] = 0
    filtered[:, 1:] = flat
    compressed = zlib.compress(filtered.tobytes(), 6)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        _PNG_SIG
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_decode(data: bytes) -> np.ndarray:
    """Decode 8-bit PNG bytes (gray/RGB/RGBA, non-interlaced) to ``(H, W, 3)`` RGB."""
    if data[:8] != _PNG_SIG:
        raise ValueError("not a PNG file")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    if bit_depth != 8:
        raise NotImplementedError(f"only 8-bit PNG supported in fallback (got {bit_depth})")
    if interlace != 0:
        raise NotImplementedError("interlaced PNG not supported in fallback")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise NotImplementedError(f"PNG color type {color_type} not supported in fallback")

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    recon = np.zeros((height, stride), dtype=np.uint8)
    bpp = channels
    i = 0
    prev = np.zeros(stride, dtype=np.int16)
    for row in range(height):
        ftype = raw[i]
        i += 1
        cur = np.frombuffer(raw[i : i + stride], dtype=np.uint8).astype(np.int16).copy()
        i += stride
        if ftype == 0:
            out = cur
        elif ftype == 2:  # Up (vectorizable)
            out = (cur + prev) & 0xFF
        else:  # Sub / Average / Paeth: per-byte recurrence
            out = cur
            for x in range(stride):
                a = out[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if ftype == 1:
                    out[x] = (cur[x] + a) & 0xFF
                elif ftype == 3:
                    out[x] = (cur[x] + ((a + b) >> 1)) & 0xFF
                elif ftype == 4:
                    out[x] = (cur[x] + _paeth(int(a), int(b), int(c))) & 0xFF
                else:
                    raise ValueError(f"bad PNG filter type {ftype}")
        recon[row] = out.astype(np.uint8)
        prev = recon[row].astype(np.int16)

    img = recon.reshape(height, width, channels)
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    if channels == 2:  # gray + alpha
        return np.repeat(img[:, :, :1], 3, axis=2)
    return img[:, :, :3]  # RGB or RGBA -> RGB


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def imwrite(path: str, img: np.ndarray) -> None:
    """Write an image to ``path``, choosing the best available backend."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lower = path.lower()

    if have_cv2():
        import cv2

        arr = ensure_uint8_rgb(img)
        bgr = arr[:, :, ::-1]
        if not cv2.imwrite(path, bgr):
            raise IOError(f"cv2 failed to write {path}")
        return

    if have_pil():
        from PIL import Image

        Image.fromarray(ensure_uint8_rgb(img)).save(path)
        return

    # Pure fallback: PNG only.
    png_bytes = _png_encode(img if np.asarray(img).ndim == 2 else ensure_uint8_rgb(img))
    if not (lower.endswith(".png")):
        warnings.warn(
            f"No cv2/Pillow available; writing PNG bytes to non-.png path {path!r}. "
            "Install terafold[vision] for JPEG support.",
            stacklevel=2,
        )
    with open(path, "wb") as f:
        f.write(png_bytes)


def imread(path: str) -> np.ndarray:
    """Read an image from ``path`` as ``(H, W, 3)`` uint8 RGB."""
    if have_cv2():
        import cv2

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cv2 failed to read {path}")
        return np.ascontiguousarray(bgr[:, :, ::-1])

    if have_pil():
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))

    with open(path, "rb") as f:
        data = f.read()
    return _png_decode(data)


def imread_gray(path: str) -> np.ndarray:
    """Read an image as ``(H, W)`` uint8 grayscale."""
    rgb = imread(path).astype(np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)
