"""Image extraction from HF rows: PIL / bytes / path / dict / missing / video.

These exercise the image-feature path of the sampler with FAKE rows (no network),
covering every value shape a LeRobot/HF dataset can hand us.
"""

from __future__ import annotations

import numpy as np
import pytest

from terafold.data.hf_streaming import (
    _coerce_image,
    _extract_images,
    _pick_top_image,
    _to_image_array,
)
from terafold.vision.imageio import _png_encode


def _rgb(h=12, w=16):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_numpy_hwc():
    img = _rgb()
    arr, info = _coerce_image(img)
    assert arr is not None and arr.shape == (12, 16, 3)
    assert info["kind"] == "ndarray"


def test_numpy_chw_float():
    chw = (np.random.default_rng(1).random((3, 10, 20))).astype("float32")  # CHW, [0,1]
    arr = _to_image_array(chw)
    assert arr is not None and arr.shape == (10, 20, 3) and arr.dtype == np.uint8


def test_pil_image():
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    pil = Image.fromarray(_rgb())
    arr, info = _coerce_image(pil)
    assert arr is not None and arr.shape == (12, 16, 3)
    assert info["kind"] == "pil"


def test_bytes_image():
    png = _png_encode(_rgb())
    arr, info = _coerce_image(png)
    assert arr is not None and arr.shape[2] == 3
    assert info["kind"] == "bytes"


def test_dict_bytes_image():
    png = _png_encode(_rgb())
    arr, info = _coerce_image({"bytes": png, "path": None})
    assert arr is not None
    assert info["kind"] == "image_dict_bytes"


def test_dict_path_image(tmp_path):
    from terafold.vision.imageio import imwrite

    p = tmp_path / "f.png"
    imwrite(str(p), _rgb())
    arr, info = _coerce_image({"bytes": None, "path": str(p)})
    assert arr is not None
    assert info["kind"] == "image_dict_path"


def test_bare_path_string(tmp_path):
    from terafold.vision.imageio import imwrite

    p = tmp_path / "g.png"
    imwrite(str(p), _rgb())
    arr, info = _coerce_image(str(p))
    assert arr is not None and info["kind"] == "path"


def test_video_reference_is_flagged_not_image():
    arr, info = _coerce_image({"path": "videos/chunk-000/observation.images.top/episode_000000.mp4",
                               "timestamp": 1.23})
    assert arr is None
    assert info["kind"] == "video_reference"
    assert "video" in info["reason"].lower()


def test_missing_image_field():
    row = {"observation.state": [0.1, 0.2], "action": [0.0], "timestamp": 1.0}
    assert _extract_images(row) == {}


def test_extract_and_prefer_top():
    row = {
        "observation.images.base": _rgb(8, 8),
        "observation.images.top": _rgb(9, 9),
        "observation.images.endeffector": _rgb(7, 7),
        "observation.state": [0.0] * 6,
    }
    imgs = _extract_images(row)
    assert set(imgs) == {
        "observation.images.base",
        "observation.images.top",
        "observation.images.endeffector",
    }
    top = _pick_top_image(imgs)
    assert top.shape == (9, 9, 3)  # picked observation.images.top
