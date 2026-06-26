"""Mock camera that renders synthetic towel scenes (numpy only).

The :class:`MockCamera` is the workhorse of the Stage-0 dry-run pipeline: it
produces realistic ``(H, W, 3)`` uint8 RGB frames via
:mod:`terafold.vision.synthetic_cloth` and exposes the *true* image-frame
:class:`~terafold.physics.cloth_state.FoldState` it rendered as
``.last_ground_truth`` so planners / scorers can be evaluated without hardware.

The :func:`build_camera` factory selects between the mock camera and the
OpenCV-backed camera purely from a :class:`CameraConfig`.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from terafold.camera.base import BaseCamera, CameraFrame
from terafold.config.schema import CameraConfig
from terafold.physics.cloth_state import ClothMaskState, FoldState

__all__ = ["MockCamera", "build_camera"]


class MockCamera(BaseCamera):
    """A synthetic camera that renders random towel scenes on every read.

    Parameters
    ----------
    config:
        Optional :class:`CameraConfig`. Controls flips and whether a cloth scene
        is rendered (``mock_render_cloth``). When ``None`` a default config is
        used.
    seed:
        Seed for the internal RNG (reproducible scene sequences).
    image_size:
        Side length (px) of the square rendered frame.
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        seed: int = 0,
        image_size: int = 480,
    ) -> None:
        super().__init__()
        self.config = config or CameraConfig(type="mock")
        self.image_size = int(image_size)
        self._rng = np.random.default_rng(seed)
        self._t0 = time.monotonic()
        self.last_ground_truth: Optional[FoldState] = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        self._open = True
        self._frame_index = 0

    def disconnect(self) -> None:
        self._open = False

    # -- capture -----------------------------------------------------------
    def read(self) -> CameraFrame:
        """Render and return the next synthetic frame as RGB uint8.

        Also stores the ground-truth image-frame :class:`FoldState` in
        ``self.last_ground_truth``.
        """
        if not self._open:
            raise RuntimeError("MockCamera.read() called before connect().")

        # Lazy import so the module stays light and import-safe.
        from terafold.vision.synthetic_cloth import (
            SyntheticClothConfig,
            render_towel_scene,
        )

        marker_mode = False
        if self.config is not None and not self.config.mock_render_cloth:
            # Still render a scene (mask gives a usable cloth), but without
            # painted markers. mock_render_cloth toggles marker painting off.
            marker_mode = False

        cfg = SyntheticClothConfig(image_size=self.image_size, marker_mode=marker_mode)
        sample = render_towel_scene(self._rng, cfg, marker_mode=marker_mode)

        fh = bool(self.config is not None and self.config.flip_horizontal)
        fv = bool(self.config is not None and self.config.flip_vertical)

        image = sample.image
        if fh:
            image = np.ascontiguousarray(image[:, ::-1, :])
        if fv:
            image = np.ascontiguousarray(image[::-1, :, :])

        # The ground truth MUST track the image: flip the labelled geometry the
        # same way, otherwise .last_ground_truth points at the un-flipped cloth.
        self.last_ground_truth = self._fold_state_from_sample(sample, fh, fv)

        frame = CameraFrame(
            image=image,
            timestamp=time.monotonic() - self._t0,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return frame

    def resolution(self) -> Tuple[int, int]:
        return (self.image_size, self.image_size)

    # -- helpers -----------------------------------------------------------
    def _flip_xy(self, arr, fh: bool, fv: bool):
        """Flip pixel coordinate(s) to match an image flip."""
        a = np.array(arr, dtype=np.float64)
        s = self.image_size - 1
        if a.ndim == 1:
            if fh:
                a[0] = s - a[0]
            if fv:
                a[1] = s - a[1]
        else:
            if fh:
                a[:, 0] = s - a[:, 0]
            if fv:
                a[:, 1] = s - a[:, 1]
        return a

    def _fold_state_from_sample(self, sample, fh: bool = False, fv: bool = False) -> FoldState:
        """Build the true image-frame FoldState, flipped to match the image."""
        from terafold.physics.cloth_state import ClothKeypoints, FoldLine

        keypoints = sample.keypoints
        polygon = sample.polygon
        mask_u8 = sample.mask
        fold_line = sample.fold_line

        if fh or fv:
            # Keypoints (4 corners + grasp/place/fold endpoints, NaN for missing).
            kp_arr = keypoints.to_array()
            finite = np.all(np.isfinite(kp_arr), axis=1)
            kp_arr[finite] = self._flip_xy(kp_arr[finite], fh, fv)
            keypoints = ClothKeypoints.from_array(kp_arr)
            if polygon is not None:
                polygon = self._flip_xy(polygon, fh, fv)
            if mask_u8 is not None:
                if fh:
                    mask_u8 = mask_u8[:, ::-1]
                if fv:
                    mask_u8 = mask_u8[::-1, :]
                mask_u8 = np.ascontiguousarray(mask_u8)
            if fold_line is not None:
                d = np.array(fold_line.direction, dtype=np.float64)
                if fh:
                    d[0] = -d[0]
                if fv:
                    d[1] = -d[1]
                fold_line = FoldLine(
                    point=self._flip_xy(fold_line.point, fh, fv),
                    direction=d,
                    a=None if fold_line.a is None else self._flip_xy(fold_line.a, fh, fv),
                    b=None if fold_line.b is None else self._flip_xy(fold_line.b, fh, fv),
                )

        mask_state = ClothMaskState(
            polygon=polygon,
            mask_shape=(self.image_size, self.image_size),
            visible_area_px=float((mask_u8 > 0).sum()),
            mask=mask_u8,
        )
        return FoldState(
            keypoints=keypoints,
            fold_line=fold_line,
            mask=mask_state,
            frame="image",
            image_shape=(self.image_size, self.image_size),
            metadata=dict(sample.meta),
        )


def build_camera(config: CameraConfig) -> BaseCamera:
    """Build a camera from a :class:`CameraConfig`.

    ``type='mock'`` -> :class:`MockCamera`; ``type in {'opencv','webcam'}`` ->
    :class:`~terafold.camera.opencv_camera.OpenCVCamera`.
    """
    if config.type == "mock":
        return MockCamera(config=config, image_size=max(config.width, config.height))
    if config.type in ("opencv", "webcam"):
        # Lazy import: OpenCVCamera lazy-imports cv2 only when used.
        from terafold.camera.opencv_camera import OpenCVCamera

        return OpenCVCamera(config)
    raise ValueError(f"Unknown camera type: {config.type!r}")
