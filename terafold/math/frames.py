"""Coordinate frame transformations for TeraFold.

Frames
------
* Image frame ``I``: pixel coordinates ``(u, v)`` (origin top-left, v down).
* Table frame ``T``: metric coordinates ``(x, y, z)``, ``z = 0`` on the table.
* Robot base frame ``B``: the robot's own coordinate frame.
* End-effector frame ``E``: gripper frame (placeholder for future hand-eye).

For a top-down camera, a planar homography ``H_IT`` (image pixels -> table
plane) is sufficient for V0. The table <-> robot mapping ``T_BT`` is an
in-plane rigid/similarity transform (3x3 homogeneous) plus a straight-through
``z`` (the arm's base z aligns with the table normal for V0). When ``T_BT`` is
unknown the identity is used and a warning-worthy flag is exposed.

A :class:`FrameTransforms` bundle carries the calibrated matrices and offers
ergonomic round-trip methods, but all the core functions are also available as
free functions so they can be unit-tested without constructing the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from terafold.math.transforms import (
    apply_affine_2d,
    apply_homography,
    invert_affine_2d,
    invert_homography,
)

ArrayLike = np.ndarray

__all__ = [
    "pixel_to_table_xy",
    "table_xy_to_pixel",
    "table_to_robot",
    "robot_to_table",
    "FrameTransforms",
]


def pixel_to_table_xy(H_IT: ArrayLike, uv: ArrayLike) -> np.ndarray:
    """Map image pixel(s) ``(u, v)`` to table-plane metric ``(x, y)``.

    ``H_IT`` is the image->table homography. Accepts ``(2,)`` or ``(N, 2)``.
    """
    return apply_homography(H_IT, uv)


def table_xy_to_pixel(H_IT: ArrayLike, xy: ArrayLike) -> np.ndarray:
    """Map table-plane metric ``(x, y)`` back to image pixel(s) ``(u, v)``.

    Uses the inverse of ``H_IT`` (i.e. ``H_TI``).
    """
    H_TI = invert_homography(H_IT)
    return apply_homography(H_TI, xy)


def table_to_robot(T_BT: ArrayLike, xy: ArrayLike) -> np.ndarray:
    """Map table-plane ``(x, y)`` to robot base-frame ``(x, y)``.

    ``T_BT`` is a 3x3 homogeneous in-plane transform (rigid or similarity).
    """
    return apply_affine_2d(T_BT, xy)


def robot_to_table(T_BT: ArrayLike, xy: ArrayLike) -> np.ndarray:
    """Map robot base-frame ``(x, y)`` back to table-plane ``(x, y)``."""
    return apply_affine_2d(invert_affine_2d(T_BT), xy)


@dataclass
class FrameTransforms:
    """Bundle of calibrated frame transforms.

    Parameters
    ----------
    H_IT:
        ``3x3`` image->table homography. If ``None``, image<->table methods
        raise (the system must be calibrated first).
    T_BT:
        ``3x3`` table->robot in-plane transform. Defaults to identity, with
        :attr:`robot_calibrated` ``False`` to flag that motion in robot frame
        is not yet trustworthy.
    z_table_in_robot:
        The robot-frame z value corresponding to the table surface (z=0 in
        table frame). Used to lift table z into robot z.
    """

    H_IT: Optional[np.ndarray] = None
    T_BT: np.ndarray = field(default_factory=lambda: np.eye(3))
    z_table_in_robot: float = 0.0
    image_calibrated: bool = False
    robot_calibrated: bool = False

    def __post_init__(self) -> None:
        if self.H_IT is not None:
            self.H_IT = np.asarray(self.H_IT, dtype=np.float64)
            self.image_calibrated = True
        self.T_BT = np.asarray(self.T_BT, dtype=np.float64)

    # -- image <-> table ------------------------------------------------
    def pixel_to_table_xy(self, uv: ArrayLike) -> np.ndarray:
        if self.H_IT is None:
            raise RuntimeError(
                "H_IT is not set: run `terafold calibrate-homography` first."
            )
        return pixel_to_table_xy(self.H_IT, uv)

    def table_xy_to_pixel(self, xy: ArrayLike) -> np.ndarray:
        if self.H_IT is None:
            raise RuntimeError(
                "H_IT is not set: run `terafold calibrate-homography` first."
            )
        return table_xy_to_pixel(self.H_IT, xy)

    # -- table <-> robot ------------------------------------------------
    def table_to_robot(self, xy: ArrayLike) -> np.ndarray:
        return table_to_robot(self.T_BT, xy)

    def robot_to_table(self, xy: ArrayLike) -> np.ndarray:
        return robot_to_table(self.T_BT, xy)

    def table_z_to_robot_z(self, z_table: float) -> float:
        """Lift a table-frame z (0 = table surface) into robot-frame z."""
        return self.z_table_in_robot + float(z_table)

    # -- convenience: full pixel -> robot pipeline ----------------------
    def pixel_to_robot_xy(self, uv: ArrayLike) -> np.ndarray:
        """Compose image->table->robot for an in-plane pixel point."""
        return self.table_to_robot(self.pixel_to_table_xy(uv))

    @classmethod
    def from_dict(cls, d: dict) -> "FrameTransforms":
        H_IT = np.asarray(d["H_IT"], dtype=np.float64) if d.get("H_IT") is not None else None
        T_BT = np.asarray(d["T_BT"], dtype=np.float64) if d.get("T_BT") is not None else np.eye(3)
        return cls(
            H_IT=H_IT,
            T_BT=T_BT,
            z_table_in_robot=float(d.get("z_table_in_robot", 0.0)),
            robot_calibrated=bool(d.get("robot_calibrated", False)),
        )

    def to_dict(self) -> dict:
        return {
            "H_IT": None if self.H_IT is None else self.H_IT.tolist(),
            "T_BT": self.T_BT.tolist(),
            "z_table_in_robot": self.z_table_in_robot,
            "image_calibrated": self.image_calibrated,
            "robot_calibrated": self.robot_calibrated,
        }
