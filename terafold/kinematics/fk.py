"""Forward kinematics primitives (standard Denavit-Hartenberg + a toy 2R arm).

This is a *scaffold*: it can REPRESENT a serial-chain kinematic model so a real
one can be dropped in later, and it provides a tiny analytic 2-link planar arm
used to test the IK solver. It does NOT contain a validated model for the custom
7-DOF arm — :mod:`terafold.kinematics.custom_arm_model` refuses IK for that arm
until a model is marked validated.

Standard (distal) DH convention for link ``i`` with joint variable ``theta_i``::

    T_i = Rz(theta_i + theta_offset_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i)

Pure numpy (a TeraFold core dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = ["DHParam", "dh_transform", "fk_dh", "toy_planar_fk"]


@dataclass
class DHParam:
    """One row of standard Denavit-Hartenberg parameters.

    Parameters
    ----------
    a:
        Link length (translation along the rotated x-axis), metres.
    alpha:
        Link twist (rotation about the x-axis), radians.
    d:
        Link offset (translation along the z-axis), metres.
    theta_offset:
        Constant added to the joint variable before the z-rotation, radians.
    """

    a: float = 0.0
    alpha: float = 0.0
    d: float = 0.0
    theta_offset: float = 0.0


def dh_transform(param: DHParam, q: float) -> np.ndarray:
    """The 4x4 homogeneous transform for a single DH link at joint value ``q``."""
    theta = float(q) + float(param.theta_offset)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(param.alpha), np.sin(param.alpha)
    a, d = float(param.a), float(param.d)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fk_dh(params: List[DHParam], q: List[float]) -> np.ndarray:
    """Forward kinematics of a serial chain as the product of DH link transforms.

    Returns the 4x4 homogeneous transform of the end-effector frame relative to
    the base frame. ``len(q)`` must equal ``len(params)``.
    """
    if len(params) != len(q):
        raise ValueError(f"got {len(q)} joint values for {len(params)} DH links")
    T = np.eye(4, dtype=np.float64)
    for param, qi in zip(params, q):
        T = T @ dh_transform(param, qi)
    return T


def toy_planar_fk(l1: float, l2: float, q1: float, q2: float) -> Tuple[float, float]:
    """Analytic forward kinematics of a 2-link planar (2R) arm.

    ``x = l1 cos q1 + l2 cos(q1 + q2)``,  ``y = l1 sin q1 + l2 sin(q1 + q2)``.
    Used as a known-good reference for the DLS IK tests.
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return float(x), float(y)
