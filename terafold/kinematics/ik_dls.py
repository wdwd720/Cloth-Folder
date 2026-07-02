"""Damped least squares (DLS) inverse kinematics for a TOY 2-link planar arm.

This solves IK for the analytic 2R arm in :func:`terafold.kinematics.fk.toy_planar_fk`
using the Levenberg-Marquardt / damped-least-squares update with a *numerical*
Jacobian. It exists to exercise and validate the IK machinery on a chain we have
a closed-form reference for. It is NOT used to move the custom 7-DOF arm; that
arm's IK stays refused until a validated model exists.

DLS update (robust through singularities)::

    dq = J^T (J J^T + lambda^2 I)^{-1} e ,   e = target - fk(q)

Pure numpy (a TeraFold core dependency).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from terafold.kinematics.fk import toy_planar_fk

__all__ = ["ik_dls_planar"]


def _planar_jacobian(l1: float, l2: float, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical 2x2 Jacobian d(x, y)/d(q1, q2) via central differences."""
    J = np.zeros((2, 2), dtype=np.float64)
    for j in range(2):
        dq = q.copy()
        dq[j] += eps
        fp = np.array(toy_planar_fk(l1, l2, dq[0], dq[1]), dtype=np.float64)
        dq[j] -= 2.0 * eps
        fm = np.array(toy_planar_fk(l1, l2, dq[0], dq[1]), dtype=np.float64)
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


def ik_dls_planar(
    target_xy: Sequence[float],
    l1: float,
    l2: float,
    q0: Optional[Sequence[float]] = None,
    lam: float = 0.1,
    iters: int = 200,
    tol: float = 1e-4,
) -> Dict[str, object]:
    """Solve 2R planar IK by damped least squares.

    Parameters
    ----------
    target_xy:
        Desired end-effector ``(x, y)``.
    l1, l2:
        Link lengths.
    q0:
        Initial joint angles (radians). Defaults to a non-singular ``[0.1, 0.5]``.
    lam:
        Damping factor (larger = more stable, slower near a solution).
    iters:
        Maximum DLS iterations.
    tol:
        Convergence tolerance on the Euclidean position error.

    Returns
    -------
    dict
        ``{"ok": bool, "q": [q1, q2], "error": float, "iters": int}``. ``ok`` is
        True only when the final position error is within ``tol`` (an unreachable
        target leaves ``ok=False`` with the residual error reported).
    """
    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    q = np.array([0.1, 0.5] if q0 is None else q0, dtype=np.float64).reshape(2)
    lam2 = float(lam) ** 2
    eye2 = np.eye(2, dtype=np.float64)

    err = float("inf")
    used = 0
    for used in range(1, int(iters) + 1):
        cur = np.array(toy_planar_fk(l1, l2, q[0], q[1]), dtype=np.float64)
        e = target - cur
        err = float(np.hypot(e[0], e[1]))
        if err <= tol:
            break
        J = _planar_jacobian(l1, l2, q)
        # dq = J^T (J J^T + lam^2 I)^{-1} e
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * eye2, e)
        q = q + dq

    # Wrap angles into (-pi, pi] for a tidy, comparable result.
    q_wrapped = ((q + np.pi) % (2.0 * np.pi)) - np.pi
    return {
        "ok": bool(err <= tol),
        "q": [float(q_wrapped[0]), float(q_wrapped[1])],
        "error": float(err),
        "iters": int(used),
    }
