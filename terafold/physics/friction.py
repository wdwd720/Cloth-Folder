"""Friction / surface-interaction heuristics.

These are deliberately simple, *interpretable* proxies — not a tribology model.
They convert qualitative hints (from the task config) into scalar coefficients
the planner and the quasi-static model can reason about, and they expose the
relationships the residual learner can later correct.
"""

from __future__ import annotations

import numpy as np

# Qualitative hint -> approximate Coulomb friction coefficient (cloth on table).
# Real towel-on-laminate mu is ~0.3-0.6; we spread the hints around that.
HINT_TO_FRICTION = {
    "very_low": 0.15,
    "low": 0.25,
    "medium": 0.40,
    "high": 0.60,
    "very_high": 0.80,
}

# Qualitative stiffness hint -> bending stiffness proxy in [0, 1].
HINT_TO_STIFFNESS = {
    "very_low": 0.10,
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "very_high": 0.95,
}

__all__ = [
    "HINT_TO_FRICTION",
    "HINT_TO_STIFFNESS",
    "estimate_table_friction",
    "estimate_stiffness",
    "friction_force_proxy",
    "slip_risk",
]


def estimate_table_friction(friction_hint: str) -> float:
    """Map a friction hint string to a coefficient. Unknown hints -> medium."""
    return float(HINT_TO_FRICTION.get(str(friction_hint).lower(), HINT_TO_FRICTION["medium"]))


def estimate_stiffness(stiffness_hint: str) -> float:
    """Map a stiffness hint string to a bending-stiffness proxy in [0, 1]."""
    return float(HINT_TO_STIFFNESS.get(str(stiffness_hint).lower(), HINT_TO_STIFFNESS["medium"]))


def friction_force_proxy(normal_force: float, mu: float) -> float:
    """Coulomb friction proxy ``F = mu * N`` (clamped non-negative)."""
    return float(max(0.0, mu * normal_force))


def slip_risk(drag_distance_m: float, lift_height_m: float, mu: float) -> float:
    """Heuristic slip risk in [0, 1].

    Slip risk grows with how far the cloth is dragged and with friction, and
    shrinks as the lift height increases (lifting reduces contact / dragging).
    """
    lift = max(lift_height_m, 1e-4)
    raw = mu * drag_distance_m / (lift + drag_distance_m)
    return float(np.clip(raw, 0.0, 1.0))
