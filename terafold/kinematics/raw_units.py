"""Raw servo-unit <-> joint-angle conversion (the *missing model* for the arm).

A bus servo reports a raw integer position (STS counts are ``0..4095`` over one
turn). Turning that into a joint angle in degrees needs three confirmed numbers
per joint: the raw ``home`` count, the ``scale`` (degrees per raw unit), and the
``sign`` (which way the joint turns as the count grows). For the custom 7-DOF arm
*none of these are confirmed*, so the scale is marked ``known=False`` and every
conversion safely returns ``None`` until a calibration provides real values.

The ``360 / 4096`` single-turn figure below is only a *guess* for the geometry of
a full-rotation STS encoder; it is deliberately NOT trusted (``known=False``) so
nothing downstream silently treats counts as validated angles.

Pure stdlib — no numpy needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DEFAULT_SCALE_DEG_PER_UNIT",
    "RawUnitScale",
    "raw_to_deg",
    "deg_to_raw",
]

# A *single-turn* STS encoder spans 4096 counts over 360 degrees. This is only a
# plausibility guess for an unconfirmed servo and is never assumed correct.
DEFAULT_SCALE_DEG_PER_UNIT: float = 360.0 / 4096.0  # ~= 0.087890625 deg/unit


@dataclass
class RawUnitScale:
    """The raw-count <-> degree mapping for ONE joint.

    Parameters
    ----------
    home:
        The raw count that corresponds to ``0`` degrees (the joint's zero pose).
    scale_deg_per_unit:
        Degrees of joint travel per raw count. ``None`` until measured.
    sign:
        ``+1`` if degrees increase with the raw count, ``-1`` if they decrease.
    known:
        ``True`` only when ``scale_deg_per_unit`` has been *confirmed* by a
        calibration. A populated-but-guessed scale must keep ``known=False``.
    """

    home: float = 2048.0
    scale_deg_per_unit: Optional[float] = None
    sign: int = 1
    known: bool = False

    @property
    def is_known(self) -> bool:
        """True only when a *confirmed* scale is available for conversions."""
        return bool(self.known) and self.scale_deg_per_unit is not None

    @classmethod
    def single_turn_guess(cls, home: float = 2048.0, sign: int = 1) -> "RawUnitScale":
        """A 360/4096 single-turn guess, deliberately marked ``known=False``."""
        return cls(home=float(home), scale_deg_per_unit=DEFAULT_SCALE_DEG_PER_UNIT,
                   sign=int(sign), known=False)

    @classmethod
    def confirmed(cls, home: float, scale_deg_per_unit: float, sign: int = 1) -> "RawUnitScale":
        """Build a CONFIRMED scale (``known=True``) from measured values."""
        return cls(home=float(home), scale_deg_per_unit=float(scale_deg_per_unit),
                   sign=int(sign), known=True)


def raw_to_deg(units: float, scale: RawUnitScale) -> Optional[float]:
    """Convert a raw servo count to joint degrees, or ``None`` if scale unknown.

    ``deg = sign * (units - home) * scale_deg_per_unit``.
    """
    if scale is None or not scale.is_known:
        return None
    return float(scale.sign) * (float(units) - float(scale.home)) * float(scale.scale_deg_per_unit)


def deg_to_raw(deg: float, scale: RawUnitScale) -> Optional[int]:
    """Convert joint degrees to a raw servo count, or ``None`` if scale unknown.

    Inverse of :func:`raw_to_deg`: ``units = home + sign * deg / scale``.
    """
    if scale is None or not scale.is_known:
        return None
    spu = float(scale.scale_deg_per_unit)
    if spu == 0.0:
        return None
    return int(round(float(scale.home) + float(scale.sign) * float(deg) / spu))
