"""Kinematics scaffold for TeraFold arms.

A place to REPRESENT a serial-chain kinematic model and validate the IK machinery
on a toy 2-link arm, while keeping the safety posture explicit: the custom 7-DOF
arm has no validated model, so :class:`CustomArmModel` REFUSES Cartesian / contact
IK until a model is measured and marked validated.

Public surface:

* raw units: :class:`RawUnitScale`, :func:`raw_to_deg`, :func:`deg_to_raw`
* forward kinematics: :class:`DHParam`, :func:`fk_dh`, :func:`toy_planar_fk`
* inverse kinematics (toy): :func:`ik_dls_planar`
* custom arm: :class:`CustomArmModel`
* status report: :func:`robot_model_status`, :func:`report_markdown`
"""

from __future__ import annotations

from terafold.kinematics.custom_arm_model import CustomArmModel
from terafold.kinematics.fk import DHParam, dh_transform, fk_dh, toy_planar_fk
from terafold.kinematics.ik_dls import ik_dls_planar
from terafold.kinematics.model_status import report_markdown, robot_model_status
from terafold.kinematics.raw_units import (
    DEFAULT_SCALE_DEG_PER_UNIT,
    RawUnitScale,
    deg_to_raw,
    raw_to_deg,
)

__all__ = [
    "RawUnitScale",
    "DEFAULT_SCALE_DEG_PER_UNIT",
    "raw_to_deg",
    "deg_to_raw",
    "DHParam",
    "dh_transform",
    "fk_dh",
    "toy_planar_fk",
    "ik_dls_planar",
    "CustomArmModel",
    "robot_model_status",
    "report_markdown",
]
