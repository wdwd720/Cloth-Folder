"""The machine-enforced safety unlock ladder (Levels 0–8).

This is the *declarative* half of the central safety system: it describes, for
each level, what artifacts are required to unlock it, which commands are allowed
or forbidden, which flags are mandatory, what must be logged, and what aborts the
motion. :mod:`terafold.robot.capabilities` is the *evaluative* half — it inspects
a real robot/config/calibration and decides which level is currently unlocked.

Design rules (straight from the Deep Research digest):

* **Dry-run is the invariant default.** No level here moves hardware by itself.
* A level is unlocked only when *every* required artifact is satisfied.
* Contact (Level 6+) is never reachable without a full calibration chain AND an
  explicit operator unlock flag.
* If an assumption is unknown the system refuses and says *which* gate failed.

Pure stdlib — no numpy/serial/torch — so it can be imported anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

__all__ = ["SafetyLevel", "LevelSpec", "LEVELS", "level_spec", "highest_label"]


class SafetyLevel(IntEnum):
    """The eight-rung unlock ladder. Higher = more physical authority."""

    SIM_ONLY = 0           # simulation only — never opens the serial port
    READ_ONLY = 1          # ping + read positions; no writes
    TINY_NUDGE = 2         # a single-servo micro motion to map/characterize
    JOINT_MAP = 3          # one-joint bounded moves (joint map exists)
    GHOST_FOLD = 4         # bounded above-table sweep; no table descent
    CALIBRATED_HOVER = 5   # hover over table points; no contact
    SOFT_CONTACT = 6       # shallow cloth touch; no table scrape
    CONSTRAINED_FOLD = 7   # one constrained fold primitive
    REPEATED_AUTONOMY = 8  # batched autonomous trials

    @property
    def label(self) -> str:
        return f"L{int(self)} {self.name.lower()}"


@dataclass(frozen=True)
class LevelSpec:
    """Everything that defines one rung of the ladder."""

    level: SafetyLevel
    title: str
    summary: str
    required_artifacts: List[str] = field(default_factory=list)
    allowed_commands: List[str] = field(default_factory=list)
    forbidden_commands: List[str] = field(default_factory=list)
    required_flags: List[str] = field(default_factory=list)
    required_logs: List[str] = field(default_factory=list)
    abort_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "level": int(self.level),
            "label": self.level.label,
            "title": self.title,
            "summary": self.summary,
            "required_artifacts": list(self.required_artifacts),
            "allowed_commands": list(self.allowed_commands),
            "forbidden_commands": list(self.forbidden_commands),
            "required_flags": list(self.required_flags),
            "required_logs": list(self.required_logs),
            "abort_conditions": list(self.abort_conditions),
        }


# Conditions that abort motion at *every* armed level (the watchdog set).
_COMMON_ABORTS = [
    "STOP_TERAFOLD file present",
    "Ctrl+C / SIGINT",
    "no readback for >250–500 ms",
    "target outside safe raw limits",
    "explicit operator abort",
]

_TWO_FLAGS = ["--enable-motion", "--i-understand-this-moves-hardware"]


LEVELS: Dict[SafetyLevel, LevelSpec] = {
    SafetyLevel.SIM_ONLY: LevelSpec(
        level=SafetyLevel.SIM_ONLY,
        title="Simulation only",
        summary="No serial port is opened. Geometry/sim/perception/data tooling only.",
        required_artifacts=[],
        allowed_commands=[
            "sim-fold", "sim-real-ghost-fold", "plan-fold", "demo-image",
            "generate-synthetic", "export-lerobot", "eval-run", "robot-status",
        ],
        forbidden_commands=["any serial open", "any servo write"],
        required_flags=[],
        required_logs=["sim/plan run metadata"],
        abort_conditions=["n/a (no hardware)"],
    ),
    SafetyLevel.READ_ONLY: LevelSpec(
        level=SafetyLevel.READ_ONLY,
        title="Serial / read-only",
        summary="Open the port, ping, and read positions. No writes are sent.",
        required_artifacts=["scservo_sdk importable", "serial port reachable"],
        allowed_commands=["robot-probe", "servo-scan --read-only", "robot-status"],
        forbidden_commands=["servo-nudge (write)", "any WritePosEx"],
        required_flags=[],
        required_logs=["runs/servo_scan/<ts>/scan.json"],
        abort_conditions=["serial read error", "STOP_TERAFOLD file present"],
    ),
    SafetyLevel.TINY_NUDGE: LevelSpec(
        level=SafetyLevel.TINY_NUDGE,
        title="Tiny single-servo nudge",
        summary="A bounded micro-move of ONE servo (then return) to map/characterize.",
        required_artifacts=["protocol confirmed (ping+read)", "current position readable"],
        allowed_commands=["servo-nudge", "map-servo-joints (per-id)", "characterize-servos"],
        forbidden_commands=["multi-joint motion", "any contact"],
        required_flags=_TWO_FLAGS,
        required_logs=["runs/real_motion_logs/<cmd>_<ts>.jsonl"],
        abort_conditions=_COMMON_ABORTS,
    ),
    SafetyLevel.JOINT_MAP: LevelSpec(
        level=SafetyLevel.JOINT_MAP,
        title="Joint mapping",
        summary="One-joint bounded moves with a known servo→joint map and signs.",
        required_artifacts=["protocol confirmed", "configs/robots/<robot>_joint_map.yaml"],
        allowed_commands=["map-servo-joints", "characterize-servos", "single bounded joint move"],
        forbidden_commands=["table contact", "Cartesian IK"],
        required_flags=_TWO_FLAGS,
        required_logs=["joint map YAML", "per-id motion logs"],
        abort_conditions=_COMMON_ABORTS,
    ),
    SafetyLevel.GHOST_FOLD: LevelSpec(
        level=SafetyLevel.GHOST_FOLD,
        title="Above-table ghost fold",
        summary="Bounded above-table sweep derived from the fold direction. No descent.",
        required_artifacts=[
            "protocol confirmed", "joint map present", "a mapped sweep joint (base_yaw)",
            "safe raw position limits", "a successful dry-run",
        ],
        allowed_commands=["real-image-ghost-fold", "replay-joint-demo (air)"],
        forbidden_commands=["any descent into the table plane", "contact grasp"],
        required_flags=_TWO_FLAGS,
        required_logs=["ghost waypoints", "readbacks", "safety-gate results"],
        abort_conditions=_COMMON_ABORTS + ["estimated EE below clearance"],
    ),
    SafetyLevel.CALIBRATED_HOVER: LevelSpec(
        level=SafetyLevel.CALIBRATED_HOVER,
        title="Calibrated hover",
        summary="Hover over image-derived table points. Still no contact.",
        required_artifacts=[
            "camera intrinsics (valid)", "table homography (valid for hover)",
            "robot→table transform (valid for hover)", "ghost fold validated",
        ],
        allowed_commands=["hover-to-grasp (future)", "calibrated hover planning"],
        forbidden_commands=["contact grasp", "table scrape"],
        required_flags=_TWO_FLAGS,
        required_logs=["hover accuracy logs", "held-out calibration errors"],
        abort_conditions=_COMMON_ABORTS + ["hover error above gate"],
    ),
    SafetyLevel.SOFT_CONTACT: LevelSpec(
        level=SafetyLevel.SOFT_CONTACT,
        title="Soft contact probe",
        summary="A very shallow cloth touch. Low speed, low force, explicit unlock.",
        required_artifacts=[
            "full calibration valid for contact", "≥10–20 hover runs logged",
            "compliance/low-force mode", "explicit contact-unlock flag",
        ],
        allowed_commands=["soft contact probe (future)"],
        forbidden_commands=["table scrape", "free exploration"],
        required_flags=_TWO_FLAGS + ["--i-understand-contact", "--unlock-contact"],
        required_logs=["contact attempt logs", "force/compliance traces"],
        abort_conditions=_COMMON_ABORTS + ["force/compliance limit exceeded"],
    ),
    SafetyLevel.CONSTRAINED_FOLD: LevelSpec(
        level=SafetyLevel.CONSTRAINED_FOLD,
        title="Constrained contact fold",
        summary="One constrained fold primitive after repeatable soft-contact probes.",
        required_artifacts=[
            "repeatability metrics within gate", "success detector",
            "≥5 shallow contact probes with no scrape",
        ],
        allowed_commands=["constrained fold primitive (future)"],
        forbidden_commands=["free exploration", "uncapped trials"],
        required_flags=_TWO_FLAGS + ["--i-understand-contact", "--unlock-contact"],
        required_logs=["fold attempt logs", "success/failure labels"],
        abort_conditions=_COMMON_ABORTS + ["fold deviation beyond envelope"],
    ),
    SafetyLevel.REPEATED_AUTONOMY: LevelSpec(
        level=SafetyLevel.REPEATED_AUTONOMY,
        title="Repeated autonomous trials",
        summary="Batched autonomous fold trials with watchdog + intervention logging.",
        required_artifacts=["watchdog active", "intervention logging", "batch caps"],
        allowed_commands=["batch fold trials (future)"],
        forbidden_commands=["uncapped runs"],
        required_flags=_TWO_FLAGS + ["--i-understand-contact", "--unlock-contact"],
        required_logs=["per-trial logs", "intervention events"],
        abort_conditions=_COMMON_ABORTS + ["intervention rate above threshold"],
    ),
}


def level_spec(level: SafetyLevel | int) -> LevelSpec:
    """Return the :class:`LevelSpec` for a level (accepts the int too)."""
    return LEVELS[SafetyLevel(int(level))]


def highest_label(level: SafetyLevel | int) -> str:
    return SafetyLevel(int(level)).label
