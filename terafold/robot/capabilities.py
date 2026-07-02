"""Central robot capability + safety-state evaluation (the single truth source).

One question, one answer: **"What is this robot allowed to do right now, and
why is everything above that blocked?"**

:func:`probe_capabilities` inspects the robot config, the joint map, the
calibration artifacts on disk, and (only if asked) performs a *read-only* serial
probe. It then evaluates the unlock ladder in :mod:`terafold.robot.safety_state`
and returns a :class:`RobotCapabilities` snapshot. The ``robot-status`` CLI is a
thin printer over this; the motion safety gate consults the same snapshot.

Safety posture:

* The probe is **read-only** — it only opens the port, pings, and reads. It never
  writes a servo. ``do_probe=False`` (the default) does not even open the port, so
  ``robot-status`` is safe and works with no hardware attached.
* Calibration artifacts are *discovered*, then their own ``valid_for_*`` flags are
  honored. Presence alone never unlocks anything.
* Contact (Level ≥6) additionally requires an explicit operator unlock flag and is
  reported as blocked here unless that flag and the full chain are present.

Pure stdlib + pyyaml. No numpy/torch/serial import at module load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from terafold.robot.arm_config import PhysicalArmConfig, configs_robots_dir, load_arm_config
from terafold.robot.joint_map import JointMap, joint_map_path, load_joint_map
from terafold.robot.safety_state import LEVELS, SafetyLevel

__all__ = [
    "RobotCapabilities",
    "probe_capabilities",
    "artifact_paths",
    "discover_artifacts",
]

# A joint that can perform the horizontal above-table sweep (Level 4 ghost fold).
SWEEP_JOINT_CANDIDATES = ("base_yaw", "base", "shoulder_yaw")


# ----------------------------------------------------------------------
# Artifact discovery (calibration / kinematics live in conventional paths)
# ----------------------------------------------------------------------


def runs_calibration_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo, "runs", "calibration")


def artifact_paths(robot: str) -> Dict[str, str]:
    """The conventional artifact locations for a robot (calibration writes here)."""
    cal = runs_calibration_dir()
    cfg = configs_robots_dir()
    return {
        "joint_map": joint_map_path(robot),
        "kinematics": os.path.join(cfg, f"{robot}_kinematics.yaml"),
        "camera_intrinsics": os.path.join(cal, f"{robot}_camera_intrinsics.yaml"),
        "table_homography": os.path.join(cal, f"{robot}_table_homography.yaml"),
        "robot_table_transform": os.path.join(cal, f"{robot}_robot_table_transform.yaml"),
    }


def _read_yaml(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            d = yaml.safe_load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return None


def _flag(d: Optional[dict], *keys: str) -> bool:
    """True if any of ``keys`` is truthy in the dict (missing dict → False)."""
    if not d:
        return False
    return any(bool(d.get(k)) for k in keys)


def discover_artifacts(robot: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Find calibration/kinematics artifacts and read their validity flags.

    ``extra`` may override any path (e.g. a ``--calibration`` passed on the CLI).
    Returns a dict of ``{name: {"path", "present", "valid_for_hover",
    "valid_for_contact", "data"}}`` plus a flat ``paths`` map.
    """
    paths = artifact_paths(robot)
    if extra:
        paths.update({k: v for k, v in extra.items() if v})
    out: Dict[str, Any] = {"paths": paths}
    for name, path in paths.items():
        present = bool(path) and os.path.exists(path)
        data = _read_yaml(path) if present else None
        out[name] = {
            "path": path,
            "present": present,
            "valid_for_hover": _flag(data, "valid_for_hover", "valid"),
            "valid_for_contact": _flag(data, "valid_for_contact"),
            "validated": _flag(data, "validated"),
            "data": data,
        }
    return out


# ----------------------------------------------------------------------
# The capability snapshot
# ----------------------------------------------------------------------


@dataclass
class RobotCapabilities:
    """Immutable-ish snapshot of what the robot can safely do right now."""

    robot: str
    port: str
    protocol: Optional[str]
    baudrate: Optional[int]
    dof: int

    sdk_available: bool = False
    probed: bool = False
    port_open: Optional[bool] = None          # None ⇒ not probed
    protocol_confirmed: bool = False

    configured_active_ids: List[int] = field(default_factory=list)
    responding_ids: List[int] = field(default_factory=list)
    active_ids: List[int] = field(default_factory=list)
    disabled_ids: List[int] = field(default_factory=list)

    can_read: bool = False
    can_write: bool = False

    joint_map_present: bool = False
    mapped_joints: List[str] = field(default_factory=list)
    joint_limits_known: bool = False
    sweep_joint_available: bool = False

    kinematics_present: bool = False
    kinematics_validated: bool = False

    camera_intrinsics_present: bool = False
    camera_intrinsics_valid: bool = False
    table_homography_present: bool = False
    table_homography_valid: bool = False
    robot_table_transform_present: bool = False
    robot_table_transform_valid: bool = False

    contact_unlock_flag: bool = False

    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    unlocked_level: int = 0
    level_status: List[dict] = field(default_factory=list)

    # -- derived ---------------------------------------------------------
    @property
    def calibration_valid_for_hover(self) -> bool:
        return (self.camera_intrinsics_valid and self.table_homography_valid
                and self.robot_table_transform_valid)

    @property
    def calibration_valid_for_contact(self) -> bool:
        return all((
            self.artifacts and self.camera_intrinsics_valid,
            self.table_homography_valid,
            self.robot_table_transform_valid,
        )) and self.contact_unlock_flag

    @property
    def contact_allowed(self) -> bool:
        """Contact is allowed ONLY at Level ≥6 with the explicit unlock flag."""
        return self.unlocked_level >= int(SafetyLevel.SOFT_CONTACT) and self.contact_unlock_flag

    def allowed_capabilities(self) -> List[str]:
        cmds: List[str] = []
        for s in self.level_status:
            if s["unlocked"]:
                cmds.extend(LEVELS[SafetyLevel(s["level"])].allowed_commands)
        # Preserve order, dedupe.
        seen, out = set(), []
        for c in cmds:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def blocked_capabilities(self) -> List[dict]:
        return [{"level": s["label"], "title": s["title"], "missing": s["missing"]}
                for s in self.level_status if not s["unlocked"]]

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["unlocked_label"] = SafetyLevel(self.unlocked_level).label
        d["contact_allowed"] = self.contact_allowed
        d["calibration_valid_for_hover"] = self.calibration_valid_for_hover
        return d

    def report_markdown(self) -> str:
        lines = [
            f"# Robot status — {self.robot}",
            "",
            f"- port: `{self.port}`  protocol: `{self.protocol}`  baud: `{self.baudrate}`  dof: {self.dof}",
            f"- SDK importable: {self.sdk_available}   probed: {self.probed}   "
            f"port_open: {self.port_open}   protocol_confirmed: {self.protocol_confirmed}",
            f"- configured active IDs: {self.configured_active_ids}",
            f"- responding IDs: {self.responding_ids if self.probed else '(not probed)'}",
            f"- disabled/missing IDs: {self.disabled_ids}",
            f"- can_read: {self.can_read}   can_write: {self.can_write}",
            f"- joint map: {self.joint_map_present}  ({', '.join(self.mapped_joints) or 'none'})",
            f"- sweep joint available: {self.sweep_joint_available}   "
            f"joint limits known: {self.joint_limits_known}",
            f"- kinematics: present={self.kinematics_present} validated={self.kinematics_validated}",
            f"- camera intrinsics: present={self.camera_intrinsics_present} "
            f"valid={self.camera_intrinsics_valid}",
            f"- table homography: present={self.table_homography_present} "
            f"valid={self.table_homography_valid}",
            f"- robot→table transform: present={self.robot_table_transform_present} "
            f"valid={self.robot_table_transform_valid}",
            f"- contact unlock flag: {self.contact_unlock_flag}",
            "",
            f"## Unlocked level: **{SafetyLevel(self.unlocked_level).label}**",
            f"Contact folding allowed: **{self.contact_allowed}**",
            "",
            "## Ladder",
        ]
        for s in self.level_status:
            mark = "✅" if s["unlocked"] else "🔒"
            lines.append(f"- {mark} {s['label']} — {s['title']}")
            if not s["unlocked"] and s["missing"]:
                for m in s["missing"]:
                    lines.append(f"    - needs: {m}")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Level evaluation
# ----------------------------------------------------------------------


def _evaluate_levels(c: RobotCapabilities) -> tuple[int, List[dict]]:
    """Return (highest unlocked level, per-level {unlocked, missing} status).

    A level is unlocked only if *all* lower levels are unlocked AND its own
    requirements are met (the ladder is monotone).
    """
    def reqs(level: SafetyLevel) -> List[str]:
        L = SafetyLevel
        if level == L.SIM_ONLY:
            return []
        if level == L.READ_ONLY:
            m = []
            if not c.sdk_available:
                m.append("scservo_sdk importable (pip install pyserial + vendor SDK)")
            if c.probed and not c.port_open:
                m.append("serial port reachable (check power + USB-C + port)")
            if not c.probed:
                m.append("a read-only probe (`robot-status --probe` / `servo-scan`)")
            return m
        if level == L.TINY_NUDGE:
            m = []
            if not c.protocol_confirmed:
                m.append("protocol confirmed (ping + read on a configured ID)")
            if not c.can_read:
                m.append("current position readable")
            return m
        if level == L.JOINT_MAP:
            m = []
            if not c.joint_map_present:
                m.append(f"a joint map ({os.path.basename(joint_map_path(c.robot))})")
            return m
        if level == L.GHOST_FOLD:
            m = []
            if not c.sweep_joint_available:
                m.append("a mapped sweep joint (e.g. base_yaw)")
            if not c.joint_limits_known:
                m.append("safe raw position limits")
            return m
        if level == L.CALIBRATED_HOVER:
            m = []
            if not c.camera_intrinsics_valid:
                m.append("valid camera intrinsics (`calibrate-camera`)")
            if not c.table_homography_valid:
                m.append("a valid table homography (`calibrate-table`)")
            if not c.robot_table_transform_valid:
                m.append("a valid robot→table transform (`fit-robot-table-transform`)")
            return m
        if level == L.SOFT_CONTACT:
            m = []
            if not c.contact_unlock_flag:
                m.append("explicit contact unlock (operator flag) — intentionally locked")
            m.append("hover-accuracy logs + low-force/compliance mode (not built yet)")
            return m
        if level == L.CONSTRAINED_FOLD:
            return ["repeatability metrics + success detector + ≥5 clean soft probes (not built yet)"]
        if level == L.REPEATED_AUTONOMY:
            return ["watchdog + intervention logging + batch caps over a validated fold (not built yet)"]
        return ["unknown"]

    status: List[dict] = []
    highest = int(SafetyLevel.SIM_ONLY)
    chain_ok = True
    for level in sorted(LEVELS, key=int):
        spec = LEVELS[level]
        missing = reqs(level)
        unlocked = chain_ok and not missing
        if not unlocked:
            chain_ok = False  # nothing above an unmet level can unlock
        if unlocked:
            highest = int(level)
        status.append({
            "level": int(level),
            "label": spec.level.label,
            "title": spec.title,
            "unlocked": unlocked,
            "missing": missing,
        })
    return highest, status


# ----------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------


def _expected_ids(cfg: PhysicalArmConfig) -> List[int]:
    """The full set of servo IDs we *expect* (1..dof), used to compute 'missing'."""
    if isinstance(cfg.servo_ids, list) and cfg.servo_ids:
        return sorted(int(i) for i in cfg.servo_ids)
    return list(range(1, int(cfg.dof) + 1))


def _joint_limits_known(jm: Optional[JointMap], cfg: PhysicalArmConfig) -> bool:
    """True if we have usable raw limits: per-joint in the map, or a global guard."""
    if jm is not None:
        for v in jm.servo_joint_map.values():
            if isinstance(v, dict) and ("raw_min" in v and "raw_max" in v):
                return True
    # A global safe-range guard is a (weaker) form of 'limits known' for the
    # bounded ghost sweep, which only commands a single mapped joint.
    return bool(cfg.safe_position_units) and len(cfg.safe_position_units) == 2


def probe_capabilities(
    robot: str,
    port: Optional[str] = None,
    do_probe: bool = False,
    backend: Any = None,
    contact_unlock: bool = False,
    calibration_overrides: Optional[Dict[str, str]] = None,
) -> RobotCapabilities:
    """Build the capability snapshot.

    Parameters
    ----------
    robot:
        Robot config name or path.
    port:
        Override the config port (else the config's).
    do_probe:
        If True, open the port READ-ONLY and ping/read to confirm the protocol.
        Default False — no serial access, purely artifact-based (safe, offline).
    backend:
        Inject a backend (used by tests / to reuse an already-open one). Must
        expose ``available()``, ``probe_protocol()``/``confirm()``,
        ``protocol_confirmed`` and ``read_all``. If None and ``do_probe``, a
        :class:`WaveshareSmsStsBackend` is created.
    contact_unlock:
        The explicit operator contact-unlock flag (default False ⇒ contact locked).
    calibration_overrides:
        Optional path overrides for calibration artifacts.
    """
    cfg = load_arm_config(robot)
    port = port or cfg.port
    caps = RobotCapabilities(
        robot=cfg.robot_name, port=port, protocol=cfg.protocol,
        baudrate=cfg.baud, dof=cfg.dof,
        configured_active_ids=list(cfg.active_servo_ids),
        contact_unlock_flag=bool(contact_unlock),
    )
    expected = _expected_ids(cfg)
    caps.disabled_ids = [i for i in expected if i not in cfg.active_servo_ids]

    # --- artifacts (always inspected; cheap, offline) ------------------
    arts = discover_artifacts(cfg.robot_name, calibration_overrides)
    caps.artifacts = arts["paths"]
    jm = load_joint_map(cfg.robot_name)
    caps.joint_map_present = jm is not None
    if jm is not None:
        caps.mapped_joints = [v.get("joint") for v in jm.servo_joint_map.values()
                              if isinstance(v, dict) and v.get("joint")]
        caps.sweep_joint_available = any(j in caps.mapped_joints for j in SWEEP_JOINT_CANDIDATES)
    caps.joint_limits_known = _joint_limits_known(jm, cfg)

    caps.kinematics_present = arts["kinematics"]["present"]
    caps.kinematics_validated = arts["kinematics"]["validated"]
    caps.camera_intrinsics_present = arts["camera_intrinsics"]["present"]
    caps.camera_intrinsics_valid = arts["camera_intrinsics"]["valid_for_hover"]
    caps.table_homography_present = arts["table_homography"]["present"]
    caps.table_homography_valid = arts["table_homography"]["valid_for_hover"]
    caps.robot_table_transform_present = arts["robot_table_transform"]["present"]
    caps.robot_table_transform_valid = arts["robot_table_transform"]["valid_for_hover"]

    # --- read-only serial probe (optional) -----------------------------
    if backend is None and do_probe:
        from terafold.robot.waveshare_sms_sts_backend import WaveshareSmsStsBackend

        backend = WaveshareSmsStsBackend(
            port=port, baudrate=cfg.baud, active_ids=cfg.active_servo_ids,
            safe_min_units=cfg.safe_position_units[0], safe_max_units=cfg.safe_position_units[1],
        )

    if backend is not None:
        caps.sdk_available = bool(getattr(backend, "available", lambda: getattr(
            backend, "sdk_available", False))())
        if do_probe:
            caps.probed = True
            probe = getattr(backend, "probe_protocol", None) or getattr(backend, "confirm")
            try:
                report = probe(cfg.active_servo_ids)
            except TypeError:
                report = probe()
            caps.port_open = bool(report.get("opened"))
            caps.responding_ids = sorted(int(i) for i in report.get("responders", []) or [])
            caps.protocol_confirmed = bool(getattr(backend, "protocol_confirmed", False)
                                           or report.get("confirmed"))
            caps.can_read = bool(report.get("read_ok")) or caps.protocol_confirmed
            caps.active_ids = [i for i in cfg.active_servo_ids if i in caps.responding_ids]
            try:
                backend.close()
            except Exception:
                pass
    else:
        # No probe: report SDK availability without opening anything.
        try:
            from terafold.robot.waveshare_sms_sts_backend import load_scservo_sdk

            caps.sdk_available = load_scservo_sdk() is not None
        except Exception:
            caps.sdk_available = False

    caps.can_write = caps.protocol_confirmed  # writes require a confirmed protocol
    if not caps.probed:
        caps.active_ids = list(cfg.active_servo_ids)  # configured (unverified)
        caps.notes.append("Not probed: levels above L0 need a read-only probe to confirm reads. "
                          "Run `robot-status --probe` (read-only) or `servo-scan`.")

    if caps.disabled_ids:
        caps.notes.append(f"IDs {caps.disabled_ids} are unconfirmed/missing and stay disabled "
                          "until physically verified (cabling/power/IDs).")
    if caps.kinematics_present and not caps.kinematics_validated:
        caps.notes.append("A kinematics model exists but is NOT validated — IK / contact stay locked.")

    caps.unlocked_level, caps.level_status = _evaluate_levels(caps)
    return caps
