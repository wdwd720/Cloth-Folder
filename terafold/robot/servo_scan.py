"""Read-only servo bus scan for a Waveshare SMS/STS arm.

This is reconnaissance only: it opens the backend with a read-only protocol probe,
pings a range of servo IDs, and (for the responders) reads back position + speed.
It NEVER writes a servo target — not even in passing — so it is always safe to run.

The report it builds answers two operator questions:

* which IDs are physically on the bus right now (``found_ids``), and
* which IDs the config EXPECTS but did not answer (``missing_ids``), together with
  concrete physical-debugging suggestions.

Hardware access goes through any object matching the verified-backend interface
(:class:`terafold.robot.waveshare_sms_sts_backend.WaveshareSmsStsBackend`): a
``ping(ids)->{id:bool}`` and ``read_pos_speed(id)->(pos, speed)`` are all that is
required. Tests inject a mock backend; no live arm is ever needed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from terafold.data.episode_schema import write_json
from terafold.robot.arm_config import load_arm_config

__all__ = ["parse_id_range", "run_servo_scan", "servo_scan_dir", "MISSING_ID_DEBUG"]


# Physical-debugging checklist printed/written for every expected-but-missing ID.
MISSING_ID_DEBUG: List[str] = [
    "Check power: the 9-12.6V supply must be ON and the bus rail powered.",
    "Check bus wiring (G/V/D) and that every daisy-chain connector is fully seated.",
    "Confirm the servo's actual ID with the vendor software (it may differ from expected).",
    "Look for ID conflicts: two servos sharing one ID will not both answer a ping.",
    "Verify the baud rate matches (this SMS/STS arm uses 1,000,000 baud).",
    "Swap the servo lead / bus port to rule out a broken cable.",
    "If the servo is intentionally absent, drop it from active_servo_ids in the config.",
]


def parse_id_range(spec: str) -> List[int]:
    """Parse a servo-ID range spec into a sorted, de-duplicated list of ints.

    Supports comma-separated single IDs and inclusive ``lo-hi`` ranges, mixed
    freely:

    * ``"1-30"``      -> ``[1, 2, ..., 30]``
    * ``"1,2,5,6"``   -> ``[1, 2, 5, 6]``
    * ``"1-5,10"``    -> ``[1, 2, 3, 4, 5, 10]``

    Whitespace around tokens is ignored and a reversed range (``"6-1"``) is
    accepted (treated as ``1-6``). Returns ``[]`` for an empty/blank spec.
    """
    ids: set[int] = set()
    if not spec:
        return []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if lo > hi:
                lo, hi = hi, lo
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(token))
    return sorted(ids)


def servo_scan_dir(now: str = "", base: Optional[str] = None) -> str:
    """Resolve the run directory ``runs/servo_scan/<now or timestamp>``."""
    base = base or os.path.join("runs", "servo_scan")
    stamp = now or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, stamp)


def _read_pos_speed(backend: Any, servo_id: int):
    """Best-effort ``(pos, speed)`` read; ``(None, None)`` if unsupported/failed."""
    fn = getattr(backend, "read_pos_speed", None)
    if callable(fn):
        try:
            res = fn(servo_id)
        except Exception:
            res = None
        if res is not None:
            try:
                pos, speed = res
                return int(pos), int(speed)
            except Exception:
                return None, None
    rp = getattr(backend, "read_position", None)
    if callable(rp):
        try:
            pos = rp(servo_id)
        except Exception:
            pos = None
        if pos is not None:
            return int(pos), None
    return None, None


def _summary_markdown(report: Dict[str, Any]) -> str:
    """Render the human-readable ``summary.md`` for a scan report."""
    lines: List[str] = []
    lines.append("# TeraFold - Servo Bus Scan (read-only)")
    lines.append("")
    lines.append(f"- robot: `{report['robot']}`")
    lines.append(f"- port: `{report['port']}`")
    rng = report["scanned_range"]
    lines.append(f"- scanned IDs: {rng[0]}..{rng[1]} ({report['num_scanned']} IDs)")
    lines.append(f"- protocol_confirmed: **{report['protocol_confirmed']}**")
    lines.append(f"- found: {report['num_found']}   missing-expected: {len(report['missing_ids'])}")
    lines.append("")
    lines.append("> Read-only reconnaissance. NO servo write was sent (0 commands).")
    lines.append("")

    lines.append("## Found servos")
    if report["found_ids"]:
        lines.append("")
        lines.append("| ID | ping | position | speed |")
        lines.append("|----|------|----------|-------|")
        for sid in report["found_ids"]:
            e = report["per_id"][str(sid)]
            pos = "-" if e["pos"] is None else e["pos"]
            spd = "-" if e["speed"] is None else e["speed"]
            lines.append(f"| {sid} | ok | {pos} | {spd} |")
    else:
        lines.append("")
        lines.append("_No servos responded in the scanned range._")
    lines.append("")

    lines.append("## Missing expected servos")
    if report["missing_ids"]:
        lines.append("")
        lines.append("Expected from the config but did NOT respond: "
                     + ", ".join(str(i) for i in report["missing_ids"]) + ".")
        lines.append("")
        lines.append("Physical debugging checklist:")
        for tip in MISSING_ID_DEBUG:
            lines.append(f"- {tip}")
    else:
        lines.append("")
        lines.append("_All expected servos responded._")
    lines.append("")

    if report["unexpected_ids"]:
        lines.append("## Unexpected responders")
        lines.append("")
        lines.append("Responded but not in the config's active_servo_ids: "
                     + ", ".join(str(i) for i in report["unexpected_ids"]) + ".")
        lines.append("Confirm these IDs before enabling them for motion.")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_servo_scan(
    robot: str,
    port: Optional[str] = None,
    id_min: int = 1,
    id_max: int = 30,
    backend: Any = None,
    out_dir: Optional[str] = None,
    now: str = "",
) -> Dict[str, Any]:
    """Read-only scan of servo IDs ``id_min..id_max`` on ``robot``'s bus.

    Opens the backend with a read-only protocol probe, pings each ID, and reads
    back ``(position, speed)`` for the responders. Computes the expected-but-missing
    IDs from the config (nominal arm complement ``1..dof`` together with
    ``active_servo_ids``) and writes ``scan.json`` + ``summary.md`` under
    ``runs/servo_scan/<now or timestamp>`` (or ``out_dir`` if given).

    NEVER writes a servo target. Returns a report dict with ``found_ids``,
    ``missing_ids``, ``scan_json``, ``summary_md`` and the full per-ID detail.
    """
    cfg = load_arm_config(robot)
    active = sorted({int(i) for i in (cfg.active_servo_ids or [])})
    # The nominal full complement (1..dof) plus any explicitly-active IDs.
    expected = sorted(set(range(1, int(cfg.dof) + 1)) | set(active))

    scanned = list(range(int(id_min), int(id_max) + 1))
    port = port or cfg.port

    owns_backend = backend is None
    if backend is None:
        from terafold.robot.waveshare_sms_sts_backend import WaveshareSmsStsBackend

        safe = cfg.safe_position_units or [400, 3700]
        backend = WaveshareSmsStsBackend(
            port=port,
            baudrate=cfg.baud,
            active_ids=active or scanned,
            safe_min_units=int(safe[0]),
            safe_max_units=int(safe[1]),
        )

    # Read-only protocol probe (alias of confirm(); never writes). Best-effort.
    probe_report: Dict[str, Any] = {}
    probe_fn = getattr(backend, "probe_protocol", None) or getattr(backend, "confirm", None)
    if callable(probe_fn):
        try:
            probe_report = probe_fn(scanned) or {}
        except TypeError:
            try:
                probe_report = probe_fn() or {}
            except Exception as exc:  # pragma: no cover - defensive
                probe_report = {"ok": False, "reason": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            probe_report = {"ok": False, "reason": str(exc)}

    # Ping the whole scanned range (read-only).
    try:
        pings = backend.ping(scanned) or {}
    except Exception as exc:  # pragma: no cover - defensive
        pings = {}
        probe_report.setdefault("ping_error", str(exc))
    pings = {int(k): bool(v) for k, v in pings.items()}

    per_id: Dict[str, Any] = {}
    found_ids: List[int] = []
    for sid in scanned:
        ok = bool(pings.get(sid, False))
        entry: Dict[str, Any] = {"ping": ok, "pos": None, "speed": None}
        if ok:
            found_ids.append(sid)
            pos, speed = _read_pos_speed(backend, sid)
            entry["pos"], entry["speed"] = pos, speed
        per_id[str(sid)] = entry
    found_ids = sorted(found_ids)

    scanned_set = set(scanned)
    missing_ids = sorted((set(expected) & scanned_set) - set(found_ids))
    unexpected_ids = sorted(set(found_ids) - set(active))

    if owns_backend:
        try:
            backend.close()
        except Exception:  # pragma: no cover - defensive
            pass

    run_dir = out_dir or servo_scan_dir(now)
    os.makedirs(run_dir, exist_ok=True)
    scan_json = os.path.join(run_dir, "scan.json")
    summary_md = os.path.join(run_dir, "summary.md")

    report: Dict[str, Any] = {
        "robot": robot,
        "port": port,
        "scanned_range": [int(id_min), int(id_max)],
        "num_scanned": len(scanned),
        "protocol_confirmed": bool(getattr(backend, "protocol_confirmed", False)),
        "expected_ids": expected,
        "active_servo_ids": active,
        "found_ids": found_ids,
        "num_found": len(found_ids),
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "per_id": per_id,
        "probe": probe_report,
        "commands_sent": 0,
        "read_only": True,
        "missing_id_debug": MISSING_ID_DEBUG if missing_ids else [],
        "scan_json": scan_json,
        "summary_md": summary_md,
        "note": ("Read-only servo bus scan. NO servo target was written. "
                 "missing_ids are expected (config) IDs that did not answer a ping."),
    }

    write_json(scan_json, report)
    with open(summary_md, "w") as f:
        f.write(_summary_markdown(report))

    return report
