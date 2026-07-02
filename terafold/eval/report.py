"""Roll a run's logs into a metrics dict + a Markdown evaluation report.

:func:`eval_run` is deliberately forgiving: it accepts a motion-log ``.jsonl``
path, an episode directory, a ``.json`` file, or an in-memory list/dict of
records, and computes whatever metric *families* are derivable from the fields
that happen to be present:

* **control** — settle error / overshoot / time-to-settle / failure-to-reach,
  from servo write targets paired with readback traces;
* **perception** — keypoint-confidence gate and the perception mode used;
* **calibration** — reprojection RMS when correspondences are logged;
* **fold** — crease crossing / edge error / success when corners are logged;
* **safety** — gate refusals, dry-run posture, and real-motion requests.

A family with no usable fields is reported as ``available: False`` with a short
note rather than raising. :func:`report_markdown` renders the dict; failures
(e.g. a servo that never reached its target) surface as ``WARNING`` lines.

Pure numpy + stdlib; reuses the JSON(L) helpers in
:mod:`terafold.data.episode_schema`.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from terafold.data.episode_schema import read_json, read_jsonl
from terafold.eval import calibration_metrics as cal
from terafold.eval import control_metrics as ctl
from terafold.eval import fold_metrics as fold
from terafold.eval import perception_metrics as perc

__all__ = ["eval_run", "report_markdown"]

_WRITE_EVENTS = {"moved", "move", "write", "write_pos", "write_pos_ex", "command", "target"}
_READBACK_EVENTS = {"readback", "read", "sample", "present", "observation"}


# ----------------------------------------------------------------------
# Record loading (path / dir / list / dict — all best-effort)
# ----------------------------------------------------------------------


def _to_records(logs: Any) -> List[Dict[str, Any]]:
    if logs is None:
        return []
    if isinstance(logs, str):
        if os.path.isdir(logs):
            return _records_from_dir(logs)
        if not os.path.exists(logs):
            return []
        if logs.endswith(".json"):
            obj = read_json(logs)
            if isinstance(obj, list):
                return [r for r in obj if isinstance(r, dict)]
            if isinstance(obj, dict):
                return _to_records(obj)
            return []
        return [r for r in read_jsonl(logs) if isinstance(r, dict)]
    if isinstance(logs, dict):
        for key in ("records", "events", "rows", "logs"):
            if isinstance(logs.get(key), list):
                return [r for r in logs[key] if isinstance(r, dict)]
        return [logs]
    if isinstance(logs, (list, tuple)):
        return [r for r in logs if isinstance(r, dict)]
    return []


def _records_from_dir(dirpath: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for fn in sorted(os.listdir(dirpath)):
        if fn.endswith(".jsonl"):
            recs.extend(r for r in read_jsonl(os.path.join(dirpath, fn)) if isinstance(r, dict))
    for special in ("result.json", "safety.json", "plan.json"):
        p = os.path.join(dirpath, special)
        if os.path.exists(p):
            try:
                recs.append({"event": special.split(".")[0], "data": read_json(p)})
            except Exception:
                pass
    return recs


def _data(rec: Dict[str, Any]) -> Dict[str, Any]:
    d = rec.get("data")
    return d if isinstance(d, dict) else rec


def _event(rec: Dict[str, Any], data: Dict[str, Any]) -> str:
    return str(rec.get("event") or data.get("event") or "").lower()


# ----------------------------------------------------------------------
# Metric families
# ----------------------------------------------------------------------


def _control_family(records: List[Dict[str, Any]], settle_tol: float) -> Dict[str, Any]:
    """Derive per-servo control metrics from writes + readback traces."""
    traces: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    targets: Dict[Any, float] = {}
    episodes: List[tuple] = []  # (servo_id, target, trace)

    for rec in records:
        data = _data(rec)
        ev = _event(rec, data)
        sid = data.get("servo_id", data.get("id"))
        embedded = data.get("trace") or data.get("readback") or data.get("samples")
        tgt = data.get("target", data.get("target_units"))
        if isinstance(embedded, list) and embedded and tgt is not None:
            episodes.append((sid, float(tgt), embedded))
            continue
        if tgt is not None and (ev in _WRITE_EVENTS or "target" in data or "target_units" in data):
            targets[sid] = float(tgt)
        pos = data.get("pos", data.get("position", data.get("present_pos")))
        if pos is not None and (ev in _READBACK_EVENTS or "pos" in data
                                or "position" in data or "present_pos" in data):
            traces[sid].append({
                "t": data.get("t", data.get("time", data.get("timestamp"))),
                "pos": pos,
                "speed": data.get("speed", data.get("present_speed", data.get("vel"))),
            })

    for sid, target in targets.items():
        if traces.get(sid):
            episodes.append((sid, target, traces[sid]))

    if not episodes:
        return {"available": False,
                "note": "no matching servo write+readback pairs in the logs."}

    per_servo: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for sid, target, trace in episodes:
        se = ctl.settle_error(trace, target)
        ftr = ctl.failure_to_reach(trace, target, settle_tol)
        row = {
            "servo_id": sid,
            "target": float(target),
            "settle_error": se,
            "overshoot": ctl.overshoot(trace, target),
            "time_to_settle": ctl.time_to_settle(trace, target, settle_tol),
            "failure_to_reach": ftr,
            "readback_noise": ctl.readback_noise(trace),
            "n_samples": len(list(trace)),
        }
        per_servo.append(row)
        if ftr:
            warnings.append(
                f"servo {sid} FAILED TO REACH target {target:.0f} "
                f"(settle_error {se:.1f} > tol {settle_tol:.0f})")

    settle_errors = [r["settle_error"] for r in per_servo if math.isfinite(r["settle_error"])]
    return {
        "available": True,
        "settle_tol": float(settle_tol),
        "n_servos": len(per_servo),
        "worst_settle_error": float(max(settle_errors)) if settle_errors else float("nan"),
        "any_failure_to_reach": any(r["failure_to_reach"] for r in per_servo),
        "per_servo": per_servo,
        "warnings": warnings,
    }


def _perception_family(records: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    confidences: List[float] = []
    modes: List[str] = []
    for rec in records:
        data = _data(rec)
        c = data.get("confidence")
        if isinstance(c, (int, float)):
            confidences.append(float(c))
        cs = data.get("confidences", data.get("keypoint_confidence"))
        if isinstance(cs, (list, tuple)):
            confidences.extend(float(x) for x in cs if isinstance(x, (int, float)))
        mode = data.get("perception")
        if isinstance(mode, str) and mode not in modes:
            modes.append(mode)

    if not confidences and not modes:
        return {"available": False,
                "note": "no perception confidence or mode recorded in the logs."}

    warnings: List[str] = []
    out: Dict[str, Any] = {"available": True, "modes": modes, "threshold": float(threshold)}
    if confidences:
        passed = perc.confidence_pass(confidences, threshold)
        out["min_confidence"] = float(min(confidences))
        out["mean_confidence"] = float(sum(confidences) / len(confidences))
        out["passed"] = passed
        if not passed:
            warnings.append(
                f"perception confidence {min(confidences):.2f} below threshold {threshold:.2f}")
    if "classical_fallback" in modes:
        out["note"] = "classical numpy fallback in use (no learned keypoint model)."
    out["warnings"] = warnings
    return out


def _calibration_family(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rms_values: List[float] = []
    for rec in records:
        data = _data(rec)
        rms = data.get("calibration_rms", data.get("rms"))
        if rms is None and all(k in data for k in ("H", "image_pts", "table_pts")):
            try:
                rms = cal.homography_rms(data["H"], data["image_pts"], data["table_pts"])
            except Exception:
                rms = None
        if rms is None and all(k in data for k in ("touch_pred", "touch_gt")):
            try:
                rms = cal.robot_touch_rms(data["touch_pred"], data["touch_gt"])
            except Exception:
                rms = None
        if isinstance(rms, (int, float)):
            rms_values.append(float(rms))
    if not rms_values:
        return {"available": False,
                "note": "no calibration correspondences or RMS recorded in the logs."}
    return {"available": True, "rms_values": rms_values,
            "worst_rms": float(max(rms_values)), "n": len(rms_values)}


def _fold_family(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for rec in records:
        data = _data(rec)
        entry: Dict[str, Any] = {}
        try:
            if "edge_error" in data:
                entry["edge_error"] = float(data["edge_error"])
            elif all(k in data for k in ("after_corners", "target_corners")):
                entry["edge_error"] = fold.final_edge_error(
                    data["after_corners"], data["target_corners"])
            if all(k in data for k in ("before_corners", "after_corners", "fold_line")):
                entry["crossed_crease"] = fold.crossed_crease(
                    data["before_corners"], data["after_corners"], data["fold_line"])
            if "edge_error" in entry and "fold_threshold" in data:
                entry["success"] = fold.fold_success(entry["edge_error"], data["fold_threshold"])
            elif "success" in data and isinstance(data["success"], bool):
                entry["success"] = data["success"]
        except Exception:
            continue
        if entry:
            results.append(entry)
            if entry.get("success") is False:
                warnings.append("a logged fold was scored UNSUCCESSFUL.")
    if not results:
        return {"available": False,
                "note": "no fold corners / edge-error recorded in the logs."}
    return {"available": True, "folds": results, "warnings": warnings}


def _safety_family(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    refusals: List[str] = []
    aborts: List[str] = []
    num_moves = 0
    dry_run_seen = False
    real_motion_requested = False
    for rec in records:
        data = _data(rec)
        ev = _event(rec, data)
        if ev == "refused":
            refusals.append(str(data.get("reason", "unspecified")))
        if ev in ("abort", "error", "estop"):
            aborts.append(str(data.get("reason", ev)))
        if ev in ("moved", "move"):
            num_moves += 1
        if data.get("dry_run") is True:
            dry_run_seen = True
        if bool(data.get("enable_motion")) and bool(data.get("acknowledge")):
            real_motion_requested = True
    warnings = [f"run aborted: {a}" for a in aborts]
    return {
        "available": True,
        "refusals": refusals,
        "aborts": aborts,
        "num_move_commands": num_moves,
        "dry_run": dry_run_seen,
        "real_motion_requested": real_motion_requested,
        "warnings": warnings,
    }


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def eval_run(logs: Any, out: Optional[str] = None, *, settle_tol: float = 20.0,
             confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """Compute metric families from ``logs`` and (optionally) write a report.

    Parameters
    ----------
    logs:
        A motion-log ``.jsonl`` path, an episode directory, a ``.json`` file, or
        an in-memory list/dict of records.
    out:
        If given, the rendered Markdown report is written here.
    settle_tol:
        Position tolerance (raw servo units) for the control-settle / reach gate.
    confidence_threshold:
        Keypoint-confidence pass threshold for the perception family.

    Returns
    -------
    A dict with one entry per family (each carrying ``available``), a flat
    ``warnings`` list, the source, and the rendered ``markdown``.
    """
    records = _to_records(logs)
    control = _control_family(records, settle_tol)
    perception = _perception_family(records, confidence_threshold)
    calibration = _calibration_family(records)
    fold_fam = _fold_family(records)
    safety = _safety_family(records)

    warnings: List[str] = []
    for fam in (control, perception, calibration, fold_fam, safety):
        warnings.extend(fam.get("warnings", []) if isinstance(fam, dict) else [])

    metrics: Dict[str, Any] = {
        "source": logs if isinstance(logs, str) else "<in-memory records>",
        "n_records": len(records),
        "control": control,
        "perception": perception,
        "calibration": calibration,
        "fold": fold_fam,
        "safety": safety,
        "warnings": warnings,
    }
    md = report_markdown(metrics)
    metrics["markdown"] = md
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(md)
        metrics["report_path"] = os.path.abspath(out)
    return metrics


def _fmt(x: Any) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
        return f"{x:.3f}" if isinstance(x, float) else str(x)
    return str(x)


def report_markdown(metrics: Dict[str, Any]) -> str:
    """Render a metrics dict (from :func:`eval_run`) as a Markdown report."""
    lines: List[str] = ["# TeraFold Evaluation Report", ""]
    lines.append(f"- source: `{metrics.get('source', '?')}`")
    lines.append(f"- records analyzed: {metrics.get('n_records', 0)}")
    lines.append("")

    warnings = metrics.get("warnings", [])
    lines.append("## Warnings")
    if warnings:
        for w in warnings:
            lines.append(f"- WARNING: {w}")
    else:
        lines.append("- none")
    lines.append("")

    # ---- Control ----
    lines.append("## Control Metrics")
    control = metrics.get("control", {})
    if control.get("available"):
        lines.append(f"- servos analyzed: {control['n_servos']}  "
                     f"(settle tolerance {control['settle_tol']:.0f} units)")
        lines.append(f"- worst settle error: {_fmt(control.get('worst_settle_error'))}")
        lines.append(f"- any failure to reach: {control.get('any_failure_to_reach')}")
        lines.append("")
        lines.append("| servo | target | settle_err | overshoot | t_settle | reached | noise |")
        lines.append("|-------|--------|-----------|-----------|----------|---------|-------|")
        for r in control.get("per_servo", []):
            reached = "no" if r["failure_to_reach"] else "yes"
            lines.append(
                f"| {r['servo_id']} | {_fmt(r['target'])} | {_fmt(r['settle_error'])} | "
                f"{_fmt(r['overshoot'])} | {_fmt(r['time_to_settle'])} | {reached} | "
                f"{_fmt(r['readback_noise'])} |")
    else:
        lines.append(f"- not available: {control.get('note', 'no data')}")
    lines.append("")

    # ---- Perception ----
    lines.append("## Perception")
    perception = metrics.get("perception", {})
    if perception.get("available"):
        if perception.get("modes"):
            lines.append(f"- mode(s): {', '.join(perception['modes'])}")
        if "min_confidence" in perception:
            lines.append(f"- min confidence: {_fmt(perception['min_confidence'])}  "
                         f"(threshold {_fmt(perception['threshold'])})")
            lines.append(f"- confidence gate passed: {perception.get('passed')}")
        if perception.get("note"):
            lines.append(f"- note: {perception['note']}")
    else:
        lines.append(f"- not available: {perception.get('note', 'no data')}")
    lines.append("")

    # ---- Calibration ----
    lines.append("## Calibration")
    calibration = metrics.get("calibration", {})
    if calibration.get("available"):
        lines.append(f"- correspondence sets: {calibration['n']}")
        lines.append(f"- worst reprojection RMS: {_fmt(calibration['worst_rms'])}")
    else:
        lines.append(f"- not available: {calibration.get('note', 'no data')}")
    lines.append("")

    # ---- Fold ----
    lines.append("## Fold Quality")
    fold_fam = metrics.get("fold", {})
    if fold_fam.get("available"):
        for i, fld in enumerate(fold_fam.get("folds", [])):
            parts = [f"{k}={_fmt(v)}" for k, v in fld.items()]
            lines.append(f"- fold {i}: {', '.join(parts)}")
    else:
        lines.append(f"- not available: {fold_fam.get('note', 'no data')}")
    lines.append("")

    # ---- Safety ----
    lines.append("## Safety")
    safety = metrics.get("safety", {})
    if safety.get("available"):
        lines.append(f"- dry-run observed: {safety.get('dry_run')}")
        lines.append(f"- real-motion requested: {safety.get('real_motion_requested')}")
        lines.append(f"- servo move commands logged: {safety.get('num_move_commands')}")
        refusals = safety.get("refusals", [])
        lines.append(f"- safety-gate refusals: {len(refusals)}"
                     + (f" ({', '.join(refusals)})" if refusals else ""))
        if safety.get("aborts"):
            lines.append(f"- aborts: {', '.join(safety['aborts'])}")
    else:
        lines.append("- not available")
    lines.append("")

    return "\n".join(lines)
