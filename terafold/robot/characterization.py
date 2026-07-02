"""Per-servo characterization: readback noise, deadband, settle, backlash, repeatability.

The metric functions are pure numpy and operate on *traces* (lists/arrays of raw
servo units), so they are unit-testable without any hardware. They quantify how a
single bus servo behaves:

* :func:`readback_stats`        - encoder readback noise at rest.
* :func:`deadband_from_sweep`   - the smallest commanded delta that produces motion.
* :func:`settle_metrics`        - steady-state error, overshoot, time-to-settle.
* :func:`backlash`              - hysteresis between approaching from below vs above.
* :func:`repeatability`         - spread of the resting position over repeated returns.

:func:`run_characterization` orchestrates these on real hardware, but defaults to a
DRY-RUN that only describes the planned tests. Real motion requires BOTH the
two-flag gate (``enable_motion`` AND ``acknowledge``) AND a backend whose protocol
is confirmed; otherwise it returns a refusal dict and moves nothing. Every step is
conservative (tiny step within the safe range, returns home immediately). Plots are
drawn only when matplotlib is importable, and skipped gracefully otherwise.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from terafold.data.episode_schema import write_json
from terafold.robot.arm_config import load_arm_config
from terafold.robot.safety import SafetyError, require_motion_enabled

__all__ = [
    "readback_stats",
    "deadband_from_sweep",
    "settle_metrics",
    "backlash",
    "repeatability",
    "run_characterization",
]


# ----------------------------------------------------------------------
# Metric functions (pure numpy, hardware-free).
# ----------------------------------------------------------------------


def readback_stats(samples: Any) -> Dict[str, float]:
    """Mean/std/min/max of resting encoder readback samples (raw units)."""
    arr = np.asarray(list(samples), dtype=float).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def deadband_from_sweep(deltas: Any, moved_flags: Any) -> Optional[float]:
    """First (smallest) commanded delta that actually produced motion.

    ``deltas`` is a sequence of commanded relative moves (ascending magnitude) and
    ``moved_flags`` the parallel booleans indicating whether the servo measurably
    moved at that delta. Returns the first delta whose flag is truthy (the effective
    deadband), or ``None`` if nothing ever moved.
    """
    for d, moved in zip(list(deltas), list(moved_flags)):
        if moved:
            return float(d)
    return None


def settle_metrics(trace: Any, target: float, tol: Optional[float] = None) -> Dict[str, Any]:
    """Steady-state error, overshoot, and time-to-settle for a step response.

    ``trace`` is the time-ordered position readback after commanding ``target``.

    * ``settle_error``    - ``abs(final - target)``.
    * ``overshoot``       - peak excursion PAST ``target`` (in the approach
      direction), as a non-negative magnitude.
    * ``time_to_settle``  - the first sample index after which every subsequent
      sample stays within ``tol`` of ``target``. ``tol`` defaults to a 5% band of
      the approach span (at least 1 unit). Equals ``len(trace)`` if it never settles.
    """
    arr = np.asarray(list(trace), dtype=float).reshape(-1)
    target = float(target)
    if arr.size == 0:
        return {"settle_error": 0.0, "overshoot": 0.0, "time_to_settle": 0}
    start = float(arr[0])
    span = abs(target - start)
    if tol is None:
        tol = max(1.0, 0.05 * span)
    final = float(arr[-1])
    settle_error = abs(final - target)
    if target >= start:
        overshoot = max(0.0, float(arr.max()) - target)
    else:
        overshoot = max(0.0, target - float(arr.min()))
    within = np.abs(arr - target) <= float(tol)
    out_of_band = np.where(~within)[0]
    if out_of_band.size == 0:
        tts = 0
    else:
        last_bad = int(out_of_band[-1])
        tts = last_bad + 1  # may equal arr.size -> never settled within the window
    return {
        "settle_error": float(settle_error),
        "overshoot": float(overshoot),
        "time_to_settle": int(tts),
    }


def backlash(pos_from_low: Any, pos_from_high: Any) -> float:
    """Hysteresis: ``|mean(approach-from-below) - mean(approach-from-above)|``.

    Accepts scalars or sequences of measurements of the SAME commanded target
    reached from below vs from above.
    """
    lo = np.asarray(pos_from_low, dtype=float).reshape(-1)
    hi = np.asarray(pos_from_high, dtype=float).reshape(-1)
    if lo.size == 0 or hi.size == 0:
        return 0.0
    return float(abs(lo.mean() - hi.mean()))


def repeatability(returns: Any) -> float:
    """Spread (std, raw units) of the resting position over repeated returns."""
    arr = np.asarray(list(returns), dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr.std())


# ----------------------------------------------------------------------
# Orchestration.
# ----------------------------------------------------------------------


def _emit(log: Callable[[str], None], msg: str) -> None:
    try:
        log(msg)
    except Exception:  # pragma: no cover - defensive
        pass


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
            return int(pos), 0
    return None, None


def _build_plan(
    robot: str,
    ids: List[int],
    modes: List[str],
    out: str,
    step_units: int,
    n_samples: int,
    lo: int,
    hi: int,
) -> Dict[str, Any]:
    """Describe the (conservative) tests that real mode WOULD run. No motion."""
    lines: List[str] = []
    for sid in ids:
        if "readback" in modes:
            lines.append(f"servo {sid}: read position {n_samples}x at rest "
                         f"(noise / readback_stats) - NO motion.")
        if "step" in modes:
            lines.append(f"servo {sid}: ONE +{step_units}-unit step within safe "
                         f"range [{lo}, {hi}], read the settle trace, then return home.")
    return {
        "robot": robot,
        "ids": list(ids),
        "modes": list(modes),
        "out": out,
        "step_units": int(step_units),
        "readback_samples": int(n_samples),
        "safe_range": [int(lo), int(hi)],
        "moves_hardware": False,
        "plan_lines": lines,
        "note": ("Conservative per-servo characterization. Real mode needs BOTH "
                 "--enable-motion and --i-understand-this-moves-hardware AND a "
                 "confirmed protocol; default is this dry-run."),
    }


def _maybe_plot(out: str, robot: str, sid: int, positions: List[int], target: int) -> Optional[str]:
    """Save a settle-trace PNG if matplotlib is importable; else ``None``."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    try:
        path = os.path.join(out, f"{robot}_servo{sid}_settle.png")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(range(len(positions)), positions, marker="o", label="position")
        ax.axhline(target, color="r", linestyle="--", label="target")
        ax.set_xlabel("read index")
        ax.set_ylabel("units")
        ax.set_title(f"servo {sid} settle")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - plotting is best-effort
        return None


def _write_trace_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = ["phase", "index", "commanded", "position", "speed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def run_characterization(
    robot: str,
    ids: List[int],
    out: str,
    port: Optional[str] = None,
    dry_run: bool = True,
    enable_motion: bool = False,
    acknowledge: bool = False,
    backend: Any = None,
    modes: Optional[List[str]] = None,
    log: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    step_units: int = 20,
    n_samples: int = 8,
    settle_reads: int = 6,
    read_interval_s: float = 0.05,
) -> Dict[str, Any]:
    """Characterize servos ``ids`` on ``robot``. DRY-RUN by default (plan only).

    In the default dry-run, nothing touches hardware: the planned conservative
    tests are described and a plan dict is returned. Real mode (``dry_run=False``)
    requires BOTH ``enable_motion`` and ``acknowledge`` AND a backend whose
    ``protocol_confirmed`` is True; otherwise a refusal dict is returned and nothing
    moves. When real, each servo gets a readback-noise pass and ONE tiny step
    (within the config's safe range, returning home immediately); metrics, per-servo
    CSV traces, a ``metrics.json`` and a ``summary.md`` are written under ``out``
    (plus PNG plots if matplotlib is available).
    """
    cfg = load_arm_config(robot)
    ids = [int(i) for i in ids]
    modes = list(modes) if modes else ["readback", "step"]
    safe = cfg.safe_position_units or [400, 3700]
    lo, hi = int(safe[0]), int(safe[1])
    # Keep the step conservative: never exceed the config's max sweep.
    step = int(min(int(step_units), int(cfg.max_sweep_units)))
    plan = _build_plan(robot, ids, modes, out, step, n_samples, lo, hi)

    # -- DRY-RUN (default): describe, move nothing -------------------------
    if dry_run:
        _emit(log, "[dry-run] servo characterization plan (NO hardware motion):")
        for line in plan["plan_lines"]:
            _emit(log, "  " + line)
        return {
            "status": "dry_run",
            "moved": False,
            "commands_sent": 0,
            "robot": robot,
            "ids": ids,
            "modes": modes,
            "out": out,
            "plan": plan,
            "wrote": [],
        }

    # -- REAL mode: two-flag gate ------------------------------------------
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as exc:
        return {
            "status": "refused",
            "moved": False,
            "commands_sent": 0,
            "refusal": str(exc),
            "robot": robot,
            "ids": ids,
            "plan": plan,
        }

    owns_backend = backend is None
    if backend is None:
        from terafold.robot.waveshare_sms_sts_backend import WaveshareSmsStsBackend

        backend = WaveshareSmsStsBackend(
            port=port or cfg.port,
            baudrate=cfg.baud,
            active_ids=ids,
            default_speed=cfg.default_speed_units,
            default_acc=cfg.default_acc_units,
            safe_min_units=lo,
            safe_max_units=hi,
        )
        probe = getattr(backend, "probe_protocol", None)
        if callable(probe):
            try:
                probe(ids)
            except Exception:  # pragma: no cover - defensive
                pass

    # -- protocol confirmation gate ----------------------------------------
    if not bool(getattr(backend, "protocol_confirmed", False)):
        if owns_backend:
            try:
                backend.close()
            except Exception:  # pragma: no cover - defensive
                pass
        return {
            "status": "refused",
            "moved": False,
            "commands_sent": 0,
            "refusal": ("protocol not confirmed; refusing real servo "
                        "characterization. Confirm the SMS/STS backend "
                        "(open + ping + read) first."),
            "robot": robot,
            "ids": ids,
            "plan": plan,
        }

    os.makedirs(out, exist_ok=True)
    speed = int(cfg.default_speed_units)
    acc = int(cfg.default_acc_units)
    commands_sent = 0
    metrics: Dict[str, Any] = {}
    traces: List[str] = []
    plots: List[str] = []

    for sid in ids:
        trace_rows: List[Dict[str, Any]] = []
        per: Dict[str, Any] = {}

        # 1) readback noise at rest (reads only).
        samples: List[int] = []
        if "readback" in modes:
            for k in range(n_samples):
                pos, spd = _read_pos_speed(backend, sid)
                if pos is not None:
                    samples.append(pos)
                trace_rows.append({"phase": "readback", "index": k, "commanded": None,
                                   "position": pos, "speed": spd})
                sleep_fn(read_interval_s)
            per["readback"] = readback_stats(samples)

        # 2) one conservative step, then return home.
        if "step" in modes:
            cur, _spd = _read_pos_speed(backend, sid)
            if cur is None:
                cur = (lo + hi) // 2
            target = max(lo, min(hi, int(cur) + step))
            positions: List[int] = []
            if target != cur:
                backend.write_position(sid, target, speed=speed, acc=acc)
                commands_sent += 1
                for k in range(settle_reads):
                    p, spd = _read_pos_speed(backend, sid)
                    positions.append(p if p is not None else target)
                    trace_rows.append({"phase": "step", "index": k, "commanded": target,
                                       "position": p, "speed": spd})
                    sleep_fn(read_interval_s)
                # Return home immediately.
                backend.write_position(sid, int(cur), speed=speed, acc=acc)
                commands_sent += 1
                rp, spd = _read_pos_speed(backend, sid)
                return_pos = rp if rp is not None else int(cur)
                trace_rows.append({"phase": "return", "index": 0, "commanded": int(cur),
                                   "position": rp, "speed": spd})
            else:
                return_pos = int(cur)
            per["step"] = {
                "start": int(cur),
                "target": int(target),
                "return_pos": int(return_pos),
                "settle": settle_metrics(positions, target) if positions else {
                    "settle_error": 0.0, "overshoot": 0.0, "time_to_settle": 0},
            }
            plot = _maybe_plot(out, robot, sid, positions, target)
            if plot:
                plots.append(plot)

        metrics[str(sid)] = per
        csv_path = os.path.join(out, f"{robot}_servo{sid}_trace.csv")
        _write_trace_csv(csv_path, trace_rows)
        traces.append(csv_path)
        _emit(log, f"[char] servo {sid} done -> {csv_path}")

    if owns_backend:
        try:
            backend.close()
        except Exception:  # pragma: no cover - defensive
            pass

    metrics_json = os.path.join(out, "metrics.json")
    summary_md = os.path.join(out, "summary.md")
    report = {
        "robot": robot,
        "ids": ids,
        "modes": modes,
        "step_units": step,
        "safe_range": [lo, hi],
        "commands_sent": commands_sent,
        "metrics": metrics,
        "traces": traces,
        "plots": plots,
        "note": ("Conservative real-hardware characterization: one tiny step per "
                 "servo within the safe range, returned home. Matplotlib plots are "
                 "optional."),
    }
    write_json(metrics_json, report)
    with open(summary_md, "w") as f:
        f.write(_summary_markdown(report))

    return {
        "status": "characterized",
        "moved": True,
        "commands_sent": commands_sent,
        "robot": robot,
        "ids": ids,
        "modes": modes,
        "out": out,
        "metrics": metrics,
        "metrics_json": metrics_json,
        "summary_md": summary_md,
        "traces": traces,
        "plots": plots,
    }


def _summary_markdown(report: Dict[str, Any]) -> str:
    """Render the ``summary.md`` for a real characterization run."""
    lines: List[str] = []
    lines.append("# TeraFold - Servo Characterization")
    lines.append("")
    lines.append(f"- robot: `{report['robot']}`")
    lines.append(f"- servos: {', '.join(str(i) for i in report['ids'])}")
    lines.append(f"- modes: {', '.join(report['modes'])}")
    lines.append(f"- step (units): {report['step_units']}   safe range: {report['safe_range']}")
    lines.append(f"- servo writes sent: {report['commands_sent']} (tiny step + return home each)")
    lines.append("")
    lines.append("## Per-servo metrics")
    lines.append("")
    for sid, per in report["metrics"].items():
        lines.append(f"### servo {sid}")
        rb = per.get("readback")
        if rb:
            lines.append(f"- readback: mean={rb['mean']:.1f} std={rb['std']:.2f} "
                         f"min={rb['min']:.0f} max={rb['max']:.0f} (n={rb['n']})")
        st = per.get("step")
        if st:
            s = st["settle"]
            lines.append(f"- step: {st['start']} -> {st['target']} (returned {st['return_pos']})")
            lines.append(f"- settle: error={s['settle_error']:.1f} "
                         f"overshoot={s['overshoot']:.1f} "
                         f"time_to_settle={s['time_to_settle']}")
        lines.append("")
    if report["plots"]:
        lines.append("## Plots")
        for p in report["plots"]:
            lines.append(f"- `{os.path.basename(p)}`")
        lines.append("")
    else:
        lines.append("_No plots (matplotlib not installed); CSV traces were written._")
        lines.append("")
    return "\n".join(lines) + "\n"
