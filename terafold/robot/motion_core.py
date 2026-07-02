"""Safety-first joint-space motion execution core (backend-agnostic).

This is the one place that actually drives a confirmed backend's
``write_position``. Everything here is built so that:

* a write happens ONLY when the caller passes ``gate_ok=True`` (the result of a
  :class:`~terafold.robot.safety_gate.MotionGate` check) AND the backend reports
  ``protocol_confirmed`` — never otherwise;
* dry-run / blocked requests fall through to a *refusal* that touches no servo;
* readback is used to decide when a joint has settled before the next step;
* returning to start and aborting are best-effort and swallow backend errors so a
  failure mid-sequence still tries to leave the arm safe.

No serial protocol is invented here; the actual packet format lives behind the
backend's ``write_position``/``read_pos_speed`` methods. Pure stdlib + the repo's
own modules — no numpy/torch import at module load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "MotionStep",
    "MotionPlan",
    "ReadbackMonitor",
    "safe_single_joint_move",
    "safe_multi_joint_move",
    "return_to_start_best_effort",
    "abort_with_reason",
    "STEP_SETTLE_S",
]

# Default per-step dwell (seconds) handed to an injected ``sleep_fn`` so a real
# servo has a moment to move before the next step / readback. Injected so tests
# never actually sleep.
STEP_SETTLE_S = 0.05


@dataclass
class MotionStep:
    """A single servo target: move ``servo_id`` to ``target_units`` (raw counts)."""

    servo_id: int
    target_units: int
    speed: int
    acc: int
    label: str = ""


@dataclass
class MotionPlan:
    """An ordered list of :class:`MotionStep` plus a human-readable note."""

    steps: List[MotionStep] = field(default_factory=list)
    note: str = ""


def _log(log: Any, event: str, data: Dict[str, Any]) -> None:
    """Emit to a :class:`~terafold.robot.command_logger.CommandLogger`-style logger.

    Accepts anything exposing ``.log(event, data)`` (the repo convention) or a
    plain ``callable``. Never raises — logging must not break a motion result.
    """
    if log is None:
        return
    try:
        if hasattr(log, "log"):
            log.log(event, data)
        elif callable(log):
            log(f"{event}: {data}")
    except Exception:
        pass


class ReadbackMonitor:
    """Decide whether a servo has reached its target and stopped moving.

    Uses ``backend.read_pos_speed(id) -> (pos, speed)`` when available, falling
    back to ``backend.read_position(id) -> pos`` (no speed -> speed gate skipped).
    A read that returns ``None`` (or a missing method) means "cannot confirm" and
    is treated as NOT settled.
    """

    def __init__(self, tol: int = 8, speed_tol: int = 20) -> None:
        self.tol = int(tol)
        self.speed_tol = int(speed_tol)

    @staticmethod
    def settled(backend: Any, servo_id: int, target: int, tol: int = 8, speed_tol: int = 20) -> bool:
        pos: Optional[int] = None
        speed: Optional[int] = None

        rps = getattr(backend, "read_pos_speed", None)
        if callable(rps):
            try:
                r = rps(servo_id)
            except Exception:
                r = None
            if r is not None:
                pos, speed = int(r[0]), int(r[1])

        if pos is None:
            rp = getattr(backend, "read_position", None)
            if callable(rp):
                try:
                    raw = rp(servo_id)
                except Exception:
                    raw = None
                if raw is not None:
                    pos = int(raw)

        if pos is None:
            return False
        pos_ok = abs(pos - int(target)) <= int(tol)
        speed_ok = True if speed is None else abs(speed) <= int(speed_tol)
        return bool(pos_ok and speed_ok)

    def is_settled(self, backend: Any, servo_id: int, target: int) -> bool:
        """Instance helper using this monitor's configured tolerances."""
        return self.settled(backend, servo_id, target, self.tol, self.speed_tol)


def _protocol_confirmed(backend: Any) -> bool:
    return bool(getattr(backend, "protocol_confirmed", False))


def safe_single_joint_move(
    backend: Any, step: MotionStep, gate_ok: bool, log: Any = None
) -> Dict[str, Any]:
    """Write ONE servo step iff the gate passed and the protocol is confirmed.

    Returns a result dict. When ``gate_ok`` is ``False`` (or the backend protocol
    is unconfirmed) NOTHING is written and ``wrote`` is ``False``. This is the
    hard floor: a refused step never reaches ``backend.write_position``.
    """
    base = {
        "servo_id": int(step.servo_id),
        "target_units": int(step.target_units),
        "speed": int(step.speed),
        "acc": int(step.acc),
        "label": step.label,
    }

    if not gate_ok:
        result = {**base, "ok": False, "wrote": False, "refused": True,
                  "reason": "safety gate not satisfied (dry-run or blocked); refusing to write."}
        _log(log, "refuse_single_joint_move", result)
        return result

    if not _protocol_confirmed(backend):
        result = {**base, "ok": False, "wrote": False, "refused": True,
                  "reason": "protocol not confirmed; refusing to write a servo."}
        _log(log, "refuse_single_joint_move", result)
        return result

    try:
        ok = bool(backend.write_position(step.servo_id, step.target_units,
                                         speed=step.speed, acc=step.acc))
        result = {**base, "ok": ok, "wrote": True, "refused": False}
    except Exception as exc:  # backend refused/erred — surface it, never crash
        result = {**base, "ok": False, "wrote": False, "refused": False, "error": str(exc)}
    _log(log, "single_joint_move", result)
    return result


def safe_multi_joint_move(
    backend: Any,
    plan: MotionPlan,
    gate_ok: bool,
    sleep_fn: Optional[Callable[[float], None]] = None,
    log: Any = None,
) -> Dict[str, Any]:
    """Execute a :class:`MotionPlan` step by step, dwelling + reading back each.

    Refuses the whole plan (no writes) when ``gate_ok`` is ``False``. Otherwise
    each step goes through :func:`safe_single_joint_move`; after a successful
    write the injected ``sleep_fn`` is called (a settle dwell) and the step's
    readback-settled status is recorded. A failed/erroring write aborts the rest.
    """
    if not gate_ok:
        result = {"ok": False, "wrote": False, "refused": True, "steps": 0,
                  "results": [], "note": plan.note,
                  "reason": "safety gate not satisfied (dry-run or blocked); refusing the plan."}
        _log(log, "refuse_multi_joint_move", result)
        return result

    monitor = ReadbackMonitor()
    results: List[Dict[str, Any]] = []
    wrote_any = False
    for step in plan.steps:
        r = safe_single_joint_move(backend, step, gate_ok=True, log=log)
        if r.get("wrote"):
            wrote_any = True
        if sleep_fn is not None and r.get("wrote"):
            try:
                sleep_fn(STEP_SETTLE_S)
            except Exception:
                pass
        r["settled"] = monitor.is_settled(backend, step.servo_id, step.target_units) \
            if r.get("wrote") else False
        results.append(r)
        if not r.get("ok"):
            break  # abort remaining steps on a failed/refused write

    ok = len(results) == len(plan.steps) and all(r.get("ok") for r in results)
    result = {"ok": ok, "wrote": wrote_any, "refused": False, "steps": len(results),
              "results": results, "note": plan.note}
    _log(log, "multi_joint_move", result)
    return result


def return_to_start_best_effort(
    backend: Any, start: Dict[int, int], speed: int, acc: int, log: Any = None
) -> Dict[str, Any]:
    """Best-effort write the servos back to ``start`` (per-servo, swallowing errors).

    Used after a sequence (or an abort) to leave the arm where it began. Requires
    a confirmed protocol; if unconfirmed it writes nothing and reports why. Any
    per-servo write error is captured rather than raised.
    """
    if not _protocol_confirmed(backend):
        result = {"ok": False, "wrote": False, "written": {}, "errors": {},
                  "reason": "protocol not confirmed; cannot return to start."}
        _log(log, "return_to_start", result)
        return result

    written: Dict[int, Dict[str, Any]] = {}
    errors: Dict[int, str] = {}
    for sid, target in start.items():
        try:
            ok = bool(backend.write_position(int(sid), int(target), speed=speed, acc=acc))
            written[int(sid)] = {"target": int(target), "ok": ok}
        except Exception as exc:
            errors[int(sid)] = str(exc)
    result = {"ok": not errors, "wrote": bool(written), "written": written, "errors": errors}
    _log(log, "return_to_start", result)
    return result


def abort_with_reason(reason: str, log: Any = None) -> Dict[str, Any]:
    """Build (and log) a uniform abort result. Writes nothing."""
    result = {"ok": False, "aborted": True, "wrote": False, "reason": str(reason)}
    _log(log, "abort", result)
    return result
