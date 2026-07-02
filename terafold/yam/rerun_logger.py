"""Rerun logging scaffold for the YAM shadow stack — optional, never crashes.

If ``rerun-sdk`` is installed, :class:`YamRerunLogger` streams the three camera
images, robot state, predicted/actual actions, task text and safety status to a
``.rrd`` file (open it later with ``rerun <file>.rrd``). If it is NOT installed
the logger degrades to a silent no-op that remembers an install hint — callers
always still write their JSON outputs, so nothing is lost without Rerun.

Every individual log call is wrapped defensively: a Rerun API change downgrades
logging to a warning, never a crash in the shadow pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

__all__ = ["RERUN_INSTALL_HINT", "YamRerunLogger", "have_rerun"]

RERUN_INSTALL_HINT = (
    "Rerun not installed — JSON outputs are still saved. "
    "For live/replayable visual logs:  pip install rerun-sdk   "
    "(or: pip install -e '.[rerun]')"
)


def have_rerun() -> bool:
    try:
        import rerun  # noqa: F401

        return True
    except Exception:
        return False


class YamRerunLogger:
    """Best-effort Rerun sink. ``available`` tells you whether it is live."""

    def __init__(
        self,
        out_dir: Optional[str] = None,
        app_id: str = "terafold_yam_shadow",
        filename: str = "shadow.rrd",
        enabled: bool = True,
    ) -> None:
        self.available = False
        self.rrd_path: Optional[str] = None
        self.install_hint = RERUN_INSTALL_HINT
        self._rr = None
        self._warned = False
        if not enabled:
            return
        try:
            import rerun as rr
        except Exception:
            return
        try:
            rr.init(app_id, spawn=False)
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                self.rrd_path = str(Path(out_dir) / filename)
                rr.save(self.rrd_path)
            self._rr = rr
            self.available = True
        except Exception:
            self._rr = None
            self.available = False

    # -- internal ----------------------------------------------------------
    def _safe(self, thunk) -> None:
        """Run a zero-arg callable; API drift must never break the shadow pipeline.

        Callers pass a *thunk* (not fn+args) so that even the ``rr.X`` attribute
        lookups happen inside the try block.
        """
        try:
            thunk()
        except Exception as e:
            if not self._warned:
                self._warned = True
                print(f"[rerun] logging degraded ({type(e).__name__}: {e}); continuing without it")

    def _log_vector(self, path: str, vec: Sequence[float]) -> None:
        rr = self._rr
        vals = [float(v) for v in vec]
        # rr.Scalars is the modern batch API; fall back to per-slot Scalar.
        if hasattr(rr, "Scalars"):
            self._safe(lambda: rr.log(path, rr.Scalars(vals)))
        else:  # pragma: no cover - depends on installed rerun version
            for i, v in enumerate(vals):
                self._safe(lambda i=i, v=v: rr.log(f"{path}/{i}", rr.Scalar(v)))

    # -- public API ---------------------------------------------------------
    def log_frame(
        self,
        step: int,
        images: Optional[Dict[str, Any]] = None,
        robot_state: Optional[Dict[str, Any]] = None,
        predicted_action: Optional[Sequence[float]] = None,
        actual_action: Optional[Sequence[float]] = None,
        task: Optional[str] = None,
        safety_status: Optional[str] = None,
    ) -> None:
        """Log one shadow-policy frame. Silent no-op when Rerun is unavailable."""
        if not self.available:
            return
        rr = self._rr
        self._safe(lambda: rr.set_time_sequence("frame", int(step)))
        for cam, img in (images or {}).items():
            if img is not None:
                self._safe(lambda cam=cam, img=img: rr.log(f"cameras/{cam}", rr.Image(img)))
        if robot_state is not None:
            vec = robot_state.get("state_vector") if isinstance(robot_state, dict) else None
            if vec is not None:
                self._log_vector("robot/state", vec)
            else:
                self._safe(lambda: rr.log("robot/state_raw", rr.TextLog(str(robot_state))))
        if predicted_action is not None:
            self._log_vector("action/predicted", predicted_action)
        if actual_action is not None:
            self._log_vector("action/actual", actual_action)
        if task is not None:
            self._safe(lambda: rr.log("task", rr.TextLog(task)))
        if safety_status is not None:
            self._safe(lambda: rr.log("safety/status", rr.TextLog(safety_status)))

    def note(self) -> str:
        """Human-readable one-liner about where logs went (or why they didn't)."""
        if self.available:
            return f"rerun: logging to {self.rrd_path or '(in-memory viewer stream)'}"
        return self.install_hint
