"""Lightweight command logger for robot actions and lifecycle events.

Every adapter logs the commands it *intends* to send (especially in dry-run) so
that a run is auditable and reproducible. The log is plain JSONL when a path is
given, and/or echoed to stdout. This is intentionally dependency-free (no torch,
no pandas) — it only touches the JSON helpers in :mod:`terafold.data`.
"""

from __future__ import annotations

import time
from typing import Optional

from terafold.data.episode_schema import append_jsonl
from terafold.robot.base import RobotAction, RobotObservation

__all__ = ["CommandLogger"]


class CommandLogger:
    """Append-only logger for robot commands and events.

    Parameters
    ----------
    path:
        Destination JSONL file. If ``None`` nothing is persisted (echo only).
    echo:
        Print each record to stdout as well.
    """

    def __init__(self, path: Optional[str] = None, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self._closed = False
        self._count = 0

    def _emit(self, record: dict) -> None:
        if self._closed:
            return
        record.setdefault("timestamp", time.time())
        record["seq"] = self._count
        self._count += 1
        if self.path:
            append_jsonl(self.path, record)
        if self.echo:
            print(f"[cmd] {record}")

    def log_action(
        self,
        action: RobotAction,
        observation: Optional[RobotObservation] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Log a commanded action and (optionally) the resulting observation."""
        record = {
            "event": "action",
            "action": action.to_dict() if action is not None else None,
            "observation": observation.to_dict() if observation is not None else None,
        }
        if extra:
            record["extra"] = extra
        self._emit(record)

    def log(self, event: str, data: Optional[dict] = None) -> None:
        """Log an arbitrary lifecycle event (connect, disconnect, stop, ...)."""
        record = {"event": event}
        if data:
            record["data"] = data
        self._emit(record)

    def close(self) -> None:
        """Flush a final marker and mark the logger closed."""
        if self._closed:
            return
        self._emit({"event": "close", "total": self._count})
        self._closed = True
