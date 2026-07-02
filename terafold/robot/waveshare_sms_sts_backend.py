"""Verified Waveshare SMS/STS bus-servo backend (scservo_sdk).

This is the *confirmed* protocol backend (sms_sts @ 1,000,000 baud). It uses the
vendor ``scservo_sdk`` (bundled under ``vendor/waveshare/STServo_Python/
stservo-env``) and the official calls:

* motion:  ``packetHandler.WritePosEx(id, target_position, speed, acc)``
* read:    ``packetHandler.ReadPosSpeed(id)``
* ping:    ``packetHandler.ping(id)``

``protocol_confirmed`` becomes True ONLY when the port opens at 1,000,000 baud,
at least one configured servo pings, and a position read succeeds. If the SDK or
the device is missing, everything fails *gracefully* (no exceptions leak; motion
is simply refused). Units are raw servo counts (0..4095), NOT degrees.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from terafold.robot.waveshare_bus_servo import PROTOCOL_REFUSAL, WaveshareBusServoError

__all__ = ["WaveshareSmsStsBackend", "load_scservo_sdk", "sdk_search_paths"]

SMS_STS_TORQUE_ENABLE = 40  # control-table address (from the vendor sms_sts.py)


def sdk_search_paths() -> List[str]:
    """Candidate dirs that contain a ``scservo_sdk`` package."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    paths = []
    env = os.environ.get("TERAFOLD_SCSERVO_SDK")
    if env:
        paths.append(env)
    paths.append(os.path.join(repo, "vendor", "waveshare", "STServo_Python", "stservo-env"))
    paths.append(os.path.join(repo, "vendor", "waveshare", "STServo_Python"))
    return paths


def load_scservo_sdk():
    """Import ``scservo_sdk`` (already installed, or from the vendor path). None if absent."""
    try:
        import scservo_sdk  # type: ignore

        return scservo_sdk
    except Exception:
        pass
    for p in sdk_search_paths():
        if p and os.path.isdir(os.path.join(p, "scservo_sdk")):
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                import scservo_sdk  # type: ignore

                return scservo_sdk
            except Exception:
                continue
    return None


class WaveshareSmsStsBackend:
    """Confirmed sms_sts backend. Read-safe; writes gated on protocol_confirmed."""

    def __init__(
        self,
        port: str,
        baudrate: int = 1000000,
        active_ids: Optional[List[int]] = None,
        default_speed: int = 300,
        default_acc: int = 20,
        safe_min_units: int = 400,
        safe_max_units: int = 3700,
        logger: object = None,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.active_ids = list(active_ids or [1, 2, 5, 6])
        self.default_speed = int(default_speed)
        self.default_acc = int(default_acc)
        self.safe_min_units = int(safe_min_units)
        self.safe_max_units = int(safe_max_units)
        self.logger = logger
        self._sdk = None
        self._port_handler = None
        self._packet = None
        self._confirmed = False

    # -- availability ---------------------------------------------------
    @property
    def sdk_available(self) -> bool:
        return load_scservo_sdk() is not None

    @property
    def protocol_confirmed(self) -> bool:
        return self._confirmed

    # -- lifecycle ------------------------------------------------------
    def open(self) -> Dict[str, Any]:
        """Open the port at the configured baud. No motion. Returns ok/reason."""
        sdk = load_scservo_sdk()
        if sdk is None:
            return {"ok": False, "reason": ("scservo_sdk not importable (need pyserial + the "
                                            "vendor SDK). pip install pyserial; SDK lives under "
                                            "vendor/waveshare/STServo_Python/stservo-env.")}
        self._sdk = sdk
        try:
            self._port_handler = sdk.PortHandler(self.port)
            if not self._port_handler.openPort():
                return {"ok": False, "reason": f"could not open port {self.port}"}
            if not self._port_handler.setBaudRate(self.baudrate):
                return {"ok": False, "reason": f"could not set baudrate {self.baudrate}"}
            self._packet = sdk.sms_sts(self._port_handler)
            return {"ok": True, "port": self.port, "baudrate": self.baudrate}
        except Exception as exc:
            return {"ok": False, "reason": f"open failed: {exc}"}

    def close(self) -> None:
        try:
            if self._port_handler is not None:
                self._port_handler.closePort()
        except Exception:
            pass
        self._port_handler = None
        self._packet = None
        self._confirmed = False

    # -- read-safe ------------------------------------------------------
    def ping(self, ids: Optional[List[int]] = None) -> Dict[int, bool]:
        ids = ids or self.active_ids
        out: Dict[int, bool] = {}
        if self._packet is None:
            return {i: False for i in ids}
        comm_ok = getattr(self._sdk, "COMM_SUCCESS", 0)
        for i in ids:
            try:
                _model, comm, _err = self._packet.ping(i)
                out[i] = (comm == comm_ok)
            except Exception:
                out[i] = False
        return out

    def read_position(self, servo_id: int) -> Optional[int]:
        if self._packet is None:
            return None
        comm_ok = getattr(self._sdk, "COMM_SUCCESS", 0)
        try:
            pos, _speed, comm, _err = self._packet.ReadPosSpeed(servo_id)
            return int(pos) if comm == comm_ok else None
        except Exception:
            return None

    def read_pos_speed(self, servo_id: int):
        """Return ``(position, speed)`` raw units, or ``None`` if the read failed.

        This is the named interface from the SDK (``ReadPosSpeed``); it exposes the
        speed too, which :mod:`characterization` uses to detect 'settled'.
        """
        if self._packet is None:
            return None
        comm_ok = getattr(self._sdk, "COMM_SUCCESS", 0)
        try:
            pos, speed, comm, _err = self._packet.ReadPosSpeed(servo_id)
            return (int(pos), int(speed)) if comm == comm_ok else None
        except Exception:
            return None

    def read_all_positions(self, ids: Optional[List[int]] = None) -> Dict[int, Optional[int]]:
        ids = ids or self.active_ids
        return {i: self.read_position(i) for i in ids}

    # Named aliases matching the required backend interface --------------
    def read_all(self, ids: Optional[List[int]] = None) -> Dict[int, Optional[int]]:
        """Alias of :meth:`read_all_positions` (the spec's interface name)."""
        return self.read_all_positions(ids)

    def ping_many(self, ids: Optional[List[int]] = None) -> Dict[int, bool]:
        """Alias of :meth:`ping` over many IDs."""
        return self.ping(ids)

    def available(self) -> bool:
        """True if the vendor SDK is importable (a backend can be attempted)."""
        return self.sdk_available

    def probe_protocol(self, ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Read-only protocol probe. Alias of :meth:`confirm` (never writes)."""
        if ids is not None:
            self.active_ids = list(ids)
        return self.confirm()

    # -- protocol-gated writes -----------------------------------------
    def write_pos_ex(self, servo_id: int, target: int, speed: Optional[int] = None,
                     acc: Optional[int] = None) -> bool:
        """Alias of :meth:`write_position` (the SDK / spec interface name)."""
        return self.write_position(servo_id, target, speed=speed, acc=acc)

    def write_position(self, servo_id: int, target: int, speed: Optional[int] = None,
                       acc: Optional[int] = None) -> bool:
        if not self._confirmed:
            raise WaveshareBusServoError(PROTOCOL_REFUSAL)
        target = int(target)
        if not (self.safe_min_units <= target <= self.safe_max_units):
            raise WaveshareBusServoError(
                f"target {target} outside safe range [{self.safe_min_units}, "
                f"{self.safe_max_units}] for servo {servo_id}; refusing.")
        speed = self.default_speed if speed is None else int(speed)
        acc = self.default_acc if acc is None else int(acc)
        comm_ok = getattr(self._sdk, "COMM_SUCCESS", 0)
        comm, _err = self._packet.WritePosEx(int(servo_id), target, speed, acc)
        ok = (comm == comm_ok)
        if self.logger is not None:
            try:
                self.logger.log("write_position",
                                {"id": servo_id, "target": target, "speed": speed,
                                 "acc": acc, "ok": ok})
            except Exception:
                pass
        return ok

    def torque_enable(self, servo_id: int, on: bool) -> bool:
        if not self._confirmed:
            raise WaveshareBusServoError(PROTOCOL_REFUSAL)
        try:
            self._packet.write1ByteTxRx(int(servo_id), SMS_STS_TORQUE_ENABLE, 1 if on else 0)
            return True
        except Exception:
            return False

    def torque_off(self, ids: Optional[List[int]] = None) -> None:
        """Best-effort torque disable on the confirmed servos (safe stop)."""
        if self._packet is None:
            return
        for i in (ids or self.active_ids):
            try:
                self._packet.write1ByteTxRx(int(i), SMS_STS_TORQUE_ENABLE, 0)
            except Exception:
                pass

    # -- confirmation ---------------------------------------------------
    def confirm(self) -> Dict[str, Any]:
        """Confirm the protocol: open + baud + >=1 ping + a successful read."""
        report: Dict[str, Any] = {"sdk_available": self.sdk_available, "opened": False,
                                  "baudrate": self.baudrate, "pings": {}, "read_ok": False,
                                  "confirmed": False}
        opened = self.open()
        report["open_reason"] = opened.get("reason")
        if not opened.get("ok"):
            self._confirmed = False
            return report
        report["opened"] = True
        pings = self.ping(self.active_ids)
        report["pings"] = pings
        responders = [i for i, ok in pings.items() if ok]
        report["responders"] = responders
        if responders:
            pos = self.read_position(responders[0])
            report["read_ok"] = pos is not None
            report["read_example"] = {responders[0]: pos}
        self._confirmed = bool(responders and report["read_ok"]
                               and opened.get("ok") and self.baudrate == 1000000)
        report["confirmed"] = self._confirmed
        if self.logger is not None:
            try:
                self.logger.log("protocol_confirm", report)
            except Exception:
                pass
        return report

    # -- safe stop / close ----------------------------------------------
    def safe_stop(self, ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Best-effort safe stop: disable torque on the confirmed servos.

        Does NOT close the port (use :meth:`close_safely` for that). Returns a
        small report so callers can log what was attempted.
        """
        attempted = self._packet is not None and self._confirmed
        if attempted:
            self.torque_off(ids)
        return {"torque_off_attempted": attempted, "ids": list(ids or self.active_ids)}

    def close_safely(self, ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Safe-stop then close the port. Always closes, even on error."""
        report = {"torque_off_attempted": False, "port_closed": False}
        try:
            report.update(self.safe_stop(ids))
        finally:
            self.close()
            report["port_closed"] = True
        return report

    # -- emergency stop -------------------------------------------------
    def emergency_stop(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"torque_off_attempted": False, "port_closed": False,
                                  "manual_instructions": [
                                      "CUT POWER NOW: switch off / unplug the DC 9-12.6V supply.",
                                      "Unplug the USB-C serial cable.",
                                      "Keep hands clear until power is removed.",
                                  ]}
        if self._packet is not None:
            result["torque_off_attempted"] = True
            self.torque_off()
        self.close()
        result["port_closed"] = True
        return result
