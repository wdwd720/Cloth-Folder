"""Adapter for a custom bus-servo arm on a Waveshare Bus Servo Adapter (A).

SAFETY FIRST. The exact bus-servo serial protocol for this arm is **not
confirmed** in the repo, and inventing packet bytes for an unknown servo bus is
dangerous. So this adapter:

* does only **read-only / passive** serial work by default (list ports, open the
  port, read USB descriptors to identify the adapter) — it never writes bytes
  that could move a servo;
* gates EVERY position read, ID scan, position write, and torque command behind a
  ``protocol_confirmed`` flag. Until a *verified* command backend is wired, all of
  those refuse with:  "Waveshare bus servo protocol not confirmed; refusing to
  move."
* emergency-stop always closes the port and prints how to cut power by hand
  (torque-off bytes are only sent when a verified backend exists).

To enable motion later, identify the servos + protocol (vendor SDK/datasheet) and
pass a ``command_backend`` object implementing ``write_position`` /
``read_positions`` / ``torque_off`` — only then does ``protocol_confirmed`` become
True. There is intentionally NO fabricated protocol here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "WaveshareBusServoAdapter",
    "WaveshareBusServoError",
    "PROTOCOL_REFUSAL",
    "PROTOCOL_NEXT_STEPS",
]

PROTOCOL_REFUSAL = "Waveshare bus servo protocol not confirmed; refusing to move."
PROTOCOL_NEXT_STEPS = [
    "Identify the servos (Feetech STS/SCS? Waveshare ST? other) and the bus voltage.",
    "Confirm the serial protocol from the vendor SDK / datasheet (packet format, baud).",
    "Add a VERIFIED command backend (write_position / read_positions / torque_off).",
    "Pass it to WaveshareBusServoAdapter(command_backend=...) — only then are "
    "reads/moves enabled. TeraFold will NOT guess packet bytes.",
]

_ADAPTER_HINTS = (
    "wave", "waveshare", "bus servo", "ftdi", "ft232", "ch340", "ch343",
    "cp210", "usb-serial", "usb single serial", "usb2.0-serial", "silicon labs",
)


class WaveshareBusServoError(Exception):
    """Raised when a motion/read command is refused (unconfirmed protocol)."""


class WaveshareBusServoAdapter:
    """Safe, read-only-by-default shell for the Waveshare bus-servo arm."""

    def __init__(
        self,
        port: str = "auto",
        baudrate: int = 1000000,
        baud_candidates: Optional[List[int]] = None,
        dof: int = 7,
        joint_names: Optional[List[str]] = None,
        dry_run: bool = True,
        logger: object = None,
        command_backend: object = None,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.baud_candidates = list(baud_candidates or [baudrate])
        self.dof = int(dof)
        self.joint_names = list(joint_names or [])
        self.dry_run = bool(dry_run)
        self.logger = logger
        # A verified backend is the ONLY thing that confirms the protocol.
        self._backend = command_backend
        self._serial = None

    # -- properties -----------------------------------------------------
    @property
    def protocol_confirmed(self) -> bool:
        """True only when a verified command backend is wired (never guessed)."""
        return self._backend is not None

    @property
    def supports_motion(self) -> bool:
        return self.protocol_confirmed

    # -- serial (read-only / passive) -----------------------------------
    def list_ports(self) -> Dict[str, Any]:
        """List candidate serial ports. NEVER writes bytes."""
        from terafold.robot.discovery import scan_serial_ports

        return scan_serial_ports(probe_readonly=False)

    def resolve_port(self) -> Optional[str]:
        if self.port and self.port != "auto":
            return self.port
        report = self.list_ports()
        likely = [p for p in report.get("ports", []) if p.get("likely_robot")]
        if likely:
            return likely[0]["device"]
        ports = report.get("ports", [])
        return ports[0]["device"] if ports else None

    def identify(self) -> Dict[str, Any]:
        """Identify the USB-serial adapter from descriptors. Sends NO bytes."""
        device = self.resolve_port()
        info: Dict[str, Any] = {
            "device": device,
            "likely_adapter": False,
            "confidence": 0.0,
            "vid": None, "pid": None, "manufacturer": None, "product": None,
            "description": None,
            "protocol_confirmed": self.protocol_confirmed,
            "note": ("USB descriptors only; the exact adapter model and the bus-servo "
                     "protocol are NOT confirmed. Motion stays disabled."),
        }
        try:
            from serial.tools import list_ports as _lp  # type: ignore

            for p in _lp.comports():
                if device and p.device != device:
                    continue
                blob = " ".join(str(x) for x in (p.description, p.manufacturer,
                                                 p.product, p.interface) if x).lower()
                info.update({
                    "device": p.device,
                    "vid": getattr(p, "vid", None), "pid": getattr(p, "pid", None),
                    "manufacturer": p.manufacturer, "product": p.product,
                    "description": p.description,
                })
                if any(h in blob for h in _ADAPTER_HINTS):
                    info["likely_adapter"] = True
                    info["confidence"] = 0.6
                break
        except Exception as exc:
            info["note"] = (f"pyserial unavailable ({exc}); install with "
                            "pip install -e \".[robot]\" for richer identification.")
        return info

    def open(self) -> Dict[str, Any]:
        """Open the serial port read-only (no bytes written). For probing only."""
        device = self.resolve_port()
        if device is None:
            return {"ok": False, "reason": "no serial port found (plug in the USB-C cable)."}
        try:
            import serial  # type: ignore

            self._serial = serial.Serial(device, self.baudrate, timeout=0.2,
                                         write_timeout=0.2, rtscts=False, dsrdtr=False)
            return {"ok": True, "device": device, "baudrate": self.baudrate}
        except Exception as exc:
            self._serial = None
            return {"ok": False, "device": device,
                    "reason": f"could not open port ({exc}). Install pyserial: "
                              "pip install -e \".[robot]\"; check permissions/driver."}

    def is_open(self) -> bool:
        return self._serial is not None and getattr(self._serial, "is_open", False)

    def close(self) -> None:
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass
        self._serial = None

    # -- protocol-gated reads -------------------------------------------
    def read_positions(self) -> Dict[str, Any]:
        """Read all servo positions — refused unless the protocol is confirmed."""
        if not self.protocol_confirmed:
            return {"supported": False, "reason": PROTOCOL_REFUSAL,
                    "next_steps": list(PROTOCOL_NEXT_STEPS)}
        return {"supported": True, "positions": self._backend.read_positions()}

    def read_position(self, servo_id: int) -> Optional[float]:
        r = self.read_positions()
        if not r.get("supported"):
            return None
        return r["positions"].get(servo_id)

    def scan_servo_ids(self, id_min: int = 1, id_max: int = 30) -> Dict[str, Any]:
        """Discover servo IDs — refused unless the protocol is confirmed."""
        if not self.protocol_confirmed:
            return {"supported": False, "reason": PROTOCOL_REFUSAL,
                    "next_steps": list(PROTOCOL_NEXT_STEPS)}
        ids = self._backend.scan_servo_ids(id_min, id_max)
        return {"supported": True, "servo_ids": list(ids)}

    # -- protocol-gated writes (also require the motion gate upstream) ---
    def torque_enable(self, on: bool, servo_id: Optional[int] = None) -> None:
        if not self.protocol_confirmed:
            raise WaveshareBusServoError(PROTOCOL_REFUSAL)
        self._backend.torque_enable(bool(on), servo_id)

    def write_position(self, servo_id: int, position_deg: float, speed: str = "very_slow") -> None:
        if not self.protocol_confirmed:
            raise WaveshareBusServoError(PROTOCOL_REFUSAL)
        self._backend.write_position(int(servo_id), float(position_deg), str(speed))

    # -- always-safe emergency stop -------------------------------------
    def emergency_stop(self) -> Dict[str, Any]:
        """Disable torque if a verified backend exists; always close the port.

        Returns a result dict and ALWAYS includes manual power-cut instructions —
        the ultimate stop is to remove DC power.
        """
        result: Dict[str, Any] = {
            "torque_off_attempted": False,
            "torque_off_ok": False,
            "port_closed": False,
            "protocol_confirmed": self.protocol_confirmed,
            "manual_instructions": [
                "CUT POWER NOW: switch off / unplug the DC 9-12.6V supply to the arm.",
                "You may also unplug the USB-C serial cable.",
                "Keep hands clear of the joints until power is removed.",
            ],
        }
        if self.protocol_confirmed and hasattr(self._backend, "torque_off"):
            result["torque_off_attempted"] = True
            try:
                self._backend.torque_off()
                result["torque_off_ok"] = True
            except Exception as exc:
                result["torque_off_error"] = str(exc)
        self.close()
        result["port_closed"] = True
        if self.logger is not None:
            try:
                self.logger.log("emergency_stop", result)
            except Exception:
                pass
        return result
