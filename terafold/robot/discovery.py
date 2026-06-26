"""Robot adapter discovery — list serial ports, NEVER send motor commands.

We don't know the exact arm yet, so this is read-only reconnaissance: enumerate
likely serial devices (``/dev/tty.usbserial-*``, ``/dev/tty.usbmodem-*``,
``/dev/cu.*`` on macOS) with whatever metadata is available, and print the steps
to identify the arm before any adapter is written. It writes nothing to any port
unless ``--probe-readonly`` is passed, and even then only reads.
"""

from __future__ import annotations

import glob
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

__all__ = ["SerialPortInfo", "scan_serial_ports", "robot_info_template_markdown", "NEXT_STEPS"]

NEXT_STEPS = [
    "Identify the robot brand and model (look on the arm / its box / the listing).",
    "Identify the controller board (e.g. Arduino, ESP32, dedicated servo driver, Feetech/Dynamixel bus).",
    "Find the serial baud rate and command protocol from the vendor docs.",
    "Test the vendor's own software FIRST — confirm the arm moves there before any custom code.",
    "Only THEN implement a TeraFold adapter that encodes the verified protocol.",
    "Until the protocol is verified, use GenericArmAdapter (dry-run) + `terafold export-trajectory`.",
]

_LIKELY_HINTS = ("usbserial", "usbmodem", "wch", "ch340", "cp210", "ftdi", "arduino", "acm")


@dataclass
class SerialPortInfo:
    device: str
    description: Optional[str] = None
    hwid: Optional[str] = None
    manufacturer: Optional[str] = None
    vid: Optional[int] = None
    pid: Optional[int] = None
    serial_number: Optional[str] = None
    likely_robot: bool = False
    probe: Optional[dict] = None  # only populated with --probe-readonly

    def to_dict(self) -> dict:
        return asdict(self)


def _looks_likely(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in _LIKELY_HINTS)


def scan_serial_ports(probe_readonly: bool = False, baudrate: int = 115200) -> Dict[str, Any]:
    """Enumerate serial ports. Returns a structured report; never writes to a port."""
    ports: List[SerialPortInfo] = []
    used_pyserial = False
    try:
        from serial.tools import list_ports  # type: ignore

        used_pyserial = True
        for p in list_ports.comports():
            info = SerialPortInfo(
                device=p.device,
                description=getattr(p, "description", None),
                hwid=getattr(p, "hwid", None),
                manufacturer=getattr(p, "manufacturer", None),
                vid=getattr(p, "vid", None),
                pid=getattr(p, "pid", None),
                serial_number=getattr(p, "serial_number", None),
            )
            info.likely_robot = _looks_likely(
                f"{info.device} {info.description} {info.hwid} {info.manufacturer}"
            )
            ports.append(info)
    except Exception:
        # numpy-free fallback: glob the common macOS/Linux device nodes.
        seen = set()
        for pattern in (
            "/dev/tty.usbserial-*", "/dev/tty.usbmodem-*", "/dev/cu.usbserial-*",
            "/dev/cu.usbmodem-*", "/dev/cu.*", "/dev/ttyUSB*", "/dev/ttyACM*",
        ):
            for dev in sorted(glob.glob(pattern)):
                if dev in seen:
                    continue
                seen.add(dev)
                ports.append(SerialPortInfo(device=dev, likely_robot=_looks_likely(dev)))

    if probe_readonly:
        for info in ports:
            info.probe = _probe_readonly(info.device, baudrate)

    return {
        "pyserial": used_pyserial,
        "num_ports": len(ports),
        "ports": [p.to_dict() for p in ports],
        "likely_ports": [p.device for p in ports if p.likely_robot],
        "next_steps": NEXT_STEPS,
        "note": (
            "Read-only scan. No motor commands were sent."
            + ("" if used_pyserial else "  (pyserial not installed; used device-node globbing. "
               "Install for richer metadata: pip install pyserial)")
        ),
    }


def _probe_readonly(device: str, baudrate: int) -> dict:
    """Open a port and READ only (no writes). Best-effort; opening may reset some boards."""
    try:
        import serial  # type: ignore
    except Exception:
        return {"status": "skipped", "reason": "pyserial not installed (pip install pyserial)"}
    try:
        # dsrdtr/rtscts off to reduce the chance of toggling a board reset line.
        with serial.Serial(device, baudrate, timeout=0.4, dsrdtr=False, rtscts=False) as s:
            data = s.read(64)  # READ ONLY — never write
        return {
            "status": "read_ok",
            "bytes_read": len(data),
            "sample_hex": data[:32].hex() if data else "",
            "note": "read-only; no commands sent",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def robot_info_template_markdown() -> str:
    """A fill-in checklist the user completes to identify and wire their arm safely."""
    return """# TeraFold — Robot Arm Identification Worksheet

Fill this in BEFORE attempting any real motion. TeraFold will not drive an
unknown arm; a verified protocol is required first.

## Hardware
- [ ] Arm brand: ______________________________
- [ ] Model: __________________________________
- [ ] Controller board: _______________________  (Arduino / ESP32 / Feetech / Dynamixel / other)
- [ ] Connection type: ________________________  (USB serial / USB HID / Bluetooth / Wi-Fi)
- [ ] Serial port (`terafold robot-scan`): ____  (e.g. /dev/tty.usbserial-XXXX)
- [ ] Baud rate: ______________________________  (e.g. 9600 / 115200 / 1000000)
- [ ] Protocol docs / SDK link: _______________
- [ ] Servo / motor count: ____________________
- [ ] Gripper type: ___________________________  (servo / suction / parallel jaw)

## Verification (in order — do not skip)
- [ ] Vendor software installed and the arm MOVES in it: yes / no
- [ ] Vendor software / SDK command format captured (logs / docs): ____
- [ ] Safe HOME pose photographed and noted: __________
- [ ] Workspace dimensions measured (x_min..x_max, y, z in meters): ____
- [ ] Emergency stop / power-cut within arm's reach during tests: yes / no

## Before real motion (TeraFold requirements)
- [ ] Homography calibrated (`terafold calibrate-table-from-image`)
- [ ] Perception above confidence threshold (markers detected, not fallback)
- [ ] A dry-run episode completed and reviewed
- [ ] A verified command backend wired into a TeraFold adapter
- [ ] Run with BOTH flags: `--enable-motion --i-understand-this-moves-hardware`

Until every box above is checked, use:
    terafold demo-image ... --dry-run
    terafold export-trajectory --plan-json result.json --out trajectory.csv
and drive the arm via the vendor software for verification.
"""
