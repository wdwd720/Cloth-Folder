# Real-hardware safety — read this before touching the arm

TeraFold can command a real bus-servo arm, but it is built so that **doing nothing
dangerous is the default and doing something physical is hard, explicit, and
logged.** This document is the operator contract.

## The golden rules

1. **Dry-run is the default everywhere.** No command moves a servo unless you pass
   BOTH `--enable-motion` AND `--i-understand-this-moves-hardware`.
2. **No motion without a confirmed protocol.** A write is refused unless the backend
   has confirmed `sms_sts` (port open @ 1,000,000 baud + a servo pinged + a position
   read succeeded).
3. **Disabled / unconfirmed servo IDs are never moved.** IDs `3, 4, 7` are unmapped;
   the system refuses to command them.
4. **Contact folding is LOCKED.** It cannot run until the full calibration chain
   exists *and validates*, plus an explicit operator unlock. There is no flag that
   skips this.
5. **Image pixels are never treated as robot coordinates.** Without calibration the
   only real motion allowed is an above-table *ghost* (no contact).
6. **Every motion session is logged** to `runs/real_motion_logs/<cmd>_<ts>.jsonl`.

## Always run this first

```bash
python3 -m terafold robot-status --robot physical_7dof_waveshare            # offline
python3 -m terafold robot-status --robot physical_7dof_waveshare --probe    # read-only serial
```

`robot-status` is the single "am I safe to proceed?" check. It prints the unlocked
safety **Level** (0–8) and, for every higher level, the exact artifact that is
missing. `--probe` opens the port **read-only** (ping + read; never writes).

## The safety unlock ladder (machine-enforced)

| Level | Name | What it allows | Gate to reach it |
|--:|---|---|---|
| 0 | sim_only | sim / planning / data tooling | always |
| 1 | read_only | ping + read positions | SDK + port reachable |
| 2 | tiny_nudge | one-servo micro move | protocol confirmed (ping+read) |
| 3 | joint_map | one-joint bounded moves | `*_joint_map.yaml` exists |
| 4 | ghost_fold | above-table sweep, no descent | mapped sweep joint + safe limits |
| 5 | calibrated_hover | hover over points, no contact | intrinsics + homography + robot→table (valid) |
| 6 | soft_contact | shallow cloth touch | hover logs + compliance + explicit unlock |
| 7 | constrained_fold | one fold primitive | repeatability + success detector |
| 8 | repeated_autonomy | batch trials | watchdog + intervention logging |

The system today reaches **Level 4 at most** (a real above-table ghost fold, once a
joint map exists). Levels 5–8 are scaffolded and **refuse** with a clear reason.

## Command classes

### Read-only (never moves a motor)
`robot-status`, `robot-status --probe`, `robot-probe --read-only`,
`servo-scan --read-only`, `robot-info-template`, `robot-model-status`.

### Dry-run only by default (prints what it *would* do)
`real-image-ghost-fold` (no flags), `characterize-servos --dry-run`,
`robot-touch-calibration --dry-run`, `replay-joint-demo --dry-run`,
`map-servo-joints` (without both flags).

### Can move hardware (requires BOTH motion flags + confirmed protocol)
`map-servo-joints --enable-motion --i-understand-this-moves-hardware`,
`servo-nudge ... --enable-motion --i-understand-this-moves-hardware`,
`characterize-servos --enable-motion --i-understand-this-moves-hardware`,
`real-image-ghost-fold --enable-motion --i-understand-this-moves-hardware`
(above-table ghost sweep ONLY — no contact).

### Forbidden until calibration (the system refuses)
Any contact fold, `real-image-fold` (image→table contact), calibrated hover with
contact, autonomous Cartesian IK on the custom arm (kinematics unvalidated).

## Emergency stop

1. **Cut power immediately**: switch off / unplug the DC 9–12.6 V supply.
2. `Ctrl+C` in the terminal — the watchdog trips a stop and disables torque if a
   verified backend is open.
3. Drop a file named `STOP_TERAFOLD` in the working directory — any in-progress
   motion check aborts at the next step.
4. `python3 -m terafold robot-estop --robot physical_7dof_waveshare` — best-effort
   torque-off + port close, then prints the manual power-cut steps.
5. Unplug the USB-C serial cable. Keep hands clear until power is removed.

## What "confirmed" means (so you trust the gate)

`protocol_confirmed` is True only when, in one session: the port opened at
1,000,000 baud, **and** at least one configured servo answered a ping, **and** a
position read returned a value. If any of those fail, writes stay refused — the arm
will not move "blind".
