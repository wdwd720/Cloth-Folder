# Custom bus-servo arm — bring-up guide

How to take the custom 7-DOF Waveshare arm from "alive on the bus" to a mapped,
characterized actuator layer that can perform a safe above-table ghost fold.
**Every motion step defaults to dry-run and needs both motion flags.** Read
`docs/REAL_HARDWARE_SAFETY.md` first.

## 0. Hardware facts (confirmed)

| Item | Value |
|---|---|
| Adapter | Waveshare Bus Servo Adapter A |
| Protocol | `sms_sts` |
| Baudrate | 1,000,000 |
| macOS port | `/dev/cu.usbmodem5AB01803321` |
| Working SDK calls | `ReadPosSpeed(id)`, `WritePosEx(id, target, speed, acc)` |
| Responding IDs | 1, 2, 5, 6 |
| Unconfirmed IDs | 3, 4, 7 |
| Power | DC 9–12.6 V |

The vendor SDK (`scservo_sdk`) is bundled under
`vendor/waveshare/STServo_Python/stservo-env`. If `pip install pyserial` is present,
TeraFold imports it automatically.

## 1. Confirm you are safe to proceed

```bash
python3 -m terafold robot-status --robot physical_7dof_waveshare --probe
```

The `--probe` opens the port **read-only** (ping + read). You want to see
`protocol_confirmed: True` and `responding IDs: [1, 2, 5, 6]`. The ladder will show
Level 2 (tiny_nudge) unlocked and Level 3 (joint_map) blocked because no joint map
exists yet.

## 2. Read-only servo scan

```bash
python3 -m terafold servo-scan --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1-30 --read-only
```

Writes `runs/servo_scan/<ts>/scan.json` and `summary.md`. It pings each ID and
reads position/speed where the ping answers. It **never** writes a servo. Use it to
confirm exactly which IDs are present and to document the missing ones (3,4,7) for
physical debugging (cabling, power branch, duplicate IDs, dead servo).

## 3. Map servo IDs → joints (this MOVES one servo at a time)

```bash
python3 -m terafold map-servo-joints --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1,2,5,6 \
  --delta-units 40 --speed very_slow \
  --enable-motion --i-understand-this-moves-hardware
```

For each ID it reads the current position, nudges by a tiny `--delta-units`, asks
you which joint moved, infers the sign from the readback, then returns to start. It
writes `configs/robots/physical_7dof_waveshare_joint_map.yaml` containing, per
servo: `joint`, `role`, `sign`, `raw_home`, conservative `raw_min`/`raw_max`,
`last_observed`, `confidence`, and notes — plus the disabled/missing IDs. Keep a
hand on the power switch. Keep every motion well above the table.

After this, `robot-status` shows Level 3 (and Level 4 if a sweep joint such as
`base_yaw` is mapped).

## 4. Characterize the servos (deadband / backlash / repeatability)

```bash
# Default DRY-RUN: prints the plan, moves nothing.
python3 -m terafold characterize-servos --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1,2,5,6 \
  --out runs/servo_characterization/session_001 --dry-run

# Real, conservative characterization (both flags):
python3 -m terafold characterize-servos --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1,2,5,6 \
  --out runs/servo_characterization/session_001 \
  --enable-motion --i-understand-this-moves-hardware
```

Records readback noise at rest, first effective delta (deadband), backlash
(approaching a target from both directions), step response (settle error,
overshoot, time-to-settle), and return-to-home repeatability. Outputs CSV traces, a
metrics JSON, and a Markdown report (plots if matplotlib is installed). Use the
measured safe range to tighten `raw_min`/`raw_max` in the joint map.

**Abandonment gate (from the roadmap):** if you cannot get readback repeatability
~5–10 raw units at rest, endpoint hover repeatability <5–8 mm, and a monotone
response without large dead zones, stop betting the folding roadmap on this arm and
move the learning effort to an SO-101.

## 5. Above-table ghost fold (no contact)

```bash
# Dry-run (default): perception → plan → ghost plan, prints everything, moves nothing.
python3 -m terafold real-image-ghost-fold --image ~/Downloads/towel_demo_pic.png \
  --robot physical_7dof_waveshare --port /dev/cu.usbmodem5AB01803321 \
  --height-clearance-m 0.10 --speed very_slow --dry-run

# Real ghost sweep (both flags). ABOVE the table only — never touches the cloth:
python3 -m terafold real-image-ghost-fold --image ~/Downloads/towel_demo_pic.png \
  --robot physical_7dof_waveshare --port /dev/cu.usbmodem5AB01803321 \
  --height-clearance-m 0.10 --speed very_slow \
  --enable-motion --i-understand-this-moves-hardware
```

This runs the *real* perception/planning pipeline, prints the image-space and
normalized table-space grasp/place, and — because the arm has no validated
kinematics — executes only a conservative, bounded **base-yaw sweep** derived from
the fold direction. It refuses if the joint map is missing, the protocol is
unconfirmed, positions cannot be read, or the sweep would leave the safe range.

## 6. What is still locked

Contact folding, calibrated hover *to contact*, and autonomous Cartesian IK on the
custom arm. See `docs/CALIBRATION_WORKFLOW.md` for the calibration chain that
unlocks hover, and why contact stays locked even after that.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `protocol_confirmed: False` | power off / wrong port / SDK missing | check DC 9–12.6 V, USB-C, port; `pip install pyserial` |
| only 4 servos respond | IDs 3,4,7 miswired/absent | isolate each motor; check cabling, power rail, duplicate IDs |
| servo moves but doesn't return | backlash / too-small step | log settle error; do not assume an exact return |
| nudge refused | missing flags or unconfirmed protocol | add both motion flags; re-run `robot-status --probe` |
| "outside safe range" refusal | target left `safe_position_units` | reduce delta; re-check joint map raw limits |
