# Calibration workflow — from image pixels to safe robot motion

Calibration is what turns "a photo of a towel" into coordinates the arm can trust.
**None of it is required for the above-table ghost fold** (that only needs a joint
map). It *is* required, and must *validate*, before any calibrated hover — and
contact stays locked even after hover validates.

The chain has three geometric objects: camera intrinsics `K`, the table homography
`H` (pixels → table metres), and the robot→table transform (table metres → robot
coordinates). Each calibration artifact carries its own `valid_for_hover` and
`valid_for_contact` flags, computed from RMS error against the gates below.
`robot-status` reads these flags — presence alone never unlocks anything.

## Gates (recommended, from the roadmap)

| Capability | Intrinsics RMS | Homography RMS | Robot-touch RMS (held-out) | Endpoint repeat. |
|---|--:|--:|--:|--:|
| Ghost in air | <1.0 px | <1.5 px | <10 mm | <10 mm |
| Calibrated hover | <0.8 px | <1.0 px | <5 mm | <6 mm |
| Soft contact | <0.6 px | <0.8 px | <3 mm | <4 mm |
| Table-contact fold | <0.5 px | <0.8 px | <2–3 mm | <3 mm |

If you only hit the **ghost** row, that is still a legitimate, valuable demo. Not
hitting the **hover** row means no contact, period.

## Step 1 — Camera intrinsics

```bash
python3 -m terafold calibrate-camera \
  --images 'data/calibration/checkerboard/*.png' \
  --out runs/calibration/physical_7dof_waveshare_camera_intrinsics.yaml
```

Uses OpenCV (`pip install -e '.[vision]'`). With a checkerboard set it estimates `K`
and distortion and stores the reprojection RMS. **If OpenCV or images are missing it
reports `unavailable` gracefully** rather than crashing — intrinsics are optional for
the homography-only path but required before hover.

## Step 2 — Table homography (pixels → table metres)

```bash
python3 -m terafold calibrate-table \
  --image data/calibration/table.png \
  --points-image "u1,v1;u2,v2;u3,v3;u4,v4" \
  --points-table "0,0;0.3,0;0.3,0.2;0,0.2" \
  --out runs/calibration/physical_7dof_waveshare_table_homography.yaml
```

Four or more coplanar correspondences (clicked corners or AprilTag centres) solve a
normalized-DLT homography (pure numpy — no OpenCV needed). The YAML stores the
source/target points, `H`, `rms_error_m`, `max_error_m`, and the validity flags.

## Step 3 — Validate on held-out points

```bash
python3 -m terafold validate-table-calibration \
  --calibration runs/calibration/physical_7dof_waveshare_table_homography.yaml \
  --points-image "..." --points-table "..."
```

Apply `H` to points you did *not* fit on and check the held-out RMS. High error
marks the calibration invalid for hover/contact (and `robot-status` will keep those
levels locked).

## Step 4 — Robot → table transform (table metres → robot coords)

```bash
# DRY-RUN plan first (moves nothing): how to touch known table points by hand.
python3 -m terafold robot-touch-calibration \
  --robot physical_7dof_waveshare \
  --joint-map configs/robots/physical_7dof_waveshare_joint_map.yaml \
  --out runs/calibration/robot_table_touch_points.json --dry-run
```

You manually bring the tip to several known table points and record the robot
coordinates, producing a `touch_points.json` with `{table_xy, robot_xyz}` pairs.
Then fit the rigid transform (Kabsch/Umeyama):

```bash
python3 -m terafold fit-robot-table-transform \
  --touch-points runs/calibration/robot_table_touch_points.json \
  --out runs/calibration/physical_7dof_waveshare_robot_table_transform.yaml
```

The YAML stores `R`, `t`, `rms_error_m`, `max_error_m`, and validity flags.

## Step 5 — Confirm what unlocked

```bash
python3 -m terafold robot-status --robot physical_7dof_waveshare --probe
```

When intrinsics, homography, and robot→table transform are all present **and valid
for hover**, the ladder unlocks **Level 5 (calibrated_hover)**. Contact (Level 6+)
remains locked: it additionally needs hover-accuracy logs, a low-force/compliance
mode, and an explicit operator unlock — and those are intentionally not built yet.

## Why contact stays locked

Even a perfect calibration only tells the arm *where* the towel is. Contact also
requires: validated repeatability under load, a compliant/low-force descent, a
gripper characterization, a success detector, and demonstrated hover accuracy over
many runs. The roadmap requires 10–20 clean hover runs and ≥5 shallow probes with no
scrape before a contact fold is attempted. TeraFold encodes that as the unlock
ladder, so the software refuses to skip it.
