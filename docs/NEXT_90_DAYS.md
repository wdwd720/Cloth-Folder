# TeraFold — the next 90 days

Derived from the Deep Research roadmap (`docs/DEEP_RESEARCH_DIGEST.md`). The theme
is **robot systems engineering first, model training second.** The bottleneck is
control + calibration + data plumbing, not model choice.

## Where we are

- Custom 7-DOF bus-servo arm is alive (`sms_sts` @ 1,000,000 baud; IDs 1,2,5,6;
  `ReadPosSpeed`/`WritePosEx` work; ID 6 moved with deadband/backlash).
- Software reaches a real **above-table ghost fold** (image → perception → plan →
  bounded base-yaw sweep) with full safety gating and logging.
- Calibration, kinematics, data, and eval are **scaffolded** and honest about what
  is not yet validated. Contact folding is locked.

## 24 hours

- Finish the full servo-ID scan; stabilize read logs for IDs 1,2,5,6
  (`servo-scan --read-only`).
- Map IDs 1,2,5,6 → joints + signs (`map-servo-joints`, two flags, very slow).
- Physically investigate IDs 3,4,7 (cabling, power branch, duplicate IDs, dead
  servo, or genuine absence).
- Save `configs/robots/physical_7dof_waveshare_joint_map.yaml`.

## 3 days

- `characterize-servos` on each mapped joint: readback noise, deadband, backlash,
  step response, repeatability. Tighten the raw limits in the joint map from the
  measured safe range.
- Confirm every write is logged and the watchdog/Ctrl+C/`STOP_TERAFOLD` paths work.
- Camera intrinsics (`calibrate-camera`, checkerboard) — start the calibration set.
- Run a real above-table ghost fold (`real-image-ghost-fold --enable-motion
  --i-understand-this-moves-hardware`) and review the log.

## 7 days

- Table homography (`calibrate-table`, 4+ known points / AprilTags) +
  `validate-table-calibration` on held-out points.
- Measure rough link lengths + a neutral pose; write a *provisional* kinematics
  file (`robot-model-status` will still report it unvalidated → IK stays refused).
- Record LeRobot-compatible episodes from ghost runs; `episode-summary` + a first
  `export-lerobot`.

## 30 days

- Bring up **one SO-101**; get LeRobot record/replay working on a known embodiment.
- 50–100 demonstrations; mirror the data schema across both arms.
- Decide if the custom arm passes the repeatability gate (readback ~5–10 units at
  rest; endpoint hover repeatability <5–8 mm). If not, demote it to a controls
  sandbox and make SO-101 the primary folding platform.
- Train a first **ACT** baseline on the real demos.

## 60 days

- `fit-robot-table-transform` from manual touch points; reach the **calibrated
  hover** gate (held-out robot↔table error <5 mm) — hover only, still no contact.
- Add a success detector; start the human-in-the-loop correction loop.

## 90 days

- First tightly-controlled **soft-contact** towel manipulations, only after 10–20
  clean hover runs and ≥5 shallow probes with no scrape.
- ACT or Diffusion Policy improving over the geometry baseline.
- Everything logged in a LeRobot-compatible format.

## Do NOT build yet (explicit)

Real contact folding before calibration · a "validated" custom-arm URDF/IK · RL
from cloth sim · full ROS 2 / MoveIt 2 migration · large VLA fine-tuning · a perfect
cloth simulator · mobile base / bimanual before one calibrated arm ghost-folds.

## Best 90-day milestone

A calibrated system performing **repeated image-driven ghost folds** plus a few
controlled soft-contact manipulations, logging everything in a LeRobot-compatible
format, with a first ACT baseline trained on real demonstrations.
