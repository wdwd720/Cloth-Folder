# TeraFold platform sprint — final report

Brutally honest. Where something is scaffolding, it says scaffolding. Where a
model/IK/calibration is unvalidated, it is not implied otherwise.

## 1. Which Deep Research file was read

`deep-research-report.md` (repo root, 51 KB) — *"Tera Robotics technical roadmap
from the current TeraFold state."* Distilled into `docs/DEEP_RESEARCH_DIGEST.md`
(immediate/30/90-day stacks, top-20 actions, safety ladder, calibration gates,
control/calibration/data/sim/perception/eval strategy, decision trees, and the
explicit "do not build yet" list).

## 2. Repo audit summary

A substantial stack already existed: numpy-only core with lazy heavy deps; a Typer
CLI; `robot/safety.py` (two-flag gate, watchdog, STOP file); a confirmed
`waveshare_sms_sts_backend.py` (write gated on `protocol_confirmed`, safe-range
clamp, e-stop); `real_image_ghost.py` (real perception→plan→derived base-yaw ghost
sweep — genuinely not hardcoded); `joint_map.py`; `data/episode_schema.py`;
perception/planning/sim/vision modules. **Baseline: 184 tests passing.** Gaps: no
central capability/level system, no kinematics/calibration packages, no
characterization, thin servo-scan, no structured episode schema/eval/sim-preview.

## 3. What was implemented

A central, machine-enforced **capability + 8-level safety-state system**; a
**kinematics** scaffold (refuses IK/contact while unvalidated); a **calibration**
package (homography, intrinsics, robot↔table, validation); a reusable **motion-core
+ safety-gate + interpolation** library; **read-only servo-scan** + **servo
characterization**; a full **episode data schema + episode logger + LeRobot export
scaffold + summary**; an **eval** metric suite + report; a **sim ghost-fold
preview**; an enriched **joint-map schema**; and six operator docs. Built layer by
layer; tests run after each layer; full suite green throughout.

## 4. Files changed (existing, edited)

- `terafold/cli.py` — wired 12 new commands; replaced `servo-scan` with the
  read-only scanner.
- `terafold/robot/waveshare_sms_sts_backend.py` — added the named interface
  (`read_pos_speed`, `read_all`, `ping_many`, `available`, `probe_protocol`,
  `write_pos_ex`, `safe_stop`, `close_safely`).
- `terafold/robot/joint_map.py` — enriched schema (`raw_home/min/max`, `role`,
  `confidence`, `disabled_ids`, `operator_notes`, `validate()`, `raw_limits_for_id`).
- `terafold/robot/real_image_ghost.py` — `run_map_servo_joints` now writes the rich
  joint map (sign inferred from readback; one operator question per servo).
- `README.md` — new "§5b Physical 7-DOF arm — capability-gated platform".

## 5. New modules

`robot/safety_state.py`, `robot/capabilities.py`, `robot/motion_core.py`,
`robot/safety_gate.py`, `robot/interpolation.py`, `robot/servo_scan.py`,
`robot/characterization.py`; `kinematics/{raw_units,fk,ik_dls,custom_arm_model,model_status}.py`;
`calibration/{table_homography,camera_intrinsics,robot_table_transform,validation}.py`;
`data/{schema,episode_logger,lerobot_export,summary}.py`;
`eval/{perception_metrics,control_metrics,calibration_metrics,fold_metrics,report}.py`;
`sim/ghost_preview.py`. **19 new module files.**

## 6. New CLI commands (12)

`robot-status`, `robot-model-status`, `characterize-servos`, `calibrate-table`,
`validate-table-calibration`, `calibrate-camera`, `robot-touch-calibration`,
`fit-robot-table-transform`, `episode-summary`, `export-lerobot-logs`, `eval-run`,
`sim-real-ghost-fold` (plus `servo-scan` rebuilt read-only). Total CLI: 55 commands.

## 7. New configs

No new YAML committed (artifacts are generated at runtime). The capability probe
discovers calibration/kinematics at conventional paths:
`configs/robots/<robot>_joint_map.yaml`, `runs/calibration/<robot>_camera_intrinsics.yaml`,
`<robot>_table_homography.yaml`, `<robot>_robot_table_transform.yaml`,
`configs/robots/<robot>_kinematics.yaml`. The existing
`configs/robots/physical_7dof_waveshare.yaml` is the source of truth for IDs/limits.

## 8. New tests (11 files)

`test_capabilities`, `test_kinematics`, `test_calibration`, `test_motion_core`,
`test_data_foundation`, `test_eval_metrics`, `test_servo_scan_char`,
`test_sim_ghost_preview`, `test_joint_map_schema`, `test_integration_seam`,
`test_docs_commands`. **Suite: 184 → 312 passing (+128), 0 failing.**

## 9. Safety gates added

- The 8-level unlock ladder (`safety_state.py`) with per-level required artifacts,
  allowed/forbidden commands, required flags, required logs, abort conditions.
- `capabilities.probe_capabilities` evaluates the ladder from real config + joint
  map + calibration validity + an optional read-only probe; `contact_allowed` is
  True only at Level ≥6 **and** with an explicit unlock flag (never reachable today).
- `MotionGate` (reusable): dry-run/flag/limit/active-id/joint-map/calibration/contact
  checks. Backend writes still refuse unless `protocol_confirmed`. Disabled IDs
  (3,4,7) are never moved. Custom-arm IK is refused (`validated=False`).

## 10. Read-only commands (never move a motor)

`robot-status` (+`--probe`), `robot-model-status`, `robot-probe --read-only`,
`servo-scan`, `robot-info-template`, `robot-scan`, `episode-summary`, `eval-run`,
`sim-real-ghost-fold`, `calibrate-table`, `validate-table-calibration`,
`calibrate-camera`, `fit-robot-table-transform`, `export-lerobot-logs`.

## 11. Dry-run-only by default (print, don't move)

`real-image-ghost-fold` (no flags), `characterize-servos --dry-run`,
`robot-touch-calibration --dry-run`, `replay-joint-demo --dry-run`,
`map-servo-joints` (without both flags), `servo-nudge` (without both flags).

## 12. Commands that CAN move hardware (both flags + confirmed protocol)

`map-servo-joints …--enable-motion --i-understand-this-moves-hardware`,
`servo-nudge …`, `characterize-servos …`, `real-image-ghost-fold …`
(above-table ghost sweep ONLY — never contact). Each needs `protocol_confirmed`
and refuses disabled IDs / out-of-range targets.

## 13. Commands still forbidden (system refuses)

Any contact fold; `real-image-fold` (image→table contact); calibrated hover *to
contact*; autonomous Cartesian IK on the custom arm (kinematics unvalidated). These
refuse with the exact failed gate.

## 14. Artifacts logged

`runs/real_motion_logs/<cmd>_<ts>.jsonl` (request args, config + joint-map
snapshots, perception, plan, each read/write, refusals, aborts, final status);
`runs/servo_scan/<ts>/{scan.json,summary.md}`; `runs/servo_characterization/…`
(CSV traces, metrics JSON, Markdown); `runs/calibration/<robot>_*.yaml`;
`runs/eval/*.md`; LeRobot scaffold under the export dir.

## 15. LeRobot / data support

`data/schema.py` (typed `EpisodeFrame`/`Episode`; **`action_vector` ordered by
ascending servo id**), `episode_logger.py` (motion-log JSONL → `Episode`),
`lerobot_export.py` (writes `meta/info.json` + per-episode Parquet/JSONL scaffold
**without requiring LeRobot**; records whether calibration was present),
`summary.py` (`episode-summary`). Verified: 207 frames / 59 episodes exported from
real ghost-fold logs to a Parquet scaffold.

## 16. Calibration support

Pure-numpy normalized-DLT **table homography** + held-out **validation**; OpenCV
**camera intrinsics** (graceful `unavailable` if cv2/images absent); **Kabsch/Umeyama
robot↔table** transform from manual touch points (dry-run plan default; never
auto-drives). Each artifact carries `valid_for_hover`/`valid_for_contact` flags, and
the capability probe discovers them — **verified end-to-end** (`test_integration_seam`):
a full valid chain unlocks Level 5 (calibrated hover); a corrupted homography drops
back to Level 4; contact stays locked throughout.

## 17. Simulations updated

Added `sim-real-ghost-fold`: image-derived fold direction + normalized path + air-only
ghost waypoints + table plane + clearance + active servo IDs (if a joint map is
given), with explicit `contact_disabled=True`, missing-calibration and
missing-kinematics warnings. Works without hardware; renders a 2D top-down preview if
matplotlib is present, else writes metadata-only. Existing `sim-fold` untouched.

## 18. Test results

`python3 -m pytest` → **312 passed, 0 failed** (~15 s). Ruff clean on all new code.
All 27 new/changed modules import; CLI `--help` loads.

## 19. What remains unknown (honest)

- IDs 3,4,7: present? miswired? — **physically unverified**.
- Joint map for the real arm: **not yet built** (needs hardware).
- Servo unit→angle scale, deadband, backlash, repeatability: **not measured**.
- Camera intrinsics, table homography, robot↔table transform: **not captured**.
- A validated custom-arm kinematic model / URDF: **does not exist** (scaffold only).
- Whether the arm passes the repeatability gate to remain the *primary* folding
  platform: **undecided** (the report's abandonment threshold applies).

## 20. Exact next real-world commands (run manually, in order)

```bash
python3 -m terafold robot-status --robot physical_7dof_waveshare --probe
python3 -m terafold servo-scan --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1-30 --read-only
python3 -m terafold map-servo-joints --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1,2,5,6 --delta-units 40 --speed very_slow \
  --enable-motion --i-understand-this-moves-hardware
python3 -m terafold characterize-servos --robot physical_7dof_waveshare \
  --port /dev/cu.usbmodem5AB01803321 --ids 1,2,5,6 \
  --out runs/servo_characterization/session_001 --dry-run        # then drop --dry-run + add both flags
python3 -m terafold real-image-ghost-fold --image ~/Downloads/towel_demo_pic.png \
  --robot physical_7dof_waveshare --port /dev/cu.usbmodem5AB01803321 \
  --height-clearance-m 0.10 --speed very_slow --dry-run          # review, THEN add both flags
```

Keep a hand on the power switch. Keep every motion above the table.

## 21. What NOT to run yet

Any contact fold; `real-image-fold`; any "hover to grasp"; autonomous Cartesian IK.
Do not raise the contact unlock. Do not trust the unit→angle scale or image→table
mapping until calibrated. Do not assume IDs 3,4,7 exist.

## 22. What the next coding sprint should do

1. After a real joint map + characterization, refactor `real_image_ghost` to route
   through `MotionGate`/`motion_core` under test protection (defense-in-depth;
   today it enforces equivalent gates inline).
2. Wire `characterize-servos` outputs back into the joint map's `raw_min/raw_max`.
3. Capture intrinsics + homography + touch points and reach Level 5 (hover) on real
   data; add hover-accuracy logging (the Level-6 prerequisite).
4. Bring up one SO-101 and mirror the schema; record 50–100 demos; train a first ACT
   baseline from the LeRobot export.
5. Add a success detector + HIL correction loop before any contact attempt.
