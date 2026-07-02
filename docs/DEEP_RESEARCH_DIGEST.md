# TeraFold — Deep Research engineering digest

> Internal, actionable distillation of `deep-research-report.md`
> ("Tera Robotics technical roadmap from the current TeraFold state").
> This is the **source-of-truth strategy** the platform code implements.
> Every claim here is the report's, condensed for engineering use.

## The one-paragraph thesis

The bottleneck is **control + calibration + data plumbing**, *not* model choice.
Keep the custom bus-servo arm because it is *already alive* (`sms_sts` @ 1,000,000
baud, IDs 1/2/5/6, `ReadPosSpeed`/`WritePosEx` working). It is the fastest path to
the real fundamentals: servo characterization, joint mapping, calibration, safe
logging, and an **image-driven ghost fold above the table**. Buy/build **one
SO-101 soon** as the standardized LeRobot data/learning platform. Strategy =
**hybrid** (custom arm = hardware-integration & safety lab; SO-101 = clean
datasets + imitation learning). Folding is the first *technical wedge*, not the
whole product.

## Current-state classification (treat as true unless logs contradict)

| Bucket | Items |
|---|---|
| **Confirmed** | Waveshare Bus Servo Adapter A; `sms_sts`; 1,000,000 baud; `ReadPosSpeed`/`WritePosEx` work; IDs 1,2,5,6 respond; ID 6 real motion (with deadband/backlash); perception + planner + sims + safety-gated pipeline already exist. |
| **Likely, unconfirmed** | Position-controlled low-cost stack w/ backlash/deadband; IDs 3,4,7 missing/miswired/bad-branch; units near 12-bit but zero-offset & sign unknown. |
| **Unknown — must resolve** | Full joint map; missing IDs; per-joint safe limits; home pose; link lengths; joint axes; gripper; table frame; camera intrinsics; cam→table homography; base→table transform; URDF quality; repeatability under load. |
| **Dangerous assumptions (must NOT drive motion)** | image pixels == robot coords; arm has accurate absolute zero; unit→angle scale known; all joints exist/mapped; contact is safe pre-calibration; a learned policy can paper over missing kinematics. |

## Recommended stacks

### Immediate (this week)
Custom arm + Waveshare · bounded joint-space + readback monitor · keypoints +
classical fallback · camera intrinsics + homography *start* · procedural cloth +
MuJoCo arm · JSON logs + begin LeRobot export · geometry baseline · **dry-run
default**.

### 30 days
Custom arm + 1 SO-101 · custom backend + SO-101 replay · segmentation + robust
keypoints + success detector · robot↔table transform + hover validation · MuJoCo
controller harness · LeRobot v3 datasets · **ACT** on demos · unlock-ladder
enforced.

### 90 days
SO-101 primary, custom secondary · calibrated hover + soft contact · uncertainty-
aware closed-loop perception · stable calibration pipeline · controlled cloth
comparisons · HIL datasets · ACT or Diffusion Policy · batch eval + replay.

## Top-20 next actions (ranked)

1. Finish full servo-ID scan; stabilize logs for IDs 1,2,5,6.
2. Map IDs 1,2,5,6 → physical joints + signs.
3. Physically investigate IDs 3,4,7 (cabling, power branch, bad servo, absence).
4. Save `*_joint_map.yaml`.
5. Measure readback noise at rest per joint.
6. Measure deadband + backlash per joint.
7. Establish conservative software limits per joint.
8. Integrate the real `waveshare_sms_sts_backend.py` into TeraFold.  ✅ present
9. Add motion logging + watchdog to every write.
10. Single-joint characterization CLI.
11. Bounded image-independent ghost sweep with the mapped base joint.
12. Calibrate camera intrinsics.
13. Calibrate table homography (4 known pts / AprilTags).
14. Measure rough link lengths + neutral pose.
15. Provisional kinematic chain + FK.
16. Validate FK with marker observations.
17. Image-driven **dry-run** ghost-fold planning.  ✅ present
18. Image-driven ghost fold in air (no contact).
19. Record LeRobot-compatible episodes from ghost runs.
20. Order/build one SO-101; mirror the data schema there.

## Safety unlock ladder (machine-enforced)

| Level | Allowed | Forbidden | Required artifacts |
|---|---|---|---|
| 0 No-hardware | simulation only | serial open | sim tests |
| 1 Read-only | ping, reads | writes | servo-scan log |
| 2 Tiny nudge | one-ID micro motion | multi-servo | per-ID joint video |
| 3 Joint map | one-joint bounded moves | contact | `joint_map.yaml` |
| 4 Ghost fold | bounded above-table sweep | any table descent | limits + dry-run compare |
| 5 Calibrated hover | hover over points | contact grasp | intrinsics + homography + robot↔table |
| 6 Soft contact | shallow cloth touch | table scrape | hover-accuracy logs + compliance |
| 7 Real fold | constrained fold primitive | free exploration | repeatability + success detector |
| 8 Repeated autonomy | batch trials | uncapped runs | watchdog + intervention logging |

Watchdog aborts on: no readback >250–500 ms · target out of limits · repeated
fail-to-reach · temp/voltage out of range · est. EE below clearance in ghost mode
· Ctrl+C · operator abort. **Dry-run is the invariant default**; real motion
needs an explicit motion flag *and* an "I understand this moves hardware" gate.

## Calibration gates (recommended, not literature-mandated)

| Capability | Intrinsics RMS | Homography RMS | Robot-touch RMS (held-out) | Endpoint repeat. | Motion allowed |
|---|--:|--:|--:|--:|---|
| Read-only | — | — | — | — | reads only |
| **Ghost in air** | <1.0 px | <1.5 px | <10 mm | <10 mm | above-table sweep |
| Calibrated hover | <0.8 px | <1.0 px | <5 mm | <6 mm | hover above grasp |
| Soft contact | <0.6 px | <0.8 px | <3 mm | <4 mm | shallow touch |
| Table-contact fold | <0.5 px | <0.8 px | <2–3 mm | <3 mm | only after repeats |

Not hitting **hover** ⇒ no contact. Hitting only **ghost** is still a real,
valuable demo.

## Custom-arm control plan

Treat each bus servo as an **inner-loop position actuator**; build a **slow outer
supervisory controller** (bounded interpolation, readback verify, watchdog,
temp/voltage/load if available, refuse contact pre-calibration). **Do NOT** wrap
an aggressive external PID around `WritePosEx` — it fights the servo's internal
loop. Experiments before any contact: full ID scan · readback stability · tiny-
nudge mapping · deadband · backlash · step response · load sag · safe-range ·
multi-joint · ghost-fold dry run.

**Units→degrees is an empirical affine calibration, not a known spec:**
`θ_i ≈ s_i·(u_i − u0_i)`. If it is a 4096-count single-turn encoder then
`s ≈ 360/4096 ≈ 0.0879°/unit` — *likely, not confirmed*. Measure per revolute
joint (gearing can differ shaft vs joint).

## Robot model without a URDF

Map IDs→joints → signs via +nudges → neutral pose → caliper link lengths → infer
axes from single-joint motion → DH or **POE** chain → collect EE observations over
many poses → optimize params (`scipy.optimize.least_squares`, Huber loss) to min
reprojection/3D error → validate held-out error → IK only in validated workspace →
visual-servo the final cm. POE is usually easier to fit than DH for custom arms.

## Calibration objects (OpenCV path)

Intrinsics `K`+distortion (`calibrateCamera`, checkerboard/Charuco) · table
homography `H` (`findHomography`, 4+ coplanar pts / AprilTags, RANSAC) · robot-on-
table (touch known pts with tip marker) · hand-eye if needed (`calibrateHandEye`,
`AX=XB`) · rigid point alignment via **Kabsch/Umeyama**. Validate on held-out pts.

## IK & motion generation

Joint-space execution is the safety backbone; Cartesian only as an intermediate
once FK is believable. First IK = **damped least squares**
`Δq = Jᵀ(JJᵀ+λ²I)⁻¹e`; add null-space secondary `Δq = J#e + (I−J#J)Δq_sec` for
7-DOF redundancy. Trajectories now: cubic **min-jerk** point-to-point or
trapezoidal velocity. **Do NOT** add CHOMP/STOMP/RRT yet.

## Perception / model strategy

Keep keypoints + classical fallback (planner is keypoint-based). Add segmentation
(recover corners, detect occlusion). Add a **success detector** to close the loop.
Later: video tracking. SAM/SAM2 for bootstrapping masks only; DINOv2 as a frozen
backbone for small real sets. Data increments: 100 → 500 → 2,000 images, then
video + success labels.

## Learning order (hybrid, concrete)

1. Classical geometry + keypoints + joint-space controller. **(now)**
2. Visual-servo correction. **(now, once table calibrated)**
3. **ACT** on real demos (50–200/task, single GPU). **(soon — first IL step)**
4. **Diffusion Policy** / chunked sequential (200–1000+ demos). **(soon after)**
5. HIL / HG-DAgger correction. **(after a baseline exists)**
6. VLAs (OpenVLA etc.) — later, selectively.

Action representation on the custom arm = **joint-space** (no kinematics needed,
maps to motor commands). Add EE-space only after a validated URDF/IK.

## Data / LeRobot strategy

Use LeRobot **now** for *dataset structure, replay, training* — not as day-one
runtime controller. v3 stores high-rate signals in Parquet, multi-cam video in
MP4 shards, metadata/normalization in JSON. Log per frame: timestamp, episode id,
task, robot type, servo ids, joint positions/targets/speeds (raw), images,
intrinsics id, homography id, perception (mask/corners/grasp/place/confidence),
planned + executed action, success, failure reason, safety flags (dry_run,
contact_enabled). LeRobot "Bring Your Own Hardware" base class is the integration
seam. Don't require a full LeRobot install to validate the internal schema.

## Simulation strategy

Do **not** rebuild sim. Keep the **procedural cloth proxy** for fold geometry/demo
logic; use **MuJoCo rigid-arm** as the controller/kinematics test harness; treat
MuJoCo flex cloth as a *research instrument*, not ground truth (sim-to-real cloth
gap is large). Don't burn weeks on a perfect towel simulator before a calibrated
hover + real ghost fold exist.

## Evaluation metrics (five families)

- **Perception:** corner err, mask IoU, grasp/place err, confidence calibration.
- **Calibration:** intrinsics reproj RMS, homography RMS, held-out table-pt err,
  robot-touch err, drift.
- **Control:** readback err, settle err, repeatability, overshoot, fail-to-reach,
  heat/load/voltage margins.
- **Cloth/Fold:** fold-line err, edge alignment, success rate, recovery, variation.
- **Product:** time/episode, interventions/episode, setup burden, operator minutes.

First pass/fail: perception median corner err <10 px (held-out) before image-driven
hover · calibration held-out robot↔table <5 mm before hover, <3 mm before contact ·
control no packet corruption during coordinated motion, settle err 5–10 units ·
ghost fold = 20 consecutive safe runs · contact only after 10–20 hover runs + 5
shallow probes with no scrape.

## Decision tree (hardware fork)

Characterize → all required joints found? → no → inspect cabling/power/IDs →
recovered? → no ⇒ **demote custom arm to controls sandbox, shift folding to
SO-101**. Yes → map + limits → repeatability/backlash acceptable? → no ⇒ ghost +
data plumbing only; get SO-101 now. Yes → provisional FK + table calibration →
hover error below gate? → no ⇒ improve calibration/visual-servo, no contact. Yes
⇒ unlock hover-only image-driven motion → soft contact stable? → yes ⇒ attempt
controlled contact folding.

**Abandonment threshold for the custom arm as the *primary folding* platform:**
can't get readback repeatability ~5–10 raw units at rest, endpoint hover
repeatability <5–8 mm in a restricted workspace, or monotone response without
large dead zones.

## "Do NOT build yet" list

- Real autonomous **contact folding** before calibration.
- A "validated" custom-arm **URDF/IK** (scaffold that *refuses*, don't fake it).
- **RL-from-sim** cloth policies.
- Full **ROS 2 / MoveIt 2** migration (revisit after a credible URDF).
- Huge **VLA** fine-tuning from scratch.
- A perfect **cloth simulator**.
- **Mobile base** / bimanual escalation before one calibrated arm ghost-folds.

## Best 90-day milestone

A calibrated system performing **repeated image-driven ghost folds** plus some
tightly-controlled **soft-contact** towel manipulations, logging everything in a
LeRobot-compatible format, with a first **ACT** baseline trained on real demos.
