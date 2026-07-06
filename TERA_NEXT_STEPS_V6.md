# TeraFold — V6 Reachability & Placement Diagnostics (Next Milestone)

Generated: 2026-07-05 — read-only audit. Companion to [`REPO_AUDIT_TERA_ROBOTICS.md`](REPO_AUDIT_TERA_ROBOTICS.md), [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md), [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md).

---

## 1. Why V6, and why it is NOT a model problem

The Brev evidence chain is unambiguous:

- **Assisted (particle-space) fold works:** towel width 0.6800 → best 0.3373 → final 0.3392. The physics scene and the fold *target* are sound.
- **Pure robot-contact fold fails:** width stayed 0.6800 → 0.6800 → 0.6800. The robot moved nothing.
- **V5 diagnostics found the cause:** the nearest jaw-to-towel-edge distance was **~0.406 m** — the jaw never reached the towel. This is a **reachability / placement / gripper-geometry** problem, not a policy-quality problem.

Two independent pieces of *local* evidence corroborate that this is geometry, not learning:

1. `terafold/sim/so101.py` sets `nominal_reach() = 0.366 m` for the SO-101 approximation. **0.366 m < 0.406 m**, so even a generous local estimate says the towel edge is *beyond the arm's total reach* at the current placement. (Caveat: derive the real number from the same URDF/USD the Isaac scene uses — `SO101_SPEC` is an approximation.)
2. `terafold/sim/fold_sim.py:981` silently scales the SO-101 links by up to **1.6×** so the local sim arm can always reach the plan. The local sim has been *hiding* exactly this failure mode. Any "reach looked fine in the local sim" result is invalid.

**Conclusion:** do not train a new policy, do not attempt contact, until reachability is proven. V6 is a pure diagnostic-and-search milestone that answers one question: *at what arm-base and towel placement (and with what gripper approach) can an SO-101 jaw actually reach the towel edge to grasp it?*

---

## 2. What local repo assets plug into V6 (and what does not)

**Directly reusable (pure numpy, unit-agnostic — feed Isaac particle positions in meters):**
- `terafold/math/geometry.py` — `fold_line_from_corners`, `reflect_point_across_line`, `point_line_distance`, `order_corners_clockwise`, `polygon_centroid`. Computes crease, grasp targets, and edge/point distances directly on towel-corner particles.
- `terafold/math/interpolation.py` — `quadratic_bezier`, `time_parameterize`, `linear_waypoints`. Generates the candidate jaw approach path.
- `terafold/eval/fold_metrics.py`, `terafold/eval/perception_metrics.py`, `terafold/physics/fold_quality.py` — crossed-crease, edge error, IoU, area ratio, side-correctness. For scoring reachability outcomes and (later) fold quality.
- `terafold/planning/fold_plan.py` + `trajectory.py` — the pick-lift-arc-place planner (10 phases). This is the piece Brev *lacks*; V6 produces the grasp/place targets it needs.

**Honest gap — local kinematics CANNOT do the reachability check as-is:**
- `terafold/kinematics/` has a **generic DH FK** (`fk.py`, arm-agnostic) but **no DH parameter set for the SO-101** (or any real arm), and its IK (`ik_dls.py`) is **hardcoded to a toy 2-link planar arm**. There is no N-DOF IK and no workspace/reachability utility anywhere.
- `terafold/kinematics/custom_arm_model.py` is bound to the 7-DOF Waveshare arm and structurally refuses IK.
- So V6 reachability must use **Isaac's own SO-101 articulation + a solver** (e.g. cuRobo / Isaac Lab IK, or a joint-space forward sweep), *not* the local kinematics package. The local `math`/`eval` code is for target computation and scoring, not for solving arm reach.

**Do not reuse:** the local MuJoCo sim's reach result (1.6× stretch), the local `so101-real` mp4s, or `configs/robot_so101_template.yaml`'s workspace numbers (0.70×0.50 m, never validated).

---

## 3. V6 milestone definition

**V6 passes when:** for the SO-101 scene, there exists a (documented, reproducible) combination of arm-base placement + towel placement + gripper approach at which a jaw reaches within the **grasp threshold of the towel edge**, and a meaningful **fraction of the towel edge lies inside the reachable workspace** — *before any contact is attempted*.

Numeric success thresholds (tune against the real URDF once known):

| Metric | Symbol | Pass threshold | Rationale |
|---|---|---|---|
| Achievable min jaw-to-nearest-edge distance | `d_min` | **≤ 0.02 m** (stretch: ≤ 0.01 m) | A parallel jaw must close on the edge; 0.406 m today is 20× too far. |
| Fraction of towel edge within reachable workspace | `edge_reach_frac` | **≥ 0.30** | Enough of an edge to pick and fold, not a single lucky point. |
| Reachable grasp poses found for the fold's grasp side | `n_grasp_poses` | **≥ 1** valid, collision-free IK solution | You only need one good grasp to start; more is better. |
| Jaw approach clears the table & other arm | `collision_free` | **true** | A reachable-but-colliding pose is not usable. |

**V6 fails** if no placement in the searched space achieves `d_min ≤ threshold` — which then tells you *which* fix is needed (see §6 decision tree).

---

## 4. Scripts to create (Brev conventions; run inside Isaac on the VM)

Land the reusable logic in `terafold/isaac/reachability.py` (import-guarded) and thin runnable entry points in `scripts/tera/`. All output is machine-readable JSON so results are diffable and greppable. Follow the existing `STATUS.json` convention.

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/tera/inspect_so101_frames_v6.py` | Dump ground-truth geometry from the loaded scene: each SO-101 base frame, each jaw/EE world pose at the home pose and across a coarse joint-space grid, and the towel edge/corner particle positions. | Built SO-101 clean-towel USD/scene | `v6_geometry.json` (base frames, EE poses per joint sample, towel corner+edge particle XYZ) |
| `scripts/tera/reachability_sweep_v6.py` | For the current placement, sweep joint space (or run IK toward sampled towel-edge points) and compute `d_min`, `edge_reach_frac`, `n_grasp_poses`, `collision_free`. | `v6_geometry.json`, gripper/grasp-threshold config | `v6_reachability.json` + `tera_checkpoints/clean_towel_v6/STATUS.json` |
| `scripts/tera/placement_search_v6.py` | If the current placement fails, grid/random-search over towel placement (x,y,yaw) and, secondarily, arm-base placement, re-scoring reachability for each candidate; rank by `d_min` then `edge_reach_frac`. | scene + search bounds | `v6_placement_candidates.json` (ranked), best candidate highlighted |
| `scripts/tera/visualize_reachability_v6.py` | Render/overlay the reachable workspace vs the towel edge (and log to Rerun) so a human can see the gap and the best candidate. | `v6_reachability.json`, `v6_placement_candidates.json` | `.rrd` trace + PNG overlays |

Reuse across all four: `terafold/math/geometry.py` for edge/crease/target math, `terafold/eval/*`/`physics/fold_quality.py` for scoring, `terafold/yam/rerun_logger.py`'s pattern for the `.rrd`. Add a `width_reduction`/`d_min` metric key where the eval aggregators expect one (it does not exist yet).

**Diagnostic data every V6 run should emit** (into `v6_reachability.json`): SO-101 base world transforms; jaw & EE world positions across the joint-space samples (the reachable point cloud); towel edge + corner particle world positions; the reachable-workspace ∩ towel-edge set; per-candidate `d_min`, `edge_reach_frac`, `n_grasp_poses`, `collision_free`; and the single best (placement, grasp pose) found. Keep the raw numbers, not just pass/fail — the numbers drive the decision tree.

---

## 5. Runtime environment

All V6 scripts run **on the Brev VM inside the Isaac interpreter** (`./python.sh` in the NGC `isaac-sim` container). They need the built SO-101 clean-towel scene, Isaac's SO-101 articulation, and an IK/collision facility (cuRobo or Isaac Lab). Nothing V6 runs on the Mac (no Isaac) or needs a physical robot. Import-guard so `terafold/isaac/reachability.py` still *imports* on a Mac for unit-testing the pure-math helpers.

---

## 6. Decision tree after V6

**If V6 PASSES** (a reachable grasp pose exists):
→ Proceed to a minimal **V7 single-grasp contact test** at the found pose: command the jaw to that grasp, close, and verify the towel particle moves (edge displacement > threshold). *Only now* is contact justified. Then re-run the Brev V5 contact-fold from the validated placement. Wire this behind a real safety unlock (do not bypass the gate — see [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md) §6).

**If V6 FAILS**, the metrics tell you which fix:
- **`d_min` improves a lot when the towel moves** (placement-limited) → move the towel to the best `v6_placement_candidates.json` pose and re-run V6. Cheapest fix; try first.
- **No towel placement gets `d_min` low enough** (arm-reach-limited — consistent with 0.366 m < 0.406 m) → **reposition the arm bases / mounts** in the scene (closer to the table, or a taller mount) and re-run. The scene arm bases (`~0.48 m` back, per the local YAM `scene_config.json`) may simply be too far — check the Brev scene's actual base placement first.
- **A reachable pose exists but the jaw cannot close on the edge / collides** (gripper-geometry-limited) → inspect jaw collision geometry, finger length/travel, and approach angle; adjust the grasp approach in `terafold/isaac/reachability.py`.
- **Reachable but IK is unstable/singular** → add joint-limit-aware seeding / multiple IK restarts (the pattern already exists in `fold_sim.py::_ik`).

Re-run V6 after each change until it passes; log every attempt's `STATUS.json` so the search is auditable.

---

## 7. Hard "do NOT do yet" list

- **No new policy training.** Nothing is reachable to fold; training a V4/V7 policy before reachability is proven repeats the V4 result (width unchanged 0.68). The blocker is geometry.
- **No real-hardware contact** — SO-101 cannot even be driven from this repo (adapter raises `NotImplementedError`), and the Waveshare arm has been dark since 2026-06-26. V6 is Isaac-only.
- **No autonomy claims.** A passing V6 proves *reachability*, not folding. Do not describe any V6/V7 result as autonomous folding.
- **No local-sim reachability as evidence.** The 1.6× link stretch (`fold_sim.py:981`) invalidates it.
- **No repo-wide refactor mid-integration.** Land V6 as an additive `terafold/isaac/` module + `scripts/tera/` scripts; defer the cli.py cleanup and YAM/SO-101 reconciliation to a separate pass.
- **No bypassing the safety gate** for the eventual contact test — wire it behind a real unlock.

---

## 8. Definition of done (V6)

`tera_checkpoints/clean_towel_v6/STATUS.json` exists and records, reproducibly: a placement where `d_min ≤ 0.02 m`, `edge_reach_frac ≥ 0.30`, `n_grasp_poses ≥ 1` (collision-free), plus the full `v6_geometry.json` / `v6_reachability.json` / `v6_placement_candidates.json` diagnostics and a `.rrd`/PNG visualization — **or**, on failure, a documented conclusion naming which fix (placement / arm-base / gripper-geometry) the metrics point to, so V6 can be re-run after that change. Commit the small STATUS.json into the repo (`docs/brev_status/`) so the result is not stranded on the VM.
