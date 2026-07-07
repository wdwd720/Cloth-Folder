/goal

Continue from current TeraFold commit feb992a.

Do not redo Modal setup.
Do not redo V11/V12/V13/V14 from scratch.
Do not use Brev.
Do not run fake training.
Do not overclaim production readiness.

Current milestone:
- V14 solved real SO-101 articulation-driven folding.
- Normal stiff towel fold succeeded.
- Best trial: jaw-mounted disk tool, radius 0.14, gentle press + slow inward drag.
- Width 0.68 -> 0.509.
- Real demos collected: 12/12 valid.
- Training v2 ran with robot_contact_data_used=true.
- Eval v2 ran: 6/6 reduce width, 3/6 clean valid folds, 3/6 fold-but-ballistic.
- autonomous_so101_fold_policy_validated = true with caveats.
- Main caveats:
  1. policy uses progress/phase clock, so it may be trajectory imitation, not state-reactive control.
  2. 3/6 eval rollouts over-drive into ballistic velocity.
  3. demo diversity is small.
  4. jaw-mounted disk tool is large compared with raw fingertip.

Primary objective:
V16/V17 must turn the current working robot-contact policy into a more robust, state-reactive, non-ballistic fold policy.

Track A — V16 state-reactive dataset and observation cleanup

Create/update:
- scripts/tera/run_so101_real_contact_demo_collection_v16.py
- scripts/tera/validate_so101_real_contact_demos_v16.py
- scripts/tera/summarize_so101_demo_dataset_v16.py

Requirements:
- Collect more real SO-101 articulation-driven demos using the proven V14 fold mechanism.
- Use actual joint targets/actions.
- Store demos only on Modal Volume.
- Randomize:
  - towel x/y start offset
  - edge target offset
  - press depth
  - drag distance
  - drag speed
  - small contact patch offsets
- Collect at least 30 demos if runtime allows, minimum 20 valid.
- Include both successes and controlled failures if useful.
- Label ballistic/fake trials and exclude from success demos.
- Save:
  /artifacts/contact_v16/demos/so101_real_contact_demos_v16.npz
  /artifacts/contact_v16/contact_v16_demo_dataset_summary.json

Observation cleanup:
- Create a state-reactive observation schema that does NOT include explicit progress/phase clock.
- Allowed observations:
  - joint positions
  - joint target/action history if needed
  - towel width/edge metrics if available in sim
  - contact patch pose
  - relative jaw-to-edge features
  - recent velocity/contact features
- Disallowed:
  - explicit normalized timestep
  - phase/progress variable that tells the policy where it is in the scripted trajectory

Track B — Training v3 state-reactive policy

Create:
- modal_apps/terafold_policy_training_v3_state_reactive.py

Requirements:
- Train on V16 demos.
- robot_contact_data_used = true.
- no_phase_clock = true.
- Add action smoothing / velocity penalty / effort penalty to reduce ballistic behavior.
- Save checkpoint only to Modal Volume:
  /artifacts/training_v3_state_reactive/policy_v3_state_reactive.pt
- Save summary:
  /artifacts/training_v3_state_reactive/training_v3_state_reactive_summary.json

Training JSON must include:
- robot_contact_data_used
- no_phase_clock
- num_episodes
- train_steps
- initial_loss
- final_loss
- validation_loss
- action_mae
- smoothness_penalty
- checkpoint_path
- final_robot_contact_policy_training_ready
- remaining_blockers

Track C — Eval v3 robust closed-loop

Create:
- modal_apps/terafold_policy_eval_v3_state_reactive.py

Requirements:
- Evaluate V3 policy closed-loop in Isaac on Modal.
- Use real SO-101 articulation actions.
- No direct particle movement.
- No progress/phase clock.
- Run at least 20 eval episodes if runtime allows, minimum 10.
- Randomize starts.
- Reject ballistic false positives.
- Save:
  /artifacts/eval_v3_state_reactive/eval_v3_state_reactive_summary.json

Eval success criteria:
- success_rate_clean_non_ballistic >= 0.8 target
- minimum acceptable improvement: better than v2 clean 0.5
- mean width_after < 0.55
- ballistic_rate < 0.2
- all results honestly labeled

Track D — Optional V17 improvements if time remains

Only after V3 eval:
- left-arm symmetry
- two-arm sequence
- smaller contact tool sweep
- more demos
- longer training
- action filtering / MPC wrapper
- hybrid scripted edge-capture + learned residual

Track E — docs and commits

Update:
docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md

Include:
- V16 dataset verdict
- Training v3 verdict
- Eval v3 verdict
- clean success rate
- ballistic rate
- whether no-phase-clock policy works
- whether final_robot_contact_policy_training_ready remains true
- whether autonomous_so101_fold_policy_validated is robust or caveated
- exact next step

Commit source/docs only:
- scripts/tera/*v16*.py
- modal_apps/*v3*.py
- docs updates

Do not commit:
- data/
- runs/
- tera_checkpoints/
- .pt
- .ckpt
- .npz
- .rrd
- .usd
- .usda
- .usdc
- Modal artifacts

Autonomous loop:
- inspect repo
- patch
- py_compile
- run Modal jobs
- pull/read JSONs
- fix up to 3 serious attempts per track
- commit/push source/docs
- stop with:
  final commit
  V16 dataset verdict
  Training v3 verdict
  Eval v3 verdict
  clean success rate
  ballistic rate
  whether no-phase-clock policy works
  exact next step

Start now.
---

## V16 / v3 SPRINT OUTCOME (recorded 2026-07-07)

Primary objective MET: the working robot-contact policy is now genuinely
state-reactive (no phase/progress clock) and less ballistic than v2. Stretch bars
partially met; honest caveats below.

- V16 dataset verdict: PASS. 21 valid diverse real SO-101 articulation-driven fold
  demos (>=20 required); 11 ballistic + 0 no-fold labeled & EXCLUDED; 4802 samples,
  state_dim=72, no_phase_clock=true, validator all-14-checks pass (incl. no clock
  feature). Valid demos fold width 0.68 -> 0.564. robot_contact_data_ready=true. Volume-only.
- Training v3 verdict: PASS. robot_contact_data_used=true, no_phase_clock=true,
  num_episodes=21, train_steps=4000, loss 0.823 -> 0.000288, val 0.000322,
  action_mae 0.00126, smoothness_penalty 0.00042 (anti-ballistic velocity penalty).
  final_robot_contact_policy_training_ready=true.
- Eval v3 verdict (closed-loop, NO clock, real articulation, randomized starts, ballistic rejected):
  RAW policy over two 20-episode runs = clean 0.90 and 0.60 (combined 30/40 = 0.75),
  ballistic 0.10 and 0.35 (combined 9/40 = 0.225); ALL rollouts reduce width; mean
  width_after ~0.58 (best clean fold 0.482). Action-filter mitigation (EMA beta=0.6,
  30 ep) = ballistic 0.067 but gentler folds (width_after 0.652, clean 0.50).
- clean success rate: ~0.75 avg (0.60-0.90 across runs) — improves on v2's 0.5.
- ballistic rate: ~0.225 raw avg (0.10-0.35, unstable) — down from v2's ~0.5; 0.067 with the action filter.
- no-phase-clock policy works: YES — the policy folds autonomously closed-loop with a
  pure state observation (no progress/phase). The #1 v2 caveat is resolved.
- Honest caveats (not robust): (1) GPU physics is non-deterministic, so the ballistic
  tail is unstable run-to-run (0.10-0.35); (2) clean folds are gentle (~0.58); the
  deepest folds (<0.55) COUPLE with ballistic over-drive, so mean width_after<0.55 is
  not met without over-driving; the demos themselves only cleanly fold to ~0.56.
- autonomous_so101_fold_policy: robust on state-reactivity, improved on non-ballistic,
  CAVEATED on fold-depth robustness (width_after ~0.58 > 0.55) and run-to-run variance.

Exact next step: V17 — collect a demo set of DEEP-but-SLOW drags (drag_dist ~0.20-0.28
over ~180-210 steps) to teach clean deep folds and break the deep<->ballistic coupling;
retrain v3 and re-eval over multiple seeds (report the distribution, not one run); tune the
action filter to beta~0.3-0.4 for a ballistic/depth balance; optionally add left-arm
symmetry / two-arm sequence / a smaller contact tool.
