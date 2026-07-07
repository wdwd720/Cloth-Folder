/goal

Continue from current TeraFold commit 020b895.

Do not redo Modal setup.
Do not redo V11-V16.
Do not use Brev.
Do not fake success.
Do not commit data/checkpoints/artifacts.

Current status:
- Real SO-101 articulation-driven folding works.
- V16 collected 21 valid diverse real demos.
- Training v3 used robot_contact_data_used=true and no_phase_clock=true.
- Eval v3 proved a no-phase-clock policy can fold closed-loop.
- Clean success rate improved to about 75%.
- Ballistic rate dropped vs v2 but remains unstable.
- Mean clean width is not deep enough.
- Main problem: deep folds and ballistic over-drive are coupled.
- Exact next step: collect deep-but-slow demos and retrain.

Primary objective:
V17 must improve robustness by collecting clean deep-fold demos, retraining the no-phase-clock policy, and evaluating across multiple seeds.

Track A — V17 deep-but-slow demo collection

Create/update:
- scripts/tera/run_so101_deep_slow_demo_collection_v17.py
- scripts/tera/validate_so101_deep_slow_demos_v17.py
- scripts/tera/summarize_so101_deep_slow_dataset_v17.py

Requirements:
- Use the proven V14/V16 SO-101 articulation fold mechanism.
- Use real joint targets/actions.
- No phase/progress clock in observation.
- Store demos only on Modal Volume.
- Collect at least 30 valid demos if runtime allows, minimum 20.
- Focus on clean deep folds:
  - drag_dist around 0.20 to 0.28 m
  - drag duration around 180 to 210 steps
  - slower action changes
  - no velocity cap / ballistic false positives
- Randomize:
  - towel start offsets
  - edge target offsets
  - press depth
  - drag distance
  - drag duration
  - contact patch offsets
- Label and exclude ballistic demos.
- Save:
  /artifacts/contact_v17/demos/so101_deep_slow_demos_v17.npz
  /artifacts/contact_v17/contact_v17_demo_dataset_summary.json

Success criteria:
- at least 20 valid non-ballistic demos
- mean width_after below V16 demo mean
- target mean width_after < 0.55 if possible
- robot_contact_data_ready=true
- no_phase_clock=true

Track B — Training v4 no-clock robust policy

Create:
- modal_apps/terafold_policy_training_v4_deep_slow.py

Requirements:
- Train on V17 demos.
- robot_contact_data_used=true.
- no_phase_clock=true.
- Add stronger anti-ballistic regularization:
  - action smoothing penalty
  - velocity/effort penalty if available
  - maybe action delta penalty
- Save checkpoint only to Modal Volume:
  /artifacts/training_v4_deep_slow/policy_v4_deep_slow.pt
- Save:
  /artifacts/training_v4_deep_slow/training_v4_deep_slow_summary.json

Training JSON:
- num_episodes
- train_steps
- initial_loss
- final_loss
- validation_loss
- action_mae
- smoothness_penalty
- robot_contact_data_used
- no_phase_clock
- checkpoint_path

Track C — Eval v4 multi-seed robustness

Create:
- modal_apps/terafold_policy_eval_v4_deep_slow.py

Requirements:
- Evaluate closed-loop in Isaac on Modal.
- Real SO-101 articulation actions only.
- No direct particle movement.
- No phase/progress clock.
- Run at least 3 seeds.
- Minimum 10 episodes per seed.
- Report distribution, not just one run.
- Test raw policy and EMA action filter.
- Try EMA beta values:
  - 0.3
  - 0.4
  - 0.6
- Save:
  /artifacts/eval_v4_deep_slow/eval_v4_deep_slow_summary.json

Eval target:
- clean_success_rate >= 0.8
- ballistic_rate < 0.15
- mean width_after < 0.55
- every result honestly labeled

Track D — Optional if time remains

Only after V4 eval:
- left-arm symmetry
- two-arm sequence
- smaller contact disk/tool sweep
- learned residual over scripted fold
- compare V3 vs V4 on same seeds

Track E — docs and commits

Update:
docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md

Include:
- V17 dataset verdict
- Training v4 verdict
- Eval v4 verdict
- clean success distribution by seed
- ballistic distribution by seed
- best EMA beta
- whether policy is robust or still caveated
- exact next step

Commit source/docs only:
- scripts/tera/*v17*.py
- modal_apps/*v4*.py
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
  V17 dataset verdict
  Training v4 verdict
  Eval v4 verdict
  clean success rate
  ballistic rate
  mean width_after
  best EMA beta
  whether robust autonomous SO-101 folding is validated
  exact next step

Start now.
---

## V17 / v4 SPRINT OUTCOME (recorded 2026-07-07)

HONEST NEGATIVE result: the deep-but-slow experiment did NOT improve robustness. V3
(the V16-demo policy) remains the best VALIDATED state-reactive policy.

- V17 dataset verdict: PASS, deeper than V16. 23 valid / 11 ballistic / 0 no-fold, 6554
  samples, no_phase_clock=true, validator all-14-pass. Valid folds 0.68->0.558 (min 0.498,
  11/23 below 0.55), beats_v16_demo_mean=true. robot_contact_data_ready=true. Volume-only.
- Training v4 verdict: PASS (attempt 2). Attempt 1 (W_smooth=1.0 + effort penalty toward
  a_mean) OVER-DAMPED the policy (eval clean 0.27, betas killed folding). Attempt 2 reverted
  to v3's recipe (W_smooth=0.5, no effort): bc_loss 8.6e-5, action_mae 0.00118, no_phase_clock=true.
- Eval v4 verdict (multi-seed x multi-beta, no clock): NEGATIVE. raw@340steps clean 0.40
  ([0.7,0.5,0.0] by seed), ballistic 0.30, mean width_after 0.599; EMA beta>=0.3 -> no fold.
  240-step horizon was too short for the ~290-step slow demos (raw clean 0.067 -> 0.40 at 340).
- clean success rate: 0.40 (raw, best beta) -- WORSE than v3's ~0.75.
- ballistic rate: 0.30 (raw) -- worse than v3's ~0.22.
- mean width_after: 0.599 (raw) -- not < 0.55; deep only on 2/3 seeds (min 0.42), 1 seed folds nothing.
- best EMA beta: 0.00 (raw); all filters over-damped the slow policy to no-fold.
- whether robust autonomous SO-101 folding is validated: NO. robust_autonomous_so101_folding_validated=false.
  V3 remains the best validated state-reactive policy.

Exact next step: keep V3 deployed. For deep AND robust: collect MODERATE-speed deep demos
(drag 0.24-0.28 m over ~140-160 steps -- deep but decisive, not slow) so the policy is deep
and fast enough for a normal horizon; or go beyond plain BC (DAgger / learned residual over the
scripted deep fold) to fix off-distribution seeds; match eval horizon to demo length; drop pure
EMA filtering (it over-damps a slow policy).
