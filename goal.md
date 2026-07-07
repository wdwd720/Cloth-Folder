/goal

You are Claude Code operating autonomously inside the local Mac repo:

~/Cloth-Folder

Primary repository:
https://github.com/wdwd720/Cloth-Folder.git

Project:
Tera Robotics / TeraFold — simulated robot towel folding using SO-101 arms, Isaac Sim / IsaacLab, Modal GPUs, and policy training.

================================================================================
SLEEP-MODE AUTONOMOUS ENGINEERING DIRECTIVE
================================================================================

The user is going to sleep. Your job is to run for a long autonomous sprint, up to about 6 hours, without asking command-by-command questions.

You may:
- inspect files
- edit code
- create scripts
- run shell commands
- run Modal jobs
- pull Modal Volume JSON artifacts
- debug failures
- patch and rerun
- create docs
- commit clean source/docs changes
- push to origin/main

You must:
- work recursively toward full autonomous SO-101 robot-contact training readiness
- keep going through fixable failures
- stop only when a real milestone is reached or a hard blocker is proven
- document everything honestly
- never fake training readiness
- never call synthetic/proxy training “real robot-contact training”
- never commit generated artifacts or large data

Maximum autonomous runtime:
- Work for up to 6 hours.
- Use bounded attempts per stage.
- Do not run infinite loops.
- Do not burn GPU forever.
- Prefer many short evidence-producing jobs over one huge opaque job.
- Every long Modal job must write a JSON summary to the Modal Volume.

At the end, leave a clean final report in:
docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md

Also print a final terminal summary with:
- final git commit
- what passed
- what failed
- whether robot_contact_data_ready is true or false
- whether Training v2 ran with robot_contact_data_used=true
- whether final_robot_contact_policy_training_ready is true or false
- exact next step

================================================================================
CURRENT PROJECT STATE
================================================================================

Known current commit from previous sprint:
2d922a3 — "V13: real SO-101 articulation reach + contact (fold not yet achieved; honest)"

Important previous commits:
869c4a7 — V12 ISOLATED: SOLVE SO-101 contact transfer
7e80c8e — V11: SOLVE contact-transfer blocker
b3cc2a5 — Add Modal policy training v1
e00feb3 — V12: SO-101 physics-drive port execution blocked
fbd7e99 — Record V12 sprint outcome in goal

Current source-of-truth:
- Mac repo: ~/Cloth-Folder
- GitHub: origin/main
- Modal workspace: terarobotics
- Modal Volume: terafold-artifacts

Brev:
- Brev is retired.
- Do not use Brev unless absolutely required for historical recovery.

Modal:
- Modal GPU PyTorch works.
- Modal Volume artifact persistence works.
- Isaac Sim 4.5 / IsaacLab path works.
- IsaacLab RL packaging blocker was fixed using import-only stubs + h5py + per-container isolation.
- SO-101 real scene builds on Modal.

Artifact policy:
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
- generated datasets
- generated checkpoints

Commit only:
- source code
- Modal app scripts
- small docs
- small status JSON only if explicitly source/status, not generated artifacts

================================================================================
ENGINEERING TRUTH SO FAR
================================================================================

V11:
- Solved primitive rigid-contact transfer.
- Root cause of V8–V10 failure:
  moving kinematic collider by USD transform teleport produced no swept velocity, so PhysX imparted no impulse to PBD cloth particles.
- Fix:
  drive contact geometry using physics API / kinematic target, specifically RigidPrim.set_world_poses or equivalent physics-resolved motion.
- Valid primitive trial:
  teleport baseline inert
  physics-driven proxy moved cloth
  edge displacement around 0.081 m
  width 0.68 -> 0.63
  particle velocity spike around 0.83 m/s
- Dynamic ballistic false positives were excluded.

V12 isolated:
- Fixed IsaacLab RL packaging blocker.
- Real SO-101 scene builds on Modal.
- Jaw-attached proxy/contact patch can transfer contact when physics-driven.
- SO-101 proxy fold succeeded at mechanism/proxy level:
  teleport baseline inert
  physics set_world_poses proxy folded towel
  edge displacement around 0.093 m
  width 0.68 -> 0.62
- Honest caveat:
  this used a jaw-attached contact-patch/proxy, not real joint-action demo data.
- robot_contact_data_ready remained false.

V13:
- Real SO-101 articulation reach solved.
- Real right arm driven only by physics-resolved joint position targets:
  set_joint_position_target -> write_data_to_sim -> sim.step
- Reach result:
  reachable_edge_found = true
  best jaw-to-edge = 0.0285 m
- Arm-attached contact patch touches cloth:
  touch = 0.0 m
  particle velocity peak = 0.62
- But actual fold failed:
  edge displacement plateau around 0.008 m
  width 0.68 -> 0.68
- The patch presses/slides over the stiff towel instead of catching and dragging the edge.
- robot_contact_data_ready = false
- Training v2 was correctly skipped.

Current high-level status:
- training_infra_ready = true
- contact_transfer_mechanism_solved = true
- SO-101 scene and articulation reach = true
- SO-101 arm-driven fold = false
- real robot-contact demos = false
- final_robot_contact_policy_training_ready = false

================================================================================
PRIMARY OBJECTIVE FOR THIS SLEEP SPRINT
================================================================================

Advance from V13 to full real robot-contact training readiness.

The critical missing milestone:
Make the real SO-101 articulation catch the towel edge and produce a valid arm-driven fold, then collect real robot-contact demos, then run Training v2 using robot_contact_data_used=true.

Do not run fake robot-contact training.
Do not call proxy-only demos real robot-contact demos.
Do not claim final policy readiness unless real joint-action demos exist and pass fold metrics.

================================================================================
STOP CONDITIONS
================================================================================

Stop successfully if all are true:
1. Real SO-101 articulation-driven fold succeeds.
2. Valid real robot-contact demos are collected.
3. Training v2 runs with robot_contact_data_used=true.
4. docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md is updated.
5. Source/docs are committed and pushed.
6. Working tree is clean or only goal.md remains intentionally modified.

Stop with hard blocker if:
1. After serious bounded attempts, real SO-101 fold still fails.
2. The blocker is documented with evidence.
3. Training v2 is honestly skipped.
4. docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md is updated.
5. Source/docs are committed and pushed.
6. Working tree is clean or only goal.md remains intentionally modified.

================================================================================
STARTUP CHECKLIST
================================================================================

First run:

- pwd
- git status --short
- git log --oneline --decorate -10
- find modal_apps -maxdepth 1 -type f | sort
- find scripts/tera -maxdepth 1 -type f | sort
- sed/read docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md
- inspect V11/V12/V13 scripts
- inspect modal apps used for Isaac 4.5 isolated SO-101 runs
- inspect modal_apps/terafold_policy_training_v2_robot_contact.py

If only goal.md is modified:
- Do not treat it as a blocker.
- Do not commit it unless you intentionally prepend a final session outcome at the end.
- Do not overwrite user changes.

If any other file is modified:
- inspect it
- decide whether to keep/commit/revert
- do not overwrite unknown user work blindly

================================================================================
TRACK A — V14 EDGE CAPTURE / HOOK / PINCH
================================================================================

Goal:
Turn the real SO-101 articulation contact into a valid arm-driven fold by catching the towel edge, not just pressing/sliding on top.

Create or update these scripts:
- scripts/tera/search_so101_edge_capture_v14.py
- scripts/tera/test_so101_outside_hook_drag_v14.py
- scripts/tera/test_so101_pinch_lift_drag_v14.py
- scripts/tera/run_so101_arm_fold_demo_v14.py
- scripts/tera/summarize_so101_fold_readiness_v14.py

Reuse existing V13 utilities. Do not rewrite the whole stack.

V14 core idea:
V13 pressed/dragged near the edge center and plateaued around 8 mm. The next strategy is edge capture:
- approach from outside the towel edge
- move inward under/around the edge lip
- optionally close/pinch the edge
- lift slightly
- drag toward center
- use real SO-101 articulation targets
- contact patch must be attached to jaw/link, not a free world proxy

Required V14 experiments:

Experiment 1 — V13 baseline replay:
- reproduce V13 press/drag behavior
- expected result:
  edge displacement around 0.008 m
  width unchanged
- purpose:
  confirm baseline is stable and comparable

Experiment 2 — outside-hook drag:
- start jaw/contact patch just outside towel edge using outside_margin
- approach slightly below/at edge lip if feasible
- move inward to catch the edge
- drag toward center
- sweep:
  outside_margin
  vertical offset
  patch offset relative to jaw
  patch radius/shape
  friction
  drag speed
  drag distance

Experiment 3 — pinch-lift-drag:
- use gripper/jaw closing if available
- create opposing contact patches if needed, but they must be attached to jaw/finger links or driven by articulation, not free proxies
- close around edge
- lift slightly
- drag inward
- compare to no-pinch and no-lift

Experiment 4 — cloth stiffness controlled comparison:
- normal towel remains the main target
- optionally create soft-cloth variant:
  reduced stretch stiffness
  reduced bend stiffness
  adjusted damping/friction
- label clearly:
  soft_cloth_variant = true
- do not count soft-only success as final normal-cloth success
- soft success can guide geometry/motion

Experiment 5 — patch geometry sweep:
- radius 0.02, 0.03, 0.04, 0.05, 0.06
- offset from jaw
- maybe capsule/cylinder-like patch if easy
- friction sweep
- avoid ballistic/fake success

Strict success criteria:
- Motion driven by real SO-101 articulation/joint targets.
- Contact geometry attached to jaw/link or equivalent robot body.
- No free world proxy success.
- No direct particle edits.
- No USD teleport success.
- actual_touch_distance < 0.01 m
- edge_displacement > 0.02 m
- width_after < width_before
- stretch goal width_after < 0.62
- particle_velocity_peak > no-contact baseline
- no velocity cap / ballistic false positive
- no synthetic-only labels counted as real demos
- robot_contact_data_ready = true only if the episode includes real state + joint_action + contact metrics and passes fold criteria

Save:
- /artifacts/contact_v14/contact_v14_fold_summary.json

V14 JSON must include:
- status
- best_trial
- all_trials table/list
- normal_cloth_success
- soft_cloth_success
- arm_driven_fold_solved
- actual_touch_distance_m
- edge_displacement_m
- width_before_m
- width_after_m
- particle_velocity_peak_mps
- no_contact_baseline_velocity_mps
- ballistic_or_fake_success_excluded
- contact_geometry_attached_to_robot
- free_proxy_used_for_success
- real_joint_actions_used
- robot_contact_data_ready
- final_robot_contact_policy_training_ready
- remaining_blocker
- exact_next_step

Run on Modal using the existing Isaac 4.5 isolated pattern. Use per-container isolation if needed.

Attempt budget:
- Up to 3 serious V14 attempts.
- Each attempt should make a principled change.
- Do not randomly tune forever.

================================================================================
TRACK B — V15 REAL ARM-DRIVEN DEMO COLLECTION
================================================================================

Only start V15 if V14 normal-cloth arm-driven fold succeeds.

Goal:
Collect real robot-contact demonstration episodes using actual SO-101 joint/action sequences.

Create:
- scripts/tera/run_so101_real_contact_demo_collection_v15.py
- scripts/tera/validate_so101_real_contact_demos_v15.py
- scripts/tera/summarize_so101_demo_dataset_v15.py

Demo requirements:
- Each episode must include:
  - initial towel metrics
  - final towel metrics
  - state observations
  - joint position targets/actions
  - jaw/contact metrics
  - actual_touch_distance
  - edge displacement
  - width before/after
  - particle velocity peak
  - success/failure label
- Store demos only in Modal Volume:
  /artifacts/contact_v15/demos/
- Do not commit demo data.
- Collect 10 to 50 episodes depending on runtime.
- Include perturbations:
  - slight towel position shifts
  - slight edge target offsets
  - drag distance/speed variation
  - maybe left/right arm variants if easy
- Validate demos before training.

V15 success:
- At least 10 valid fold demos
- robot_contact_data_ready = true
- no fake/proxy-only demos
- no direct particle movement
- real_joint_actions_used = true

Save:
- /artifacts/contact_v15/contact_v15_demo_dataset_summary.json

V15 JSON must include:
- num_episodes_total
- num_valid_success
- num_valid_failure
- robot_contact_data_ready
- dataset_path
- action_schema
- observation_schema
- metrics_schema
- mean_edge_displacement
- mean_width_reduction
- success_rate
- final_robot_contact_policy_training_ready

If V14 fails:
- Do not run V15.
- Write V15 skipped reason in docs/status.

================================================================================
TRACK C — TRAINING V2 REAL ROBOT-CONTACT POLICY
================================================================================

Only run Training v2 if V15 demos exist and validate.

Use or update:
- modal_apps/terafold_policy_training_v2_robot_contact.py

Goal:
Train first real robot-contact policy from actual SO-101 joint-action demos.

Requirements:
- Modal L40S
- PyTorch CUDA
- Read demos from Modal Volume
- robot_contact_data_used = true
- final policy label must reflect demo quality honestly
- bounded run, not overnight runaway
- save checkpoint only to Modal Volume
- save JSON summary to:
  /artifacts/training_v2_robot_contact/training_v2_robot_contact_summary.json

Training v2 JSON must include:
- status
- cuda_available
- gpu_name
- robot_contact_data_used
- dataset_path
- num_episodes
- train_steps
- initial_loss
- final_loss
- action_mae
- validation_loss
- validation_action_mae
- checkpoint_path
- policy_architecture
- observation_schema
- action_schema
- ready_for_real_training
- final_robot_contact_policy_training_ready
- remaining_blockers
- exact_next_step

Do not run Training v2 if:
- demos are missing
- demos are proxy-only
- fold failed
- robot_contact_data_ready = false

If Training v2 is skipped, write:
- robot_contact_data_used = false
- training_v2_skipped_reason = "real SO-101 arm-driven fold demos not ready"
- final_robot_contact_policy_training_ready = false

================================================================================
TRACK D — EVALUATION V2 / CLOSED-LOOP TEST
================================================================================

Only run this if Training v2 succeeds.

Create:
- modal_apps/terafold_policy_eval_v2_robot_contact.py
or scripts/tera/eval_so101_robot_contact_policy_v2.py as appropriate.

Goal:
Evaluate trained robot-contact policy in Isaac on Modal.

Requirements:
- Load checkpoint from Modal Volume
- Run at least 3 to 10 eval episodes
- No direct particle movement
- Real SO-101 articulation actions
- Save JSON:
  /artifacts/eval_v2_robot_contact/eval_v2_robot_contact_summary.json

Eval JSON:
- num_eval_episodes
- success_count
- success_rate
- mean_width_before
- mean_width_after
- mean_edge_displacement
- failures
- final_robot_contact_policy_training_ready
- autonomous_fold_policy_validated

Honest interpretation:
- If policy only imitates demo but does not fold in rollout, say so.
- If open-loop demo replay works but learned policy fails, say so.
- Do not claim autonomous policy success without eval.

================================================================================
TRACK E — OPTIONAL IMPROVEMENT LOOPS
================================================================================

Only after V14/V15/V2 exists, use remaining time for improvement.

Possible improvements:
1. Increase demo count from 10 to 50.
2. Add randomized towel starts.
3. Add left-arm symmetry.
4. Add two-arm fold sequence:
   - right edge capture
   - left edge capture
   - center align
5. Train v2 longer or with validation split.
6. Evaluate policy.
7. Make docs clearer.

Do not do these before real demos exist.

================================================================================
MODAL / SOURCE IMPLEMENTATION GUIDELINES
================================================================================

Use tiny source tarballs:
- git archive --format=tar.gz --output=/tmp/terafold_modal_src.tar.gz HEAD

Avoid huge tars from working tree.

If using local files in Modal:
- use add_local_file(..., copy=True)
or add local file last according to Modal requirements.

Isaac Sim:
- Use the Isaac Sim 4.5 environment that worked for V12/V13.
- Use IsaacLab version/path that worked in prior scripts.
- Use RL import stubs and h5py fix.
- Use per-container isolation:
  one SimulationApp per Modal function/container when possible.
- Share artifacts through Modal Volume.
- Do not run multiple SimulationApp sessions in one process if it causes Kit config poisoning.

Python:
- Run py_compile before every commit:
  python3 -m py_compile modal_apps/*.py scripts/tera/*.py
or targeted py_compile if full compile is too slow.
- Make scripts print final JSON summaries when reasonable.

Git:
- Commit source/docs only after tests.
- Push to origin/main.
- Do not rewrite history.
- Do not force push.

================================================================================
FAILURE HANDLING
================================================================================

Do not stop at first failure.

For each track:
- inspect logs
- identify root cause
- patch
- rerun
- save evidence

But do not loop forever:
- V14 max 3 serious attempts
- V15 max 2 serious attempts
- Training v2 max 2 serious attempts
- Eval max 2 serious attempts

Common V14 failure interpretations:
- reach insufficient
- edge capture geometry bad
- contact patch too smooth/small
- jaw cannot hook under edge
- towel stiffness too high
- friction insufficient
- patch slides on top
- gripper cannot pinch due geometry
- articulation path misses edge
- robot-link collision constraints

For every failure, record:
- exact trial
- metrics
- why it failed
- next best fix

================================================================================
DOCUMENTATION
================================================================================

Always update:
docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md

Include a final table:
item | status | evidence | blocker | next step

Required items:
- Modal GPU PyTorch
- Modal Volume
- Modal Isaac
- IsaacLab/RL stubs
- V11 primitive contact
- V12 proxy SO-101 contact
- V13 articulation reach/contact impulse
- V14 arm-driven fold
- V15 real robot demos
- Training v2 robot-contact
- Eval v2
- final_robot_contact_policy_training_ready

Also update docs/MODAL_MIGRATION_PLAN.md only if the runtime or workflow changed.

At the very end, optionally prepend a session outcome block to goal.md, but only commit it if it is intended as session history and not just a directive. Do not let goal.md remain confusing.

================================================================================
FINAL REPORT FORMAT
================================================================================

When finished, print:

SLEEP SPRINT FINAL REPORT

Final commit:
<hash and message>

Runtime summary:
- start/end time if available
- major Modal jobs run
- artifacts written

V14 verdict:
- arm_driven_fold_solved:
- best trial:
- edge_displacement:
- width before/after:
- robot_contact_data_ready:

V15 verdict:
- demos collected:
- valid success demos:
- dataset path:
- robot_contact_data_ready:

Training v2 verdict:
- ran or skipped:
- robot_contact_data_used:
- final loss:
- validation MAE:
- checkpoint path:

Eval verdict:
- ran or skipped:
- success rate:
- autonomous policy validated:

Final authorization:
- final_robot_contact_policy_training_ready:
- autonomous_so101_fold_policy_validated:

Honest blocker:
<only if applicable>

Exact next step:
<one precise command or engineering goal>

================================================================================
START NOW
================================================================================

Begin with the startup checklist, then execute Tracks A through E in order.
Do not ask for user confirmation unless there is a hard external blocker.