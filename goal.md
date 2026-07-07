/goal

================================================================================
SESSION OUTCOME — V12 / Training-v2 sprint COMPLETE (stop condition B satisfied)
================================================================================
Final git commit: e00feb3 (branch main, pushed to origin/wdwd720/Cloth-Folder)

contact_v12 verdict:
- SO-101 real arm scene BUILDS on Modal (leisaac + downloaded so101_follower.usd;
  arm [base..jaw], joints [shoulder_pan..gripper], jaws (+-0.79,-0.24,0.37),
  2691-particle towel). Brev replaceable for SO-101 too.
- Drive audit DONE: arm = articulation targets (physics); V8 fingertip proxies =
  set_prim_translation teleport (the exact V11 defect). Correct fix = V11
  RigidPrim.set_world_poses on the proxies.
- so101_contact_transfer_solved = false: the physics-drive test did NOT execute.
  HARD BLOCKER (strong evidence, 7 runs): contact-drive scripts fatally fail at
  SimulationApp startup because IsaacLab isaaclab_rl auto-loads and imports
  rl_games/rsl_rl/sb3/skrl; installing them reinstalls a conflicting torch (cu13)
  that breaks Isaac Sim 4.5. Packaging/kit issue, NOT a contact-physics issue.

training_v2 verdict:
- SKIPPED honestly (gated). robot_contact_data_used = false,
  training_v2_skipped_reason = "V12 robot-contact demos not ready".
  NO fake robot-contact training was run.

final autonomous SO-101 policy training authorized? NO.
- final_robot_contact_policy_training_ready = false.
- The physics-drive MECHANISM is proven (V11, primitive collider: teleport fails,
  physics set_world_poses moves cloth, width 0.68->0.63). It is not yet a trainable
  SO-101 robot-contact policy (no valid robot-joint-action demos).

exact next step (also in docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md):
1. Provide lightweight sys.meta_path STUB packages for rl_games/rsl_rl/
   stable_baselines3/skrl so isaaclab_rl loads without pulling torch.
2. Isolate each SO-101 Isaac script in its own Modal container (fresh kit config);
   share state (towel USD, JSONs, demos) via the terafold-artifacts volume.
3. Re-run terafold_modal_isaac45_so101_v12.py -> expect teleport-fails vs
   physics-set_world_poses-works on the SO-101 fingertip proxy.
4. Then V6 reach + articulation fold on Modal -> real (state, joint_action) demos
   -> training_v2_robot_contact with robot_contact_data_used=true.

Evidence: Modal Volume terafold-artifacts (/contact_v12/*, /training_v2_robot_contact/*).
Status doc: docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md (full V12 table + blocker + next).
================================================================================

Continue from the current committed TeraFold state.

Current winning evidence:
- Modal is the source GPU/Isaac runtime.
- Brev is retired.
- Training v1 completed on Modal L40S.
- V11 solved the rigid-contact transfer blocker.
- Root cause: USD transform teleporting gave no swept velocity, so no impulse.
- Fix: physics-driven kinematic target using RigidPrim.set_world_poses moves cloth.
- Valid V11 trial:
  kinematic_velocity_slow
  edge_disp = 0.081 m
  width = 0.68 -> 0.63
  particle_velocity_peak = 0.83 m/s
- Dynamic ballistic trials are excluded as false positives.
- Final robot-contact policy is allowed only after converting this mechanism to the SO-101 gripper/articulation and generating valid robot-contact demonstrations.

Primary objective:
Build V12: port the solved V11 physics-driven contact mechanism from primitive sphere to SO-101 gripper/fingertips, generate valid robot-contact demonstration data, then start Training v2 using real robot-contact data if and only if the SO-101 contact metrics pass.

Do not redo Modal setup.
Do not redo PyTorch smoke.
Do not redo Isaac smoke.
Do not use Brev.
Do not commit artifacts, weights, data, USD, NPZ, PT, CKPT, RRD.

TRACK A — V12 SO-101 physics-driven contact

Goal:
Make the SO-101 gripper/fingertip move the towel using the same valid physics-driven contact method that solved V11.

Create/modify scripts under scripts/tera:

1. inspect_so101_drive_modes_v12.py
Purpose:
Inspect how SO-101 links/fingertips are currently driven.
Find whether current V6/V7 scripts are using teleport-like pose edits, articulation position targets, or actual physics-resolved motion.

2. test_so101_fingertip_physics_drive_v12.py
Purpose:
Drive the SO-101 fingertip/contact proxy through physics API / articulation targets / kinematic target equivalents, not USD teleport.
Compare against baseline teleport.

3. run_so101_contact_demo_collection_v12.py
Purpose:
If SO-101 contact works, collect small demo episodes:
- state/action/contact metrics
- towel width before/after
- edge displacement
- particle velocity near contact
Save demos only to Modal Volume, not git.

4. summarize_so101_contact_readiness_v12.py
Purpose:
Write a clear JSON summary:
- so101_contact_transfer_solved
- best_trial
- width_before
- width_after
- edge_displacement_m
- actual_touch_distance_m
- particle_velocity_peak
- valid_controlled_contact
- ballistic_or_teleport_excluded
- robot_contact_data_ready
- final_robot_contact_policy_training_ready

Success criteria:
- SO-101/proxy contact, not pure primitive-only sphere.
- actual_touch_distance < 0.01 m
- edge_displacement > 0.02 m
- width reduction from contact, ideally width_after < 0.63 first, stretch goal < 0.55
- particle velocity spike without hitting artificial cap
- no direct particle teleporting
- no ballistic fling counted as success
- clear baseline showing teleport fails and physics drive works

Save V12 JSON to:
/artifacts/contact_v12/contact_v12_summary.json

TRACK B — Training v2 robot-contact data

Goal:
Only if V12 produces valid SO-101 robot-contact demos, train a first robot-contact policy.

Create:
modal_apps/terafold_policy_training_v2_robot_contact.py

Requirements:
- Use Modal L40S.
- Read V12 demo data from Modal Volume.
- If V12 demo data is missing or invalid, do not fake it.
- If missing, write a JSON saying:
  robot_contact_data_used = false
  training_v2_skipped_reason = "V12 robot-contact demos not ready"
- If valid data exists:
  train policy on robot-contact demos
  save checkpoint to Modal Volume
  save summary JSON to:
  /artifacts/training_v2_robot_contact/training_v2_robot_contact_summary.json

Training v2 JSON must include:
- cuda_available
- gpu_name
- robot_contact_data_used
- dataset_path
- num_episodes
- train_steps
- initial_loss
- final_loss
- action_mae
- validation_metrics
- checkpoint_path
- final_robot_contact_policy_training_ready
- remaining_blockers

TRACK C — Documentation and commits

Update:
docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md

Add:
- V12 SO-101 contact status
- whether SO-101 contact transfer is solved
- whether robot-contact demos exist
- whether Training v2 ran or was honestly skipped
- exact next step after V12/V2

Update:
docs/MODAL_MIGRATION_PLAN.md only if needed.

Commit only source/docs:
- scripts/tera/*v12*.py
- modal_apps/terafold_policy_training_v2_robot_contact.py
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

Autonomous loop:
1. Inspect existing V6/V7/V8/V11 scripts.
2. Reuse code instead of rewriting everything.
3. Implement V12.
4. Run py_compile.
5. Run Modal V12.
6. Pull/read JSON.
7. Patch and rerun up to 3 serious attempts.
8. If V12 passes, run Training v2 robot-contact.
9. If V12 fails, document exact blocker and do not run fake robot-contact training.
10. Commit and push source/docs.
11. End with:
   - final git commit
   - contact_v12 verdict
   - training_v2 verdict
   - whether final autonomous SO-101 policy training is now authorized
   - exact next step

Start now.