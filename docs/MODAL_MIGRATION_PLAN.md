# Modal Migration Plan

## Current decision

Mac is the source-of-truth development machine.
GitHub is the source-of-truth code repo.
Modal is the primary GPU runtime for PyTorch training and artifact storage.
Brev is now a frozen historical Isaac VM unless Isaac-on-Modal fails and we need to recover files.

## Proven

Modal is authenticated in the terarobotics workspace.
The `terafold-artifacts` Modal Volume exists.
A Modal L40S GPU job successfully ran PyTorch CUDA.
A Modal L40S Isaac Sim container successfully launched SimulationApp headless after resetting the container entrypoint.

### Track B — training pipeline (PROVEN, non-Isaac path)

`modal_apps/terafold_policy_training_v0.py` ran on a Modal L40S and:
- confirmed `cuda_available=true`, `gpu_name=NVIDIA L40S`
- unpacked the source-only git archive and imported the repo's own
  `clean_towel_v4_common` (`build_training_dataset`, `PolicyMLP`) with no Isaac
- trained the compact-state policy for 300 steps: loss 1.021 -> 0.000149,
  action MAE 0.00107
- wrote `/training_v0/training_v0_summary.json` and a checkpoint to the
  `terafold-artifacts` volume (checkpoint is volume-only, never committed)

Honest labeling: `dataset_source=generated_rect_grid_fallback`,
`synthetic_or_real=synthetic`, `robot_contact_data_used=false`. This proves
training INFRASTRUCTURE only. It is NOT a robot-contact folding policy. The
remaining data blocker is that no robot-contact demonstration dataset exists,
and real robot-contact policy training requires proven cloth/rigid contact
transfer first.

### Track A — Isaac on Modal (6.0.1 infra PROVEN; particle-cloth version blocker found)

`modal_apps/terafold_modal_repo_isaac_smoke.py` (v4) ran on Modal L40S with
Isaac Sim `6.0.1` and proved the Modal Isaac *infrastructure*:
- repo unpack + py_compile of all tera scripts: OK
- IsaacLab (`main`) clone + `from isaaclab.app import AppLauncher`: OK
- `torch` installed into `/isaac-sim/python.sh` (the v3 blocker): OK,
  torch `2.12.1+cu130`, `cuda=true` (fixes v3 ModuleNotFoundError: torch)
- `create_clean_rect_towel_usd_v2.py` ran (writes a USDA with PhysX schema)

BUT the actual cloth simulation cannot run on 6.0.1:
- `test_clean_rect_towel_usd_v2.py` failed with
  `NotImplementedError: SingleClothPrim is no longer available. Omniverse PhysX
  removed the deprecated particle-based cloth features. Please use the new
  deformable body API in isaacsim.core.experimental instead.`
- The Tera clean-towel stack (V2..V8) is built on
  `isaacsim.core.prims.SingleClothPrim` + `PhysxSchema.PhysxParticleClothAPI`
  (the Isaac Sim 4.5 particle-cloth API), which 6.0.1 removed.
- `test_clean_towel_primitive_contact_v8.py` returned exit 0 but its own JSON
  reported an internal failure (missing `flatdict`); that is a red herring next
  to the removed cloth API.

Conclusion: to run the real cloth/contact scripts on Modal we must match the
Isaac Sim version Brev used. See `modal_apps/terafold_modal_isaac45_cloth_smoke.py`
(Isaac Sim 4.5.0 + IsaacLab v2.0.2).

### Track A — Isaac Sim 4.5.0 on Modal (particle cloth PROVEN + contact blocker reproduced)

`modal_apps/terafold_modal_isaac45_cloth_smoke.py` ran on Modal L40S with Isaac
Sim `4.5.0` + IsaacLab `v2.0.2` (torch `2.5.1+cu118` bundled) and proved that
the REAL Tera particle-cloth scripts run on Modal:
- `create_clean_rect_towel_usd_v2.py` -> `CLEAN_RECT_TOWEL_USD_V2_CREATED`
- `test_clean_rect_towel_usd_v2.py`   -> `CLEAN_RECT_TOWEL_USD_V2_TEST_OK`
  (this is the exact test that FAILS on 6.0.1; `SingleClothPrim` instantiates
  and simulates on 4.5)
- `test_clean_towel_primitive_contact_v8.py` -> `CLEAN_TOWEL_PRIMITIVE_CONTACT_V8_DONE`

=> Brev can be abandoned for cloth/contact work too; Modal + Isaac Sim 4.5.0 is
the replacement runtime.

The V8 contact result reproduced the Brev blocker EXACTLY on Modal:
- `actual_touch_achieved=true`, `nearest_collider_to_towel_particle_m=0.0`
- `primitive_moves_towel=false`, `selected_edge_particle_displacement=0.0`
- `start_width==best_width==final_width==0.68` (cloth completely static)

The displacement being *exactly* 0.0 (not merely small) triggered the V9
particle-sim sanity probe (`scripts/tera/inspect_clean_towel_particle_sim_v9.py`),
which asks the more fundamental question: does the cloth simulate under gravity
at all? (Result recorded in the autonomy status doc.)

## Contact transfer — SOLVED at the primitive level (V11)

The clean-towel rigid/particle contact blocker is resolved. V11 showed that
driving the collider through the physics kinematic target
(`RigidPrim.set_world_poses`) instead of teleporting its USD transform
(`set_prim_translation`, the V8 method) transfers real motion: edge displacement
`0.081 m` and width `0.68 → 0.63` via a valid rigid contact (not a particle
teleport). See `docs/TERAFOLD_MODAL_AUTONOMY_STATUS.md` for the full V11 table
and honest caveats (dynamic high-velocity flings are ballistic artifacts and are
excluded).

## Not proven yet

Full autonomous robot-contact towel FOLDING (not just partial motion) with the
SO-101 gripper. Remaining engineering: port the physics-driven drive to the
SO-101 articulation, achieve a full fold (`best_width < 0.5`), and record real
contact demonstrations for policy training.

## Modal roles

Use Modal for:
- PyTorch training
- dataset conversion
- model evaluation that does not require Isaac
- artifact storage in `terafold-artifacts`
- later Isaac headless smoke tests if the container works

Do not use Modal for:
- claiming Isaac works before smoke test
- claiming policy training is valid before contact mechanics are fixed
- committing artifacts to Git

## Artifact policy

Do not commit:
- data/
- runs/
- tera_checkpoints/
- .pt
- .npz
- .rrd
- .usd
- .usda
- .ckpt

Use Modal Volume or external storage for generated artifacts.

## Next step

Run the next Isaac-on-Modal smoke test against the actual LeIsaac/Tera scripts.
The next test must answer whether Modal can run our repo's Isaac scripts, not just launch Isaac Sim.
