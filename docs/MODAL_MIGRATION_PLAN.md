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

## Not proven yet

Isaac Sim headless on Modal has passed the first smoke test.
Isaac Lab / LeIsaac repo scripts on Modal have not been proven yet.
Autonomous robot-contact towel folding has not been proven.
The clean towel rigid-contact issue is not solved.

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
