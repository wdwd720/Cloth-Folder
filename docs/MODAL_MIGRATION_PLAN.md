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

## Not proven yet

Isaac Sim / Isaac Lab headless on Modal has not been proven.
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

Create an Isaac-on-Modal smoke test as an experimental scaffold only.
The first Isaac test must answer whether Modal can run the Isaac/Isaac Lab environment headless on an RTX-capable GPU.
