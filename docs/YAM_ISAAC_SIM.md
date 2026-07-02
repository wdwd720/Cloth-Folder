# YAM Digital Twin in Isaac Sim (Visual Only)

Sprint 2 of the YAM/MolmoAct2 stack: see the dual-YAM rig — table, towel
placeholder, two arms, top/left/right cameras — inside Isaac Sim on a cloud
GPU. **This is a visual digital twin only**: no cloth-policy training, no
controllers, and (as always in this stack) no path to real hardware.

## Why Isaac runs on a separate pod from the MolmoAct2 one

The MolmoAct2 smoke test wants a plain CUDA PyTorch pod (`pip install
transformers ...` and go). Isaac Sim is a different beast: a ~10 GB NGC
container with its own bundled Python (`./python.sh`), RTX-capable driver
requirements, EULA acceptance, and livestream ports. Mixing them in one image
means version fights between Isaac's bundled torch and the one transformers
wants, and a giant image you rebuild constantly. Keep them separate: PyTorch
pod for policy work, Isaac pod for the twin. They only share files
(episodes, USD scenes), never an environment.

## Requirements (RunPod or any cloud GPU host)

- NVIDIA GPU with RTX capability (L4/L40/A10/RTX 4090-class or better),
  recent NVIDIA driver, and `nvidia-container-toolkit` on the host.
- Docker access to `nvcr.io` (free NGC account; `docker login nvcr.io`).
- ~30 GB disk for the image + cache mounts.
- **Ports for livestream: TCP 49100 and UDP 47998** must be reachable from
  your machine (RunPod: expose them in the pod template; `--network=host`
  inside the container).

## Build and view the scene

Generate the scene package locally (no GPU needed):

```bash
python3 -m terafold yam-build-isaac-scene \
  --config configs/yam_dual_reference.yaml \
  --out sim/yam_dual_towel_scene
```

Then print the exact container commands (pull, run with `--gpus all` +
`ACCEPT_EULA=Y` + cache mounts, compatibility check, headless livestream, and
the scene build):

```bash
python3 -m terafold yam-print-isaac-cloud-commands
```

Copy the repo (or just `sim/yam_dual_towel_scene/`) to the GPU host first —
the printed `docker run` mounts it at `/workspace/yam_scene`.

## Connecting with the Isaac Sim WebRTC Streaming Client

1. Download the **Isaac Sim WebRTC Streaming Client** for your OS from the
   NVIDIA Isaac Sim documentation/downloads page.
2. Start the livestream in the container: `./runheadless.sh -v` (the reliable
   path — it bakes in the streaming extensions). Alternatively,
   `./python.sh /workspace/yam_scene/build_isaac_scene.py --wait` builds the
   scene and then *attempts* to enable the livestream extension itself — watch
   its console: if it prints that livestream is NOT active, fall back to
   `./runheadless.sh -v` and open the saved USD there.
3. Wait for the console line `Isaac Sim ... app ready` (first boot compiles
   shaders and can take several minutes).
4. In the client, enter the pod's **public IP** (TCP 49100 / UDP 47998 must be
   open) and connect.

## What success looks like

- `build_isaac_scene.py` prints the camera order `['top', 'left', 'right']`,
  `usd: .../yam_dual_towel_scene.usd`, and
  `simulation only: no hardware was (or can be) commanded`.
- `yam_dual_towel_scene.usd` appears next to the script (open it later in any
  USD viewer or a fresh Isaac session).
- In the streaming client you see: a table with four legs, a flat blue towel
  rectangle on top, two orange arm placeholders (or the real YAM meshes once
  `paths.urdf` / `paths.usd` is configured), and three camera prims under
  `/World/cameras` in the top/left/right order MolmoAct2 expects.

## Why visual-only, not cloth policy training

Rigid placeholder ≠ cloth. Real towel dynamics (bending, friction,
self-collision) are exactly what sim gets wrong, which is why the folding
policy will be trained on real YAM teleop demos
(see docs/YAM_MOLMOACT2_QUICKSTART.md), not on this scene. The twin's job is
cheaper things sim is *good* at: camera placement, workspace/reach sanity
checks, replaying recorded episodes, and later collision-checking planned
motions. Deformable-cloth simulation and any sim-trained behavior are
explicitly out of scope for this sprint.

## Safety

Same rules as the rest of the YAM stack: the scene generator and the generated
script import no robot SDK, contain no motion commands, and cannot reach
hardware. `scene_config.json` records `simulation_only: true` and
`hardware_commanded: false`.
