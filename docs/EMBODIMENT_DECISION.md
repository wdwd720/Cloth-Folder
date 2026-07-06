# TeraFold Embodiment Decision

As of 2026-07-05, TeraFold will use dual SO-101 with action_dim 12 as the canonical cloth-folding embodiment.

## Decision

- Canonical folding embodiment: dual SO-101
- Canonical action dimension: 12
- Canonical simulated cloth: clean rectangular towel v2, 0.68m x 0.38m
- YAM / MolmoAct2 remains a separate frozen shadow-mode research lane.
- SO-101 12D data must not be mixed with YAM 14D data or single-arm 7D mock LeRobot data.

## Honesty note

No autonomous robot-contact towel folding has been proven yet.

Brev verified:
- clean rectangular PhysX towel
- SO-101 + clean towel scene
- particle-space assisted fold

Brev did not verify:
- pure robot-contact folding
- autonomous towel folding

The next milestone is V6 reachability and towel placement diagnostics.
