"""TeraFold — a trainable, physically-grounded cloth-folding robot stack.

TeraFold is a *hybrid* robot-learning system. It uses explicit geometry and
physics as a prior/scaffold, and learns data-driven corrections on top so that
every fragile part can be replaced or refined from data.

Design rules enforced throughout the package:

* The Stage-0 pipeline (synthetic data -> tests -> mock dry-run -> episode
  recording -> LeRobot export) runs with numpy only. Heavy dependencies
  (torch, opencv, pillow, lerobot, huggingface) are *optional extras* and are
  always lazy-imported. Importing :mod:`terafold` must never require them.
* Physical motion is disabled by default. Real hardware execution requires
  explicit safety flags; see :mod:`terafold.robot.safety`.

See the module docstrings and the README for the full architecture.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
