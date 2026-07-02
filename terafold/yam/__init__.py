"""TeraFold YAM/MolmoAct2 stack — dual I2RT YAM Standard arms, shadow mode only.

This package is the hardware-free foundation for the MolmoAct2 Research Kit
(dual YAM arms, cameras ordered top/left/right, norm tag
``yam_dual_molmoact2``). It contains config loading, asset checks, dummy
episode tooling, the MolmoAct2 smoke-test adapter, an offline shadow policy,
Rerun logging, and pure safety clamps.

Nothing here imports a robot SDK, opens a serial port, or commands hardware —
see :mod:`terafold.yam.safety` for the gates that keep it that way. Heavy deps
(torch / transformers / rerun) are lazy-imported by the modules that need them.
"""

from __future__ import annotations

__all__: list[str] = []
