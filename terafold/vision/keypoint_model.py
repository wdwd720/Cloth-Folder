"""Small U-Net keypoint heatmap model (lazy torch).

Model 1 of the TeraFold perception stack: it maps a ``(B, 3, 256, 256)`` RGB
image to ``(B, 8, 64, 64)`` keypoint heatmaps (one channel per
:data:`terafold.physics.cloth_state.DETECTOR_KEYPOINTS` entry).

torch is imported **lazily inside** :func:`build_keypoint_model` /
:func:`heatmap_loss` so the module always imports with numpy alone; downstream
numpy code (dataset, fallback predictor) never pulls in torch by importing this.
"""

from __future__ import annotations

from terafold.physics.cloth_state import DETECTOR_KEYPOINTS

__all__ = [
    "IMAGE_SIZE",
    "HEATMAP_SIZE",
    "NUM_KEYPOINTS",
    "build_keypoint_model",
    "heatmap_loss",
]

IMAGE_SIZE: int = 256
HEATMAP_SIZE: int = 64
NUM_KEYPOINTS: int = len(DETECTOR_KEYPOINTS)


def build_keypoint_model(num_keypoints: int = NUM_KEYPOINTS):
    """Build a small U-Net mapping ``(B,3,256,256)`` -> ``(B,K,64,64)``.

    Two downsampling stages (256 -> 128 -> 64) with a skip connection, a
    bottleneck, then a 1x1 heatmap head. Output is at 64x64, matching the
    target heatmap resolution. Requires torch.
    """
    try:
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "build_keypoint_model requires torch. Install it with "
            "`pip install -e '.[torch]'` (or `pip install torch`)."
        ) from exc

    def conv_block(cin: int, cout: int) -> "nn.Module":
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class KeypointUNet(nn.Module):
        """Compact encoder-decoder heatmap regressor."""

        def __init__(self, k: int) -> None:
            super().__init__()
            self.enc1 = conv_block(3, 32)  # 256
            self.pool1 = nn.MaxPool2d(2)  # -> 128
            self.enc2 = conv_block(32, 64)  # 128
            self.pool2 = nn.MaxPool2d(2)  # -> 64
            self.bottleneck = conv_block(64, 128)  # 64
            # Decode at 64x64; skip from enc2 (downsampled to 64).
            self.dec = conv_block(128 + 64, 64)
            self.skip_pool = nn.MaxPool2d(2)  # enc2 (128) -> 64
            self.head = nn.Conv2d(64, k, 1)

        def forward(self, x):  # noqa: ANN001
            e1 = self.enc1(x)  # (B,32,256,256)
            e2 = self.enc2(self.pool1(e1))  # (B,64,128,128)
            b = self.bottleneck(self.pool2(e2))  # (B,128,64,64)
            skip = self.skip_pool(e2)  # (B,64,64,64)
            import torch

            d = self.dec(torch.cat([b, skip], dim=1))  # (B,64,64,64)
            return self.head(d)  # (B,k,64,64)

    return KeypointUNet(int(num_keypoints))


def heatmap_loss(pred, target):
    """Mean-squared-error loss between predicted and target heatmaps."""
    try:
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "heatmap_loss requires torch. Install it with `pip install torch`."
        ) from exc
    return F.mse_loss(pred, target)
