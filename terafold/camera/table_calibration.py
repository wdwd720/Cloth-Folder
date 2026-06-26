"""Manual table calibration: map 4 image pixels to 4 table coordinates.

For a phone/top-down camera, a planar homography is enough. The user provides
four image pixel points and the matching four table-frame metric coordinates
(meters); we solve ``H_IT`` and persist it. AprilTag/checkerboard auto-detection
can be added later, but is intentionally NOT required now.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

__all__ = ["parse_xy_list", "calibrate_table_from_image"]


def parse_xy_list(spec: str) -> List[List[float]]:
    """Parse ``"u1,v1;u2,v2;u3,v3;u4,v4"`` into ``[[u,v], ...]``."""
    points: List[List[float]] = []
    for chunk in str(spec).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(" ", "").split(",") if p != ""]
        if len(parts) != 2:
            raise ValueError(f"bad point {chunk!r}; expected 'a,b'")
        points.append([float(parts[0]), float(parts[1])])
    return points


def calibrate_table_from_image(
    image_pts, table_pts, out: str, image_path: str = None
) -> Dict[str, Any]:
    """Solve + save the image->table homography from >=4 correspondences."""
    from terafold.camera.homography import save_homography
    from terafold.math.transforms import compute_homography, homography_reprojection_error

    image_pts = np.asarray(image_pts, dtype=np.float64)
    table_pts = np.asarray(table_pts, dtype=np.float64)
    if image_pts.shape != table_pts.shape or image_pts.shape[0] < 4 or image_pts.shape[1] != 2:
        raise ValueError("need >=4 matched (image, table) point pairs of shape (N, 2)")

    H = compute_homography(image_pts, table_pts)
    err_m = homography_reprojection_error(H, image_pts, table_pts)
    save_homography(out, H)
    return {
        "status": "ok",
        "out": out,
        "num_points": int(image_pts.shape[0]),
        "reprojection_error_m": float(err_m),
        "image": image_path,
        "calibrated": True,
        "note": "Homography saved. Plans can now run in metric table coordinates.",
    }
