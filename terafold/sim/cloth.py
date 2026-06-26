"""Procedural cloth grid for a convincing (non-FEM) towel-fold proxy.

A rectangular grid of vertices is laid flat over the detected towel quad. The
*stationary* half stays on the table; the *moving* half is hinged at the crease
and folds over it as a fold fraction ``f`` goes 0->1 (a rigid-flap rotation about
the in-plane crease axis, plus sag and a handle-snap that pins the grasped edge to
the gripper). This is kinematic/procedural — no finite-element physics — but it
visually communicates pick -> lift -> fold -> place, keeps cloth above the table,
and rests the folded half on top of the stationary half.

SIMULATION ONLY — not calibrated to real hardware.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class ClothGrid:
    def __init__(self, corners, direction, crease_a, crease_b,
                 nx: int = 25, ny: int = 15, z_table: float = 0.0,
                 thickness: float = 0.004, grasp_xy=None):
        self.nx, self.ny = int(nx), int(ny)
        self.z_table = float(z_table)
        self.thickness = float(thickness)
        self.direction = direction

        tl, tr, br, bl = [np.asarray(c, float)[:2] for c in corners]
        us = np.linspace(0.0, 1.0, self.nx)
        vs = np.linspace(0.0, 1.0, self.ny)
        grid = np.zeros((self.ny, self.nx, 3))
        for iv, v in enumerate(vs):
            top = (1 - us)[:, None] * tl + us[:, None] * tr
            bot = (1 - us)[:, None] * bl + us[:, None] * br
            grid[iv, :, :2] = (1 - v) * top + v * bot
        grid[:, :, 2] = self.z_table
        self.flat = grid.copy()
        self.pos = grid.copy()

        self.crease_pt = np.array([crease_a[0], crease_a[1], self.z_table])
        cdir = _unit(np.array([crease_b[0] - crease_a[0], crease_b[1] - crease_a[1], 0.0]))
        # In-plane normal to the crease.
        normal = np.array([-cdir[1], cdir[0], 0.0])

        # Signed side of every vertex; the moving half is the grasp side.
        rel = self.flat - self.crease_pt
        d = rel @ normal  # (ny, nx)
        if grasp_xy is None:
            grasp_xy = tr if direction in ("right_to_left",) else tl
        gsign = float(np.sign((np.asarray(grasp_xy, float)[:2] - self.crease_pt[:2]) @ normal[:2]))
        if gsign == 0.0:
            gsign = 1.0
        self.moving = (d * gsign) > 1e-9
        self.moving_init_sign = gsign
        self.side_normal = normal  # original in-plane crease normal (sign-stable)
        denom = float(np.abs(d[self.moving]).max()) if self.moving.any() else 1.0
        self.s = np.clip(np.abs(d) / max(denom, 1e-9), 0.0, 1.0)  # 0 at crease, 1 at far edge

        # Orient the rotation axis so the moving half lifts UP (+z) for theta>0.
        axis = cdir.copy()
        if np.cross(axis, gsign * normal)[2] < 0:
            axis = -axis
        self.axis = axis

        # Grasp handle vertex = moving vertex nearest the planned grasp point.
        gxy = np.asarray(grasp_xy, float)[:2]
        dist = np.linalg.norm(self.flat[:, :, :2] - gxy, axis=2)
        dist[~self.moving] = np.inf
        self.grasp_idx = np.unravel_index(int(np.argmin(dist)), dist.shape)

    # ----------------------------------------------------------------- update
    def update(self, f: float, handle: Optional[np.ndarray] = None, grasped: bool = False,
               sag_amt: float = 0.05) -> None:
        f = float(np.clip(f, 0.0, 1.0))
        theta = f * np.pi
        pos = self.flat.copy()

        # Rigid-flap rotation of the moving half about the crease axis (Rodrigues).
        rel = self.flat - self.crease_pt
        a = self.axis
        rot = (rel * np.cos(theta)
               + np.cross(np.broadcast_to(a, rel.shape), rel) * np.sin(theta)
               + np.broadcast_to(a, rel.shape) * ((rel @ a)[..., None]) * (1 - np.cos(theta)))
        rot = rot + self.crease_pt
        m = self.moving
        pos[m] = rot[m]

        # Sag: rows farther from the grasped row droop during mid-fold (sin theta).
        vrow = self.grasp_idx[0]
        vdist = np.abs(np.arange(self.ny) - vrow) / max(self.ny - 1, 1)
        sag = sag_amt * np.sin(theta) * (vdist[:, None]) * self.s
        pos[:, :, 2] -= np.where(m, sag, 0.0)

        # Folded flap rests slightly ON TOP of the stationary half.
        pos[:, :, 2] += np.where(m, self.thickness * f, 0.0)

        # Handle-snap: pin the grasped edge to the gripper (weight 0 at crease -> 1 at edge).
        if grasped and handle is not None:
            pr_grasp = pos[self.grasp_idx]
            offset = np.asarray(handle, float) - pr_grasp
            w = self.s[..., None]
            pos = pos + np.where(m[..., None], w * offset, 0.0)

        # Cloth never goes below the table.
        pos[:, :, 2] = np.maximum(pos[:, :, 2], self.z_table)
        self.pos = pos

    # --------------------------------------------------------------- geometry
    def mesh_geoms(self, top_rgba=(0.16, 0.52, 0.60, 1.0),
                   under_rgba=(0.78, 0.74, 0.52, 1.0),
                   move_rgba=(0.24, 0.62, 0.68, 1.0), fold_f: float = 0.0):
        """Per-cell (center, mat9, size, rgba) arrays for shaded mesh rendering."""
        p = self.pos
        p00, p01 = p[:-1, :-1], p[:-1, 1:]
        p10, p11 = p[1:, :-1], p[1:, 1:]
        center = (p00 + p01 + p10 + p11) / 4.0
        ex = p01 - p00
        ey = p10 - p00
        lx = np.linalg.norm(ex, axis=2, keepdims=True)
        ly = np.linalg.norm(ey, axis=2, keepdims=True)
        exn = ex / np.maximum(lx, 1e-9)
        ez = np.cross(exn, ey)
        lz = np.linalg.norm(ez, axis=2, keepdims=True)
        ezn = ez / np.maximum(lz, 1e-9)
        eyn = np.cross(ezn, exn)
        # Rotation matrix per cell, columns = (exn, eyn, ezn), flattened row-major.
        R = np.stack([exn, eyn, ezn], axis=-1)  # (..., 3, 3)
        m_cell = self.moving[:-1, :-1]
        cb = self.s[:-1, :-1]  # for checker/grid shading

        n = center.shape[0] * center.shape[1]
        centers = center.reshape(n, 3)
        mats = R.reshape(n, 9)
        sizes = np.zeros((n, 3))
        sizes[:, 0] = (lx.reshape(n) * 0.55)
        sizes[:, 1] = (ly.reshape(n) * 0.55)
        sizes[:, 2] = self.thickness * 0.5

        top = np.array(top_rgba)
        under = np.array(under_rgba)
        move = np.array(move_rgba)
        cols = np.zeros((n, 4))
        mflat = m_cell.reshape(n)
        # Moving cells fade top->underside colour as the fold completes.
        moving_col = (1 - fold_f) * move + fold_f * under
        cols[mflat] = moving_col
        cols[~mflat] = top
        # Subtle checker for grid visibility.
        ii, jj = np.meshgrid(np.arange(center.shape[0]), np.arange(center.shape[1]), indexing="ij")
        checker = ((ii + jj) % 2).reshape(n).astype(bool)
        cols[checker, :3] *= 0.88
        return centers, mats, sizes, cols

    def moving_edge_points(self):
        """World points along the moving (grasped) far edge — for debug drawing."""
        col = self.grasp_idx[1]
        return self.pos[:, col, :]

    def grasp_point(self):
        return self.pos[self.grasp_idx]

    # ---------------------------------------------------------------- metrics
    def success_metrics(self, place_target, edge_tol: float = 0.06) -> Dict[str, Any]:
        edge = self.grasp_point()
        place_target = np.asarray(place_target, float)
        pt = place_target if place_target.shape[0] == 3 else np.array([*place_target[:2], self.z_table])
        final_edge_error = float(np.linalg.norm(edge[:2] - pt[:2]))

        normal = self.side_normal
        moving_centroid = self.pos[self.moving].mean(0)
        side = float((moving_centroid[:2] - self.crease_pt[:2]) @ normal[:2])
        crossed = bool(side * self.moving_init_sign < 0)
        settled = crossed and bool(self.pos[self.moving][:, 2].max() < self.z_table + 0.05)
        success = bool(crossed and final_edge_error <= edge_tol)
        notes = []
        if not crossed:
            notes.append("moving half did not cross the crease")
        if final_edge_error > edge_tol:
            notes.append(f"grasped edge {final_edge_error*100:.1f}cm from target (> {edge_tol*100:.0f}cm)")
        if not notes:
            notes.append("fold completed: moving half crossed the crease and the edge landed on target")
        return {
            "fold_visually_successful": success,
            "final_edge_error_m": round(final_edge_error, 4),
            "crossed_crease": crossed,
            "settled_on_target_side": settled,
            "notes": "; ".join(notes),
        }
