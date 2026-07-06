import argparse
import json
from pathlib import Path

import numpy as np


OUT_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
DEFAULT_CAPTURE_PATH = OUT_DIR / "clean_rect_towel_usd_v2_capture.npz"
DEFAULT_GRID_PATH = OUT_DIR / "clean_rect_towel_grid_v2.npz"
DEFAULT_OBJ_PATH = OUT_DIR / "clean_rect_towel_usd_v2_after_100_surface.obj"
DEFAULT_RRD_PATH = OUT_DIR / "clean_rect_towel_usd_v2_after_100_surface.rrd"
DEFAULT_SUMMARY_PATH = OUT_DIR / "visualize_clean_towel_usd_v2_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OBJ/Rerun visual outputs for clean towel v2.")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--capture_path", type=Path, default=DEFAULT_CAPTURE_PATH)
    parser.add_argument("--grid_path", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--max_points", type=int, default=24000)
    return parser.parse_args()


def write_obj(path: Path, points: np.ndarray, face_indices: np.ndarray | None) -> int:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Clean rectangular towel v2 surface\n")
        for p in points:
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        if face_indices is not None:
            for i in range(0, len(face_indices), 3):
                a, b, c = face_indices[i : i + 3]
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")
            return int(len(face_indices) // 3)
    return 0


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = None
    points = None
    if args.capture_path.exists():
        capture = np.load(args.capture_path)
        points = capture["after_100_points"].astype(np.float32)
        source = str(args.capture_path)
    elif args.grid_path.exists():
        grid = np.load(args.grid_path)
        points = grid["points"].astype(np.float32)
        source = str(args.grid_path)
    else:
        summary = {
            "status": "CLEAN_TOWEL_USD_V2_VISUAL_FAILED",
            "capture_path": str(args.capture_path),
            "grid_path": str(args.grid_path),
            "reason": "missing_capture_and_grid",
        }
        summary_path = args.out_dir / DEFAULT_SUMMARY_PATH.name
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        return

    points = points[np.isfinite(points).all(axis=1)]
    grid_face_indices = None
    if args.grid_path.exists():
        grid = np.load(args.grid_path)
        grid_points = grid["points"].astype(np.float32)
        if len(points) == len(grid_points):
            grid_face_indices = grid["face_indices"].astype(np.int32)

    obj_path = args.out_dir / DEFAULT_OBJ_PATH.name
    num_triangles = write_obj(obj_path, points, grid_face_indices)

    sample_count = min(args.max_points, len(points))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(points), size=sample_count, replace=False)
    sampled = points[sample_indices]
    rrd_path = args.out_dir / DEFAULT_RRD_PATH.name
    rerun_status = "not_attempted"
    rerun_error = None
    try:
        import rerun as rr

        rr.init("clean_rect_towel_usd_v2", spawn=False)
        rr.save(str(rrd_path))
        rr.log(
            "world/clean_rect_towel_v2_points",
            rr.Points3D(sampled, radii=0.0025, colors=np.tile(np.array([[46, 115, 204]], dtype=np.uint8), (sample_count, 1))),
        )
        if grid_face_indices is not None and len(points) == len(sampled):
            rr.log(
                "world/clean_rect_towel_v2_surface",
                rr.Mesh3D(
                    vertex_positions=points,
                    triangle_indices=grid_face_indices.reshape(-1, 3),
                    vertex_colors=np.tile(np.array([[46, 115, 204]], dtype=np.uint8), (len(points), 1)),
                ),
            )
        rerun_status = "rerun_ok"
    except Exception as exc:
        rerun_status = "rerun_failed"
        rerun_error = repr(exc)

    cmin = points.min(axis=0)
    cmax = points.max(axis=0)
    size = cmax - cmin
    summary = {
        "status": "CLEAN_TOWEL_USD_V2_VISUAL_DONE",
        "source_points": source,
        "obj_path": str(obj_path),
        "rrd_path": str(rrd_path),
        "num_points": int(len(points)),
        "num_sampled_points": int(sample_count),
        "num_triangles": int(num_triangles),
        "width_x": float(size[0]),
        "height_y": float(size[1]),
        "thickness_z": float(size[2]),
        "rerun_status": rerun_status,
        "rerun_error": rerun_error,
        "honest_note": "This visualizes the v2 clean rectangular towel from the smoke-test capture when present, otherwise from the generated grid.",
    }
    summary_path = args.out_dir / DEFAULT_SUMMARY_PATH.name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
