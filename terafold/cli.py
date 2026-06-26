"""TeraFold command-line interface.

A single ``terafold`` entry point wires the whole stack together. Every command
imports its dependencies *lazily* so that:

* the CLI starts fast,
* a command that needs a heavy dep (torch / cv2 / lerobot) fails with a clear,
  actionable message instead of breaking the whole CLI at import time,
* and the numpy-only commands work with nothing extra installed.

Physical motion is OFF by default everywhere. Commands that can move hardware
require BOTH ``--enable-motion`` and ``--i-understand-this-moves-hardware``.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="TeraFold — trainable cloth-folding robot stack (hybrid geometry + learning).",
)


def _echo(msg: str) -> None:
    typer.echo(msg)


def _err(msg: str) -> None:
    typer.echo(typer.style(msg, fg=typer.colors.RED), err=True)


def _ok(msg: str) -> None:
    typer.echo(typer.style(msg, fg=typer.colors.GREEN))


# --------------------------------------------------------------------------
# Perception / data generation
# --------------------------------------------------------------------------


@app.command("generate-synthetic")
def generate_synthetic(
    num: int = typer.Option(10000, help="Number of samples to generate."),
    out: str = typer.Option(..., help="Output dataset directory."),
    image_size: int = typer.Option(256, help="Square image size (px)."),
    seed: int = typer.Option(0, help="RNG seed."),
    markers: bool = typer.Option(False, "--markers/--no-markers", help="Marker-assisted mode."),
    pairs: bool = typer.Option(False, help="Also generate before/after fold pairs for the success model."),
):
    """Render synthetic top-down towel images with keypoint/mask labels."""
    from terafold.vision.synthetic_cloth import generate_synthetic_dataset, generate_fold_pairs

    _echo(f"Generating {num} synthetic samples -> {out} ({image_size}px, markers={markers})")
    summary = generate_synthetic_dataset(out, num, image_size=image_size, seed=seed, marker_mode=markers)
    _ok(f"Done: {summary}")
    if pairs:
        psum = generate_fold_pairs(out + "_pairs", max(1, num // 5), image_size=image_size, seed=seed + 1)
        _ok(f"Fold pairs: {psum}")


@app.command("train-keypoints")
def train_keypoints_cmd(
    data: str = typer.Option(..., help="Synthetic dataset directory."),
    out: str = typer.Option(..., help="Run output directory (model.pt)."),
    epochs: int = typer.Option(30),
    batch_size: int = typer.Option(16),
    lr: float = typer.Option(1e-3),
    limit: Optional[int] = typer.Option(None, help="Limit number of samples (debug)."),
):
    """Train the keypoint/grasp-place heatmap detector (Model 1)."""
    try:
        from terafold.vision.train_keypoints import train_keypoints
    except Exception as e:  # pragma: no cover
        _err(f"Cannot import trainer: {e}")
        raise typer.Exit(1)
    try:
        hist = train_keypoints(data, out, epochs=epochs, batch_size=batch_size, lr=lr, limit=limit)
    except ImportError as e:
        _err(str(e))
        _err("Install the ML extra:  pip install -e '.[ml]'")
        raise typer.Exit(1)
    _ok(f"Trained. Final: {hist.get('final', hist)}")


@app.command("infer-keypoints")
def infer_keypoints_cmd(
    image: str = typer.Option(..., help="Input image path."),
    checkpoint: Optional[str] = typer.Option(None, help="Keypoint model checkpoint (optional)."),
    out: Optional[str] = typer.Option(None, help="Save overlay visualization to this path."),
):
    """Run keypoint inference on a single image (learned model or classical fallback)."""
    from terafold.vision.imageio import imread
    from terafold.vision.infer_keypoints import resolve_keypoint_predictor

    img = imread(image)
    predictor, pinfo = resolve_keypoint_predictor(checkpoint)
    _report_perception(pinfo, checkpoint)
    fs = predictor.predict(img)
    _ok("Detected keypoints:")
    _echo(f"  corners: {fs.keypoints.corners.tolist()}")
    if out:
        from terafold.vision.visualization import draw_keypoints, draw_fold_line, save_visualization

        vis = draw_keypoints(img.copy(), fs.keypoints)
        if fs.fold_line is not None:
            vis = draw_fold_line(vis, fs.fold_line)
        save_visualization(out, vis)
        _ok(f"Saved overlay -> {out}")


@app.command("live-keypoints")
def live_keypoints(
    checkpoint: Optional[str] = typer.Option(None),
    camera_index: int = typer.Option(0),
):
    """Live keypoint overlay from a webcam (requires opencv)."""
    try:
        from terafold.camera.opencv_camera import OpenCVCamera
        from terafold.config.schema import CameraConfig
        from terafold.vision.infer_keypoints import get_keypoint_predictor
    except Exception as e:  # pragma: no cover
        _err(f"Import error: {e}")
        raise typer.Exit(1)
    predictor = get_keypoint_predictor(checkpoint)
    cam = OpenCVCamera(CameraConfig(type="opencv", index=camera_index))
    try:
        cam.connect()
    except Exception as e:
        _err(f"Could not open camera {camera_index}: {e}")
        _err("Install the vision extra:  pip install -e '.[vision]'")
        raise typer.Exit(1)
    _echo("Press Ctrl+C to stop.")
    try:
        import cv2

        while True:
            frame = cam.read()
            fs = predictor.predict(frame.image)
            from terafold.vision.visualization import draw_keypoints

            vis = draw_keypoints(frame.image.copy(), fs.keypoints)
            cv2.imshow("terafold live-keypoints", vis[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()


# --------------------------------------------------------------------------
# Camera / calibration
# --------------------------------------------------------------------------


@app.command("camera-test")
def camera_test(
    camera_index: int = typer.Option(0),
    camera: str = typer.Option("opencv", help="opencv | mock"),
    out: Optional[str] = typer.Option(None, help="Save a captured frame here."),
):
    """Capture a frame and report camera health."""
    from terafold.config.schema import CameraConfig

    if camera == "mock":
        from terafold.camera.mock_camera import MockCamera

        cam = MockCamera(CameraConfig(type="mock"))
    else:
        from terafold.camera.opencv_camera import OpenCVCamera

        cam = OpenCVCamera(CameraConfig(type="opencv", index=camera_index))
    try:
        cam.connect()
        frame = cam.read()
    except Exception as e:
        _err(f"Camera error: {e}")
        _err("For a real webcam install:  pip install -e '.[vision]'")
        raise typer.Exit(1)
    _ok(f"Captured frame {frame.image.shape} from {camera} camera.")
    if out:
        from terafold.vision.imageio import imwrite

        imwrite(out, frame.image)
        _ok(f"Saved -> {out}")
    cam.disconnect()


@app.command("capture-calibration")
def capture_calibration(
    camera_index: int = typer.Option(0),
    out: str = typer.Option("data/calib/table.jpg"),
):
    """Capture a still image of the table for homography calibration."""
    from terafold.camera.calibration_cli import capture_calibration_image
    from terafold.camera.opencv_camera import OpenCVCamera
    from terafold.config.schema import CameraConfig

    cam = OpenCVCamera(CameraConfig(type="opencv", index=camera_index))
    try:
        path = capture_calibration_image(cam, out)
    except Exception as e:
        _err(f"Capture failed: {e}")
        raise typer.Exit(1)
    _ok(f"Saved calibration image -> {path}")


@app.command("calibrate-homography")
def calibrate_homography(
    image: str = typer.Option(..., help="Table calibration image."),
    out: str = typer.Option("data/calib/homography.json"),
    table_width: float = typer.Option(0.70, help="Known table rectangle width (m)."),
    table_height: float = typer.Option(0.50, help="Known table rectangle height (m)."),
    points: Optional[str] = typer.Option(None, help="JSON of {image_pts, table_pts} correspondences."),
):
    """Compute the image->table homography from corner correspondences."""
    from terafold.camera.calibration_cli import interactive_calibrate, run_calibration
    from terafold.data.episode_schema import read_json

    if points:
        d = read_json(points)
        frames = run_calibration(d["image_pts"], d["table_pts"], out)
    else:
        frames = interactive_calibrate(image, out, table_size=(table_width, table_height))
    _ok(f"Homography saved -> {out} (calibrated={frames.image_calibrated})")


# --------------------------------------------------------------------------
# Planning / execution
# --------------------------------------------------------------------------


def _load_task(task_path: str, keypoint_checkpoint: Optional[str] = None):
    from terafold.config.load import load_task_config
    from terafold.config.validate import validate_task_config
    from terafold.planning.fold_task import FoldTask

    cfg = load_task_config(task_path)
    # A `--checkpoint` from the CLI overrides the task config at runtime, so the
    # learned keypoint model is actually wired into the planner (and the "no
    # checkpoint" validation warning does not spuriously fire).
    if keypoint_checkpoint:
        cfg.learning.keypoint_checkpoint = keypoint_checkpoint
        cfg.learning.use_keypoint_model = True
    warnings = validate_task_config(cfg)
    for w in warnings:
        _err(f"[config warning] {w}")
    return FoldTask.from_config(cfg)


def _report_perception(info: dict, checkpoint: Optional[str]) -> None:
    """Print which perception backend is in use; warn loudly on fallback."""
    from terafold.vision.infer_keypoints import LEARNED_KIND

    if info["kind"] == LEARNED_KIND:
        _echo(f"perception: {info['kind']}")
        _echo(f"checkpoint: {info['checkpoint']}")
    else:
        _echo(f"perception: {info['kind']}")
        if checkpoint:
            _err(
                f"WARNING: --checkpoint {checkpoint!r} was provided but the planner "
                f"fell back to the classical predictor ({info['reason']})."
            )
        else:
            _echo("checkpoint: none")


def _maybe_frames(calibration: Optional[str]):
    if not calibration:
        return None
    from terafold.camera.homography import load_homography

    return load_homography(calibration)


@app.command("plan-fold")
def plan_fold_cmd(
    image: Optional[str] = typer.Option(None, help="Input image (else use --camera)."),
    task: str = typer.Option("configs/task_fold_towel_half.yaml"),
    camera: str = typer.Option("mock", help="mock | opencv (used if --image omitted)."),
    checkpoint: Optional[str] = typer.Option(None, help="Keypoint model checkpoint."),
    use_residual: Optional[str] = typer.Option(None, help="Residual model checkpoint."),
    calibration: Optional[str] = typer.Option(None, help="Homography JSON for metric planning."),
    out: Optional[str] = typer.Option(None, help="Save plan.json here."),
    overlay_out: Optional[str] = typer.Option(
        None, help="Save keypoints/fold-plan overlay visualization to this path."
    ),
    viz: Optional[str] = typer.Option(None, help="Alias of --overlay-out (deprecated)."),
):
    """Compute a fold plan from a camera image.

    Pass ``--checkpoint`` to use the learned keypoint model (requires torch and a
    valid checkpoint); otherwise the classical fallback predictor is used. The
    chosen perception backend is printed as ``perception: ...``.
    """
    from terafold.planning.towel_half_fold import TowelHalfFoldPlanner
    from terafold.vision.infer_keypoints import resolve_keypoint_predictor

    # The checkpoint is wired into the task config so the planner truly uses it.
    ft = _load_task(task, keypoint_checkpoint=checkpoint)
    frames = _maybe_frames(calibration)
    if image:
        from terafold.vision.imageio import imread

        img = imread(image)
    else:
        from terafold.camera.mock_camera import build_camera
        from terafold.config.schema import CameraConfig

        cam = build_camera(CameraConfig(type=camera))
        cam.connect()
        img = cam.read().image
        cam.disconnect()

    # Resolve perception explicitly so we can report it and warn on fallback.
    predictor, pinfo = resolve_keypoint_predictor(checkpoint)
    _report_perception(pinfo, checkpoint)

    residual = use_residual  # path; apply_residual_correction handles loading
    planner = TowelHalfFoldPlanner(
        ft, frames=frames, keypoint_predictor=predictor, residual_model=residual,
    )
    plan = planner.plan_from_image(img)
    _echo(plan.summary())
    if out:
        plan.save(out)
        _ok(f"Saved plan -> {out}")
    overlay = overlay_out or viz
    if overlay:
        from terafold.vision.visualization import draw_fold_plan, save_visualization

        save_visualization(overlay, draw_fold_plan(img.copy(), plan, frames))
        _ok(f"Saved overlay -> {overlay}")


@app.command("dry-run-fold")
def dry_run_fold(
    task: str = typer.Option("configs/task_fold_towel_half.yaml"),
    robot: str = typer.Option("mock", help="mock | learm | so101"),
    camera: str = typer.Option("mock", help="mock | opencv"),
    checkpoint: Optional[str] = typer.Option(None),
    calibration: Optional[str] = typer.Option(None),
    viz: Optional[str] = typer.Option(None, help="Save plan overlay image here."),
):
    """Plan a fold and execute it on a robot in DRY-RUN (no physical motion)."""
    from terafold.config.schema import CameraConfig
    from terafold.camera.mock_camera import build_camera
    from terafold.planning.towel_half_fold import TowelHalfFoldPlanner
    from terafold.planning.trajectory import trajectory_to_actions
    from terafold.vision.infer_keypoints import resolve_keypoint_predictor
    from terafold.robot.mock_robot import MockRobot

    ft = _load_task(task, keypoint_checkpoint=checkpoint)
    frames = _maybe_frames(calibration)
    cam = build_camera(CameraConfig(type=camera))
    cam.connect()
    img = cam.read().image
    cam.disconnect()

    predictor, pinfo = resolve_keypoint_predictor(checkpoint)
    _report_perception(pinfo, checkpoint)
    planner = TowelHalfFoldPlanner(ft, frames=frames, keypoint_predictor=predictor)
    plan = planner.plan_from_image(img)
    _echo(plan.summary())

    if robot != "mock":
        _err(f"dry-run-fold currently executes only on the mock robot (got {robot!r}).")
        raise typer.Exit(1)
    rob = MockRobot(workspace=ft.workspace, dry_run=True)
    rob.connect()
    actions = trajectory_to_actions(plan.trajectory, frames=frames)
    _echo(f"Executing {len(actions)} dry-run waypoints on MockRobot...")
    obs = rob.execute_trajectory(actions)
    rob.disconnect()
    _ok(f"Dry-run complete. Final EE pose: {obs[-1].ee_pose.tolist() if obs[-1].ee_pose is not None else 'n/a'}")
    if viz:
        from terafold.vision.visualization import draw_fold_plan, save_visualization

        save_visualization(viz, draw_fold_plan(img.copy(), plan, frames))
        _ok(f"Saved overlay -> {viz}")


# --------------------------------------------------------------------------
# Demonstration recording / data
# --------------------------------------------------------------------------


@app.command("list-cameras")
def list_cameras_cmd(
    max_index: int = typer.Option(8, help="Scan camera indices 0..max-index."),
    out: Optional[str] = typer.Option(None, help="Save snapshots here (camera_<i>.png)."),
    preview_seconds: float = typer.Option(0.0, help="Briefly show each working feed."),
):
    """Scan camera indices and report which are usable (phone/Continuity/USB webcam)."""
    from terafold.camera.discovery import scan_cameras

    report = scan_cameras(max_index=max_index, out=out, preview_seconds=preview_seconds)
    if not report.get("cv2", True) and not report.get("available", False):
        _err(report.get("hint", "OpenCV not available."))
        raise typer.Exit(1)
    for cam in report["cameras"]:
        if not cam["opened"]:
            _echo(f"  [{cam['index']}] not opened ({cam.get('error')})")
            continue
        tag = "USABLE" if cam["usable"] else "marginal"
        _echo(
            f"  [{cam['index']}] {tag}  {cam['width']}x{cam['height']}  "
            f"brightness={cam['brightness']:.0f}  sharpness={cam['blur']:.0f}"
            + (f"  -> {cam['snapshot_path']}" if cam.get("snapshot_path") else "")
        )
    _ok(f"Usable camera indices: {report.get('usable_indices', [])}")


@app.command("demo-image")
def demo_image_cmd(
    image: str = typer.Option(..., help="Input image path (top-down photo)."),
    mode: str = typer.Option("markers", help="markers | model | claude."),
    task: str = typer.Option("configs/task_fold_towel_half.yaml"),
    checkpoint: Optional[str] = typer.Option(None, help="Keypoint model checkpoint (mode=model)."),
    calibration: Optional[str] = typer.Option(None, help="Homography JSON (mode metric)."),
    homography: Optional[str] = typer.Option(None, help="Alias of --calibration."),
    marker_map: Optional[str] = typer.Option(None, help="e.g. red:top_left,blue:top_right,..."),
    debug_markers: bool = typer.Option(False, "--debug-markers", help="Save per-color masks."),
    overlay_out: Optional[str] = typer.Option(None, help="Overlay image output path."),
    save_json: Optional[str] = typer.Option(None, help="JSON result output path."),
    confidence_threshold: float = typer.Option(0.70),
):
    """Perceive a still image, plan the fold, save an overlay + JSON result."""
    from terafold.demo import run_demo_image

    res = run_demo_image(
        image_path=image, mode=mode, task_path=task, checkpoint=checkpoint,
        calibration=calibration or homography, marker_map=marker_map, debug_markers=debug_markers,
        overlay_out=overlay_out, save_json=save_json, confidence_threshold=confidence_threshold,
        on_log=_echo,
    )
    _ok(f"demo-image complete (perception={res['perception']}, conf={res['confidence']:.2f}).")


@app.command("demo-today")
def demo_today_cmd(
    camera_index: int = typer.Option(0, help="Webcam index (from list-cameras)."),
    camera: str = typer.Option("opencv", help="opencv | mock (mock renders a marked towel)."),
    mode: str = typer.Option("markers", help="markers | model | claude."),
    task: str = typer.Option("configs/task_fold_towel_half.yaml"),
    robot: str = typer.Option("mock", help="mock | generic | so101 | learm."),
    checkpoint: Optional[str] = typer.Option(None, help="Keypoint model checkpoint (mode=model)."),
    calibration: Optional[str] = typer.Option(None, help="Homography JSON for metric planning."),
    out: Optional[str] = typer.Option(None, help="Episodes root (default data/episodes/<task>_demo)."),
    overlay_out: Optional[str] = typer.Option(None, help="Overlay image path."),
    save_frame: Optional[str] = typer.Option(None, help="Save the captured frame here."),
    save_json: Optional[str] = typer.Option(None, help="Save the JSON result here."),
    marker_map: Optional[str] = typer.Option(None, help="e.g. red:top_left,blue:top_right,..."),
    debug_markers: bool = typer.Option(False, "--debug-markers", help="Save per-color masks."),
    confidence_threshold: float = typer.Option(0.70),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Force dry-run (no motion)."),
    enable_motion: bool = typer.Option(False, "--enable-motion", help="Allow physical motion."),
    i_understand: bool = typer.Option(
        False, "--i-understand-this-moves-hardware", help="Required second motion flag."
    ),
):
    """Live demo: camera -> perceive -> plan -> dry-run / refuse real motion -> record.

    Safe by default (dry-run). Real motion needs ALL of: trusted perception above
    --confidence-threshold, a calibration, a verified adapter, a prior dry-run,
    --robot so101 --enable-motion --i-understand-this-moves-hardware.
    """
    from terafold.demo import run_demo_today

    try:
        res = run_demo_today(
            task_path=task, camera=camera, camera_index=camera_index, mode=mode, robot=robot,
            checkpoint=checkpoint, calibration=calibration, out=out, overlay_out=overlay_out,
            save_frame=save_frame, save_json=save_json, marker_map=marker_map,
            debug_markers=debug_markers, confidence_threshold=confidence_threshold,
            dry_run=dry_run, enable_motion=enable_motion, acknowledge=i_understand, on_log=_echo,
        )
    except Exception as exc:
        from terafold.robot.safety import SafetyError

        if isinstance(exc, SafetyError):
            _err(f"Safety: {exc}")
            raise typer.Exit(2)
        _err(f"demo-today failed: {exc}")
        raise typer.Exit(1)

    status = res.get("status")
    if status == "ok":
        _ok(f"demo-today complete ({'REAL' if res.get('real_motion') else 'dry-run'}).")
    elif status == "real_motion_refused":
        _err("Real motion refused — unmet requirements:")
        for m in res.get("missing", []):
            _echo(f"   - {m}")
        _echo("Dry-run plan + overlay + JSON + episode were still saved.")
    elif status == "real_robot_unavailable":
        _err(f"Robot bring-up unavailable: {res.get('robot_error')}")
    elif status == "aborted":
        _err(f"Aborted: {res.get('stop_reason')}")
    elif status == "error":
        _err(f"Error: {res.get('error')}")
        raise typer.Exit(1)
    if res.get("episode_dir"):
        _echo(f"  episode : {res['episode_dir']}")
    if res.get("overlay_path"):
        _echo(f"  overlay : {res['overlay_path']}")
    if res.get("json_path"):
        _echo(f"  json    : {res['json_path']}")


@app.command("robot-scan")
def robot_scan_cmd(
    probe_readonly: bool = typer.Option(False, "--probe-readonly", help="Open ports and READ only."),
    save_json: Optional[str] = typer.Option(None, help="Save the port report JSON here."),
):
    """List likely serial ports for an unknown arm. NEVER sends motor commands."""
    from terafold.robot.discovery import scan_serial_ports

    report = scan_serial_ports(probe_readonly=probe_readonly)
    if not report["ports"]:
        _echo("No serial ports found. Plug the arm in (and install vendor drivers).")
    for p in report["ports"]:
        flag = " <-- likely robot" if p["likely_robot"] else ""
        _echo(f"  {p['device']}  {p.get('description') or ''}{flag}")
        if p.get("probe"):
            _echo(f"      probe: {p['probe']}")
    _echo("\nRecommended next steps:")
    for step in report["next_steps"]:
        _echo(f"  - {step}")
    _echo(f"\n{report['note']}")
    if save_json:
        from terafold.data.episode_schema import write_json

        write_json(save_json, report)
        _ok(f"Saved -> {save_json}")


@app.command("robot-info-template")
def robot_info_template_cmd(
    out: str = typer.Option("runs/robot_scan/robot_info_template.md"),
):
    """Write a fill-in checklist to identify and safely wire your arm."""
    import os

    from terafold.robot.discovery import robot_info_template_markdown

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(robot_info_template_markdown())
    _ok(f"Wrote robot info template -> {out}")


@app.command("export-trajectory")
def export_trajectory_cmd(
    plan_json: str = typer.Option(..., help="A FoldPlan / demo result JSON."),
    out: str = typer.Option(..., help="Output CSV (or .json)."),
    fmt: str = typer.Option("csv", help="csv | json"),
):
    """Export a planned trajectory to CSV/JSON for manual / vendor-software testing."""
    from terafold.data.trajectory_export import export_trajectory

    res = export_trajectory(plan_json, out, fmt=fmt)
    _ok(f"Exported {res['rows']} waypoints -> {res['out']} ({res['format']})")


@app.command("calibrate-table-from-image")
def calibrate_table_from_image_cmd(
    image: str = typer.Option(..., help="Calibration image (for reference)."),
    out: str = typer.Option("data/calib/homography.json"),
    image_points: Optional[str] = typer.Option(
        None, help="4+ pixel pts 'u1,v1;u2,v2;...' (else read --points-json)."
    ),
    table_points: Optional[str] = typer.Option(
        None, help="4+ table metres 'x1,y1;x2,y2;...'."
    ),
    points_json: Optional[str] = typer.Option(
        None, help="JSON file with {image_pts:[...], table_pts:[...]}."
    ),
):
    """Solve + save the image->table homography from manual point correspondences."""
    from terafold.camera.table_calibration import calibrate_table_from_image, parse_xy_list

    if points_json:
        from terafold.data.episode_schema import read_json

        d = read_json(points_json)
        img_pts, tbl_pts = d["image_pts"], d["table_pts"]
    elif image_points and table_points:
        img_pts, tbl_pts = parse_xy_list(image_points), parse_xy_list(table_points)
    else:
        _err("Provide --image-points and --table-points, or --points-json.")
        raise typer.Exit(1)

    try:
        res = calibrate_table_from_image(img_pts, tbl_pts, out, image_path=image)
    except Exception as exc:
        _err(f"Calibration failed: {exc}")
        raise typer.Exit(1)
    _ok(f"Saved homography -> {res['out']}  (reprojection error {res['reprojection_error_m']:.4f} m)")


@app.command("record-demo")
def record_demo_cmd(
    task: str = typer.Option("configs/task_fold_towel_half.yaml"),
    robot: str = typer.Option("mock", help="mock | learm | so101"),
    camera: str = typer.Option("webcam", help="webcam | mock"),
    out: Optional[str] = typer.Option(None, help="Episodes root (default data/episodes/<task>)."),
    operator: str = typer.Option("unknown"),
    enable_motion: bool = typer.Option(False, "--enable-motion", help="Allow physical motion."),
    i_understand: bool = typer.Option(
        False, "--i-understand-this-moves-hardware", help="Required second motion flag."
    ),
):
    """Record a folding demonstration episode (mock by default; real needs both motion flags)."""
    from terafold.config.schema import CameraConfig
    from terafold.data.recorder import record_demo
    from terafold.planning.towel_half_fold import TowelHalfFoldPlanner

    ft = _load_task(task)
    dry = not (enable_motion and i_understand)
    if not dry:
        from terafold.robot.safety import require_motion_enabled

        require_motion_enabled(enable_motion, i_understand)

    # Build camera.
    cam_type = "mock" if camera == "mock" else "opencv"
    if cam_type == "mock":
        from terafold.camera.mock_camera import MockCamera

        cam = MockCamera(CameraConfig(type="mock"))
    else:
        from terafold.camera.opencv_camera import OpenCVCamera

        cam = OpenCVCamera(CameraConfig(type="opencv"))

    # Build robot.
    if robot == "mock":
        from terafold.robot.mock_robot import MockRobot

        rob = MockRobot(workspace=ft.workspace, dry_run=dry)
    elif robot == "learm":
        from terafold.config.load import load_robot_config
        from terafold.robot.learm_adapter import LeArmAdapter

        rob = LeArmAdapter(load_robot_config("configs/robot_learm_template.yaml"), dry_run=dry)
    else:
        from terafold.config.load import load_robot_config
        from terafold.robot.so101_adapter import SO101Adapter

        rob = SO101Adapter(load_robot_config("configs/robot_so101_template.yaml"), dry_run=dry)

    planner = TowelHalfFoldPlanner(ft)
    task_root = out or f"data/episodes/{ft.name}"
    ep = record_demo(ft, rob, cam, planner, task_root=task_root, operator=operator, dry_run=dry)
    _ok(f"Recorded episode -> {ep}")


@app.command("replay-episode")
def replay_episode_cmd(episode: str = typer.Option(..., help="Episode directory.")):
    """Replay / summarize a recorded episode."""
    from terafold.data.replay import summarize_episode

    _echo(summarize_episode(episode))


@app.command("label-episode")
def label_episode(
    episode: str = typer.Option(...),
    success: bool = typer.Option(..., help="Was the fold successful?"),
    failure_mode: str = typer.Option("none"),
    notes: str = typer.Option(""),
):
    """Attach a human success/failure label to an episode."""
    import os

    from terafold.data.episode_schema import (
        EpisodePaths,
        EpisodeResult,
        read_json,
        write_json,
    )

    paths = EpisodePaths(episode)
    result = EpisodeResult(success=success, failure_mode=failure_mode, notes=notes)
    write_json(paths.result, result.to_dict())
    if os.path.exists(paths.metadata):
        meta = read_json(paths.metadata)
        meta["success"] = success
        meta["failure_mode"] = failure_mode
        write_json(paths.metadata, meta)
    _ok(f"Labeled {episode}: success={success}, failure_mode={failure_mode}")


@app.command("summarize-episodes")
def summarize_episodes(episodes: str = typer.Option(..., help="Episodes root directory.")):
    """Summarize all episodes under a task directory."""
    from terafold.data.episode_schema import list_episode_dirs
    from terafold.data.replay import summarize_episode

    dirs = list_episode_dirs(episodes)
    _echo(f"{len(dirs)} episodes under {episodes}")
    for p in dirs:
        _echo(summarize_episode(p.root))


@app.command("export-lerobot")
def export_lerobot_cmd(
    episodes: str = typer.Option(..., help="Episodes root directory."),
    out: str = typer.Option(..., help="LeRobot dataset output directory."),
    fps: int = typer.Option(10),
):
    """Export recorded episodes to a LeRobot-compatible dataset."""
    from terafold.data.export_lerobot import export_to_lerobot, print_lerobot_training_commands

    summary = export_to_lerobot(episodes, out, fps=fps)
    _ok(f"Exported: {summary}")
    _echo(print_lerobot_training_commands(out))


@app.command("data-sources")
def data_sources_cmd():
    """List known public cloth/folding data sources and how to use them."""
    from terafold.data.sources import format_data_sources

    _echo(format_data_sources())


def _print_inspect_report(report: dict) -> None:
    order = [
        "repo_id", "in_registry", "available", "online", "embodiment", "est_size",
        "modalities", "image_keys", "video_keys", "num_episodes", "num_frames",
        "state_dim", "action_dim", "policy_compatible", "action_space_matches_our_robot",
        "recommended_use", "action_advice", "streaming_checked", "streaming_available",
        "streaming_note", "notes", "error", "guidance",
    ]
    for k in order:
        v = report.get(k)
        if v not in (None, "", []):
            _echo(f"  {k}: {v}")


@app.command("inspect-hf-dataset")
def inspect_hf_dataset_cmd(
    repo_id: str = typer.Option(..., help="HuggingFace dataset repo id."),
    streaming: bool = typer.Option(
        False, "--streaming", help="Peek one streamed example (no download)."
    ),
):
    """Inspect a HuggingFace/LeRobot dataset and judge fit for our robot."""
    from terafold.data.inspect_hf_dataset import inspect_hf_dataset

    report = inspect_hf_dataset(repo_id, streaming=streaming)
    _print_inspect_report(report)


def _print_hf_result(res: dict) -> None:
    """Print a streaming/sample/cache/import result; exit non-zero on failure."""
    status = res.get("status")
    if status == "ok":
        _ok(
            f"{res.get('repo_id')}: {res.get('saved', res.get('episodes', 0))} "
            f"{'samples' if 'saved' in res else 'episodes'} -> {res.get('out')}"
        )
        for k in ("source_format", "camera", "backend", "episodes_used", "episodes_downloaded", "rows"):
            if res.get(k) is not None:
                _echo(f"  {k}: {res[k]}")
        if res.get("note"):
            _echo(f"  note: {res['note']}")
        return
    if status == "missing_dependency":
        _err(res["message"])
        _err(f"Run:  {res['install_command']}")
        raise typer.Exit(1)
    if status == "needs_video_decoder":
        _err(res.get("message", "Native LeRobot video decoding is needed."))
        if res.get("available_cameras"):
            _echo(f"  cameras: {res['available_cameras']}")
        _err(f"Run:  {res.get('install_command', 'pip install av')}")
        raise typer.Exit(3)
    if status == "no_images_found":
        _err(res.get("message", "No image fields found."))
        _echo(f"  row keys: {res.get('row_keys')}")
        raise typer.Exit(1)
    if status == "requires_allow_large_download":
        _err(res.get("message", "This would require a large download."))
        _err("Re-run with --allow-large-download to permit it (or set a smaller --max-*).")
        raise typer.Exit(2)
    if status == "offline":
        _err(f"Offline / unreachable: {res.get('error')}")
        if res.get("hint"):
            _err(res["hint"])
        raise typer.Exit(1)
    _err(f"Failed: {res}")
    raise typer.Exit(1)


@app.command("debug-hf-row")
def debug_hf_row_cmd(
    repo_id: str = typer.Option(...),
    num_rows: int = typer.Option(3, help="How many streamed rows to inspect."),
    streaming: bool = typer.Option(True, "--streaming/--no-streaming"),
    split: str = typer.Option("train"),
):
    """Inspect streamed row schema: keys, types, image candidates, sample decisions."""
    from terafold.data.hf_streaming import debug_hf_row

    report = debug_hf_row(repo_id, num_rows=num_rows, streaming=streaming, split=split)
    if report.get("status") == "missing_dependency":
        _err(report["message"])
        _err(f"Run:  {report['install_command']}")
        raise typer.Exit(1)
    if report.get("status") not in ("ok", None):
        _err(f"{report.get('status')}: {report.get('error')}")
        raise typer.Exit(1)
    _echo(f"repo_id: {report['repo_id']}")
    _echo(f"info image/video keys: {report.get('info_image_or_video_keys')}")
    _echo(f"video_path template  : {report.get('video_path_template')}")
    _echo(f"video backend        : {report.get('video_backend')}   total_episodes: {report.get('total_episodes')}")
    for row in report.get("rows", []):
        _echo("")
        _echo(f"── row {row['index']} ── keys: {row['keys']}")
        for k, d in row["fields"].items():
            tag = ""
            if d.get("image_candidate"):
                if d.get("coerce_ok"):
                    tag = f"  [IMAGE ✓ {d.get('coerce_kind')} -> {d.get('coerced_shape')}]"
                else:
                    tag = f"  [image? ✗ {d.get('coerce_kind')}: {d.get('coerce_reason')}]"
            shape = d.get("shape") or d.get("len") or d.get("nested_keys") or d.get("pil") or ""
            _echo(f"    {k}: {d['type']} {shape}{tag}")
        _echo(f"    state_dim={row['state_dim']} action_dim={row['action_dim']} "
              f"image_usable_in_row={row['image_usable_in_row']}")
        _echo(f"    -> {row['sampling_decision']}")


@app.command("sample-hf-dataset")
def sample_hf_dataset_cmd(
    repo_id: str = typer.Option(..., help="HuggingFace dataset repo id."),
    max_samples: int = typer.Option(500, help="Max examples to stream (bounded)."),
    out: Optional[str] = typer.Option(None, help="Output dir (default data/public_samples/<name>)."),
    split: str = typer.Option("train"),
    convert: bool = typer.Option(
        True, "--convert/--no-convert", help="Convert images to TeraFold perception format."
    ),
    image_key: Optional[str] = typer.Option(
        None, help="Force a specific image/camera field (e.g. observation.images.top)."
    ),
    frames_per_episode: int = typer.Option(
        0, help="(video datasets) max frames per episode; 0 = auto."
    ),
    frame_stride: int = typer.Option(
        0, help="(video datasets) decode every Nth frame; 0 = ~1 fps."
    ),
    allow_large_download: bool = typer.Option(False, "--allow-large-download"),
):
    """Stream a bounded image sample from a public dataset (NO full download).

    For LeRobot *video* datasets the frames live in MP4s, so a bounded set of
    episode videos is downloaded and decoded (needs a video backend such as av).
    """
    from terafold.data.hf_streaming import sample_hf_dataset

    res = sample_hf_dataset(
        repo_id, max_samples=max_samples, out=out, split=split,
        convert=convert, image_key=image_key, frames_per_episode=frames_per_episode,
        frame_stride=frame_stride, allow_large_download=allow_large_download,
    )
    _print_hf_result(res)


@app.command("cache-hf-subset")
def cache_hf_subset_cmd(
    repo_id: str = typer.Option(...),
    max_episodes: int = typer.Option(20, help="Max episodes to cache (bounded)."),
    out: Optional[str] = typer.Option(None, help="Output dir (default data/cache/<name>)."),
    split: str = typer.Option("train"),
    allow_large_download: bool = typer.Option(False, "--allow-large-download"),
):
    """Cache a bounded subset of episodes locally by streaming (NO full download)."""
    from terafold.data.hf_streaming import cache_hf_subset

    res = cache_hf_subset(
        repo_id, max_episodes=max_episodes, out=out, split=split,
        allow_large_download=allow_large_download,
    )
    _print_hf_result(res)


@app.command("import-hf-lerobot")
def import_hf_lerobot_cmd(
    repo_id: str = typer.Option(...),
    max_episodes: int = typer.Option(20, help="Max episodes to import (bounded)."),
    out: Optional[str] = typer.Option(None, help="Output dir (default data/public/<name>)."),
    split: str = typer.Option("train"),
    allow_large_download: bool = typer.Option(False, "--allow-large-download"),
):
    """Import a bounded subset of a LeRobot dataset into TeraFold's format.

    Foreign actions are imported for analysis/representation ONLY — never for
    direct rollout on the LeArm.
    """
    from terafold.data.hf_streaming import import_hf_lerobot

    res = import_hf_lerobot(
        repo_id, max_episodes=max_episodes, out=out, split=split,
        allow_large_download=allow_large_download,
    )
    _print_hf_result(res)


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


@app.command("train-residual")
def train_residual_cmd(
    episodes: str = typer.Option(..., help="Episodes dir with recorded demos."),
    out: str = typer.Option(...),
    epochs: int = typer.Option(100),
):
    """Train the residual correction model (Stage 3) over geometry."""
    from terafold.planning.train_residual import train_residual

    try:
        hist = train_residual(episodes, out, epochs=epochs)
    except ImportError as e:
        _err(str(e))
        _err("Install the ML extra:  pip install -e '.[ml]'")
        raise typer.Exit(1)
    _ok(f"Trained residual: {hist}")


@app.command("train-act")
def train_act_cmd(
    dataset: str = typer.Option(...),
    out: str = typer.Option(...),
    run: bool = typer.Option(False, help="Actually launch training (needs lerobot+GPU)."),
):
    """Prepare/launch ACT training on an exported LeRobot dataset."""
    from terafold.learning.train_act_wrapper import train_act

    res = train_act(dataset, out, run=run)
    _echo(res.get("message", str(res)))


@app.command("train-smolvla")
def train_smolvla_cmd(
    dataset: str = typer.Option(...),
    out: str = typer.Option(...),
    run: bool = typer.Option(False),
):
    """Prepare/launch SmolVLA training on an exported LeRobot dataset."""
    from terafold.learning.train_smolvla_wrapper import train_smolvla

    res = train_smolvla(dataset, out, run=run)
    _echo(res.get("message", str(res)))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@app.command("score-fold")
def score_fold_cmd(
    before: str = typer.Option(...),
    after: str = typer.Option(...),
    checkpoint: Optional[str] = typer.Option(None),
):
    """Score whether a fold succeeded from before/after images."""
    from terafold.eval.score_fold import score_fold
    from terafold.vision.imageio import imread
    from terafold.vision.infer_keypoints import get_keypoint_predictor

    res = score_fold(imread(before), imread(after), predictor=get_keypoint_predictor(checkpoint))
    for k, v in res.items():
        _echo(f"  {k}: {v}")


@app.command("benchmark")
def benchmark_cmd(
    episodes: Optional[str] = typer.Option(None, help="Episodes dir to benchmark."),
):
    """Aggregate fold success metrics over recorded episodes."""
    from terafold.eval.benchmark import benchmark_episodes, print_benchmark

    if not episodes:
        _err("Provide --episodes <dir>.")
        raise typer.Exit(1)
    report = benchmark_episodes(episodes)
    _echo(print_benchmark(report))


@app.command("sim-fold")
def sim_fold_cmd(
    plan_json: str = typer.Option(..., help="A FoldPlan / demo result JSON."),
    out: str = typer.Option("runs/sim/fold_demo.mp4", help="Output video (.mp4 / .gif)."),
    level: str = typer.Option("ee-only", help="ee-only | arm-ik | cloth-proxy | so101-real."),
    view: str = typer.Option("iso", help="Camera (MuJoCo): top | iso | side | follow-ee | gripper."),
    all_views: bool = typer.Option(False, "--all-views", help="Render iso/top/side/gripper (MuJoCo)."),
    save_frames: bool = typer.Option(False, "--save-frames", help="Also export PNG frames."),
    slowmo: bool = typer.Option(False, "--slowmo", help="Slower playback for review."),
    waypoint_dots: Optional[bool] = typer.Option(
        None, "--waypoint-dots/--no-waypoint-dots",
        help="Show planned-waypoint dots (default on for so101-real)."),
    fps: int = typer.Option(20),
    max_seconds: float = typer.Option(10.0, help="Cap the (time-compressed) video length."),
    width: int = typer.Option(640),
    height: int = typer.Option(480),
    use_mujoco: bool = typer.Option(False, "--use-mujoco", help="Force the MuJoCo 3D scene."),
):
    """Simulate the planned fold (virtual SO-101 / end-effector). SIMULATION ONLY.

    ee-only / cloth-proxy work without MuJoCo (2D top-down renderer). arm-ik and
    so101-real need MuJoCo:  python3 -m pip install -e ".[sim]".  so101-real is a
    realistic 5-DOF SO-101 approximation. No motor commands are ever sent.
    """
    from terafold.sim.fold_sim import run_sim_fold

    res = run_sim_fold(
        plan_json, out=out, level=level, view=view, all_views=all_views,
        save_frames=save_frames, slowmo=slowmo, waypoint_dots=waypoint_dots,
        fps=fps, max_seconds=max_seconds, width=width, height=height,
        use_mujoco=use_mujoco, on_log=_echo,
    )
    status = res.get("status")
    if status == "missing_dependency":
        _err(res["message"])
        _err(f"Run:  {res['install_command']}")
        raise typer.Exit(1)
    if status == "error":
        _err(res["error"])
        if res.get("hint"):
            _echo(res["hint"])
        raise typer.Exit(1)
    _ok(f"sim-fold complete ({res['renderer']}, {res['frames']} frames, ~{res['duration_s']}s).")
    if res.get("so101_assets"):
        a = res["so101_assets"]
        which = (f"official {a['type']} @ {a['path']}" if a["found"]
                 else "approximation (no official assets found)")
        _echo(f"  SO-101 assets : {which}")
    if res.get("all_views"):
        for v, p in res["views"].items():
            _echo(f"  {v:10s}: {p}")
    else:
        _echo(f"  video : {res.get('video') or res.get('out')}")
    for v, d in (res.get("frames_dirs") or {}).items():
        _echo(f"  frames[{v}]: {d}")
    _echo(f"  meta  : {res.get('meta_json')}")
    _echo(f"  {res.get('note')}")


def main() -> None:  # console-script friendly
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
