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


# ----------------------------------------------------------------------
# Physical bus-servo arm: SAFE real-motion workflow (default dry-run).
# Real motion is double-gated AND requires a confirmed protocol (none yet).
# ----------------------------------------------------------------------


@app.command("robot-status")
def robot_status_cmd(
    robot: str = typer.Option(..., help="Robot config name (e.g. physical_7dof_waveshare)."),
    port: str = typer.Option("auto", help="Serial port, or 'auto' (use the config's)."),
    probe: bool = typer.Option(
        False, "--probe/--no-probe",
        help="Open the port READ-ONLY and ping/read to confirm the protocol. Never moves."),
    unlock_contact: bool = typer.Option(
        False, "--unlock-contact", help="Operator contact-unlock flag (still gated)."),
    out: Optional[str] = typer.Option(None, help="Also write the Markdown report here."),
    save_json: Optional[str] = typer.Option(None, help="Also write the JSON snapshot here."),
):
    """Am-I-safe-to-proceed check: the robot's capabilities + unlocked safety level.

    Read-only and offline by default (no serial access). Add ``--probe`` to confirm
    the protocol with a read-only ping/read. Prints the unlock ladder with the exact
    reason each higher level is blocked.
    """
    from terafold.robot.capabilities import probe_capabilities

    caps = probe_capabilities(
        robot, port=None if port == "auto" else port, do_probe=probe,
        contact_unlock=unlock_contact,
    )
    from terafold.robot.safety_state import SafetyLevel

    _echo(caps.report_markdown())
    _echo("")
    suffix = "" if caps.contact_allowed else "  (contact folding LOCKED)"
    _ok(f"Unlocked: {SafetyLevel(caps.unlocked_level).label}{suffix}")
    if out:
        import os as _os

        _os.makedirs(_os.path.dirname(_os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(caps.report_markdown())
        _ok(f"Wrote report -> {out}")
    if save_json:
        from terafold.data.episode_schema import write_json

        write_json(save_json, caps.to_dict())
        _ok(f"Wrote snapshot -> {save_json}")


def _load_arm(robot: str, port: str):
    from terafold.robot.arm_config import load_arm_config
    from terafold.robot.waveshare_bus_servo import WaveshareBusServoAdapter

    cfg = load_arm_config(robot)
    if port and port != "auto":
        cfg.port = port
    adapter = WaveshareBusServoAdapter(
        port=cfg.port, baudrate=cfg.baud, baud_candidates=cfg.baudrate_candidates,
        dof=cfg.dof, joint_names=cfg.joint_names,
    )
    return cfg, adapter


@app.command("robot-probe")
def robot_probe_cmd(
    robot: str = typer.Option(..., help="Robot config name (e.g. physical_7dof_waveshare)."),
    port: str = typer.Option("auto", help="Serial port, or 'auto'."),
    read_only: bool = typer.Option(True, "--read-only/--no-read-only",
                                   help="Read-only (default). Never moves motors."),
    allow_torque: bool = typer.Option(False, "--allow-torque",
                                      help="Explicitly allow torque (still refused: protocol unconfirmed)."),
):
    """Identify the serial port + Waveshare adapter. READ-ONLY; never moves motors."""
    cfg, adapter = _load_arm(robot, port)
    _echo(f"Robot: {cfg.robot_name}  dof={cfg.dof}  adapter={cfg.adapter}")
    report = adapter.list_ports()
    if not report["ports"]:
        _echo("No serial ports found. Plug in the USB-C cable + DC 9-12.6V power.")
    for p in report["ports"]:
        flag = " <-- likely robot" if p["likely_robot"] else ""
        _echo(f"  {p['device']}  {p.get('description') or ''}{flag}")
    ident = adapter.identify()
    _echo("\nAdapter identification (USB descriptors only, no bytes sent):")
    _echo(f"  device={ident['device']}  likely_adapter={ident['likely_adapter']} "
          f"(conf {ident['confidence']:.1f})  product={ident.get('product')}")
    _echo(f"  {ident['note']}")
    if allow_torque:
        _err("Torque was requested but is REFUSED: protocol not confirmed.")
    _echo(f"\n  protocol_confirmed={adapter.protocol_confirmed}  (motion disabled)")
    _echo("  Next: `terafold servo-scan` (needs a confirmed protocol) or "
          "`terafold robot-info-template` to record your hardware.")


@app.command("servo-scan")
def servo_scan_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    port: str = typer.Option("auto", help="Serial port, or 'auto'."),
    ids: str = typer.Option("", help="ID range/list, e.g. '1-30' (overrides --id-min/--id-max)."),
    id_min: int = typer.Option(1), id_max: int = typer.Option(30),
    read_only: bool = typer.Option(True, "--read-only/--no-read-only",
                                   help="Always read-only; this flag is informational."),
    out: Optional[str] = typer.Option(None, help="Scan output dir (default runs/servo_scan/<ts>)."),
):
    """Read-only servo-ID scan: ping + read each ID, write scan.json + summary.md.

    NEVER writes a servo. Refuses (non-zero) if the protocol cannot be confirmed.
    """
    from terafold.robot.servo_scan import parse_id_range, run_servo_scan

    if ids.strip():
        rng = parse_id_range(ids)
        if rng:
            id_min, id_max = min(rng), max(rng)
    res = run_servo_scan(robot, port=None if port == "auto" else port,
                         id_min=id_min, id_max=id_max, out_dir=out)
    if not res.get("protocol_confirmed"):
        _err("protocol not confirmed: no ping/read on the bus. Check power (DC 9-12.6V), "
             "the USB-C cable, the port, and that pyserial + the vendor scservo_sdk are installed.")
        _echo(f"  (read-only scan still written -> {res['scan_json']})")
        raise typer.Exit(1)
    _ok(f"Found servo IDs: {res['found_ids']}  (missing expected: {res['missing_ids']})")
    _echo(f"  scan    : {res['scan_json']}")
    _echo(f"  summary : {res['summary_md']}")
    if res["missing_ids"]:
        _echo("  Missing IDs — physical debugging:")
        for s in res.get("missing_id_debug", []):
            _echo(f"    - {s}")


@app.command("servo-nudge")
def servo_nudge_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    port: str = typer.Option("auto"),
    servo_id: int = typer.Option(..., help="Which servo to nudge."),
    delta_deg: float = typer.Option(3.0, help="Tiny angle delta (default 3, hard limit 5)."),
    speed: str = typer.Option("slow", help="very_slow | slow."),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
    dangerous_allow_larger_motion: bool = typer.Option(False, "--dangerous-allow-larger-motion"),
    allow_open_loop_nudge: bool = typer.Option(False, "--allow-open-loop-nudge"),
):
    """Move ONE servo by a tiny amount, then return it. Default = refuse (dry-run)."""
    from terafold.robot.real_motion import (check_nudge_delta, countdown, motion_logger,
                                            print_estop_instructions)
    from terafold.robot.safety import SafetyError, require_motion_enabled
    from terafold.robot.waveshare_bus_servo import PROTOCOL_NEXT_STEPS, PROTOCOL_REFUSAL

    cfg, adapter = _load_arm(robot, port)
    log = motion_logger("servo_nudge")
    log.log("servo_nudge_request", {"robot": robot, "servo_id": servo_id, "delta_deg": delta_deg,
                                     "speed": speed, "enable_motion": enable_motion,
                                     "acknowledge": acknowledge})
    # 1. Two-flag human gate.
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as e:
        _err(str(e))
        log.log("refused", {"reason": "flags"})
        raise typer.Exit(1)
    # 2. Delta limits.
    try:
        check_nudge_delta(delta_deg, cfg.hard_delta_limit_deg, dangerous_allow_larger_motion)
    except SafetyError as e:
        _err(str(e))
        log.log("refused", {"reason": "delta"})
        raise typer.Exit(1)
    if abs(delta_deg) > cfg.max_delta_per_test_deg:
        _echo(f"[warn] delta {delta_deg:.1f} deg exceeds the default tiny max "
              f"{cfg.max_delta_per_test_deg:.1f} deg (allowed, but be careful).")
    # 3. Closed-loop check.
    adapter.open()
    pos = adapter.read_position(servo_id)
    if pos is None and not allow_open_loop_nudge:
        _err("Cannot read the current servo position (protocol unconfirmed). Refusing "
             "to move blind. Pass --allow-open-loop-nudge for a tiny low-speed pulse "
             "ONLY if you accept the risk.")
        log.log("refused", {"reason": "no_position"})
        adapter.close()
        raise typer.Exit(1)
    if pos is None:
        _echo("[warn] OPEN-LOOP nudge: current position is UNKNOWN. This is risky — "
              "keep a hand on the power switch.")
    # 4. Protocol gate (the real safety wall — no verified protocol exists yet).
    if not adapter.supports_motion:
        _err(PROTOCOL_REFUSAL)
        for s in PROTOCOL_NEXT_STEPS:
            _echo(f"  - {s}")
        log.log("refused", {"reason": "protocol_unconfirmed"})
        adapter.close()
        raise typer.Exit(1)
    # 5. Verified-backend path (only reachable once a real protocol is wired).
    print_estop_instructions(_echo)
    countdown(5, _echo)
    start = pos if pos is not None else 0.0
    try:
        adapter.write_position(servo_id, start + delta_deg, speed=speed)
        adapter.write_position(servo_id, start, speed=speed)  # return to start
        log.log("moved", {"servo_id": servo_id, "delta_deg": delta_deg})
        _ok(f"Nudged servo {servo_id} by {delta_deg} deg and returned.")
    finally:
        adapter.close()


@app.command("real-ghost-fold")
def real_ghost_fold_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    plan_json: str = typer.Option(..., help="A FoldPlan / demo result JSON."),
    height_clearance_m: float = typer.Option(0.10, help="Min height above the table (>=0.10)."),
    speed: str = typer.Option("slow", help="very_slow | slow."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Trace the fold path IN THE AIR above the table. Never touches the towel."""
    from terafold.robot.real_motion import ghost_fold_trajectory, motion_logger
    from terafold.robot.safety import SafetyError, require_motion_enabled

    cfg, adapter = _load_arm(robot, port="auto")
    clearance = max(height_clearance_m, cfg.table_clearance_m)
    ghost = ghost_fold_trajectory(plan_json, height_clearance_m=clearance)
    log = motion_logger("real_ghost_fold")
    log.log("ghost_request", {"robot": robot, "clearance_m": clearance, "speed": speed,
                              "dry_run": dry_run, "enable_motion": enable_motion})
    _echo(f"Ghost fold (AIR-ONLY): {ghost['num_waypoints']} waypoints, "
          f"min_z={ghost['min_z']}m (clearance {clearance}m), touches_table={ghost['touches_table']}")
    for w in ghost["waypoints"]:
        _echo(f"  {w['phase']:<10} xyz={w['xyz']} gripper={w['gripper']}")
    _echo(f"  {ghost['note']}")

    real = enable_motion or acknowledge or not dry_run
    if not real:
        _ok("DRY-RUN: nothing moved. To execute above the table, you need joint-space "
            "waypoints — use `teach-ghost-fold` then `replay-joint-demo`.")
        return
    # Real motion requested.
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as e:
        _err(str(e))
        raise typer.Exit(1)
    # Kinematics are unknown AND the protocol is unconfirmed -> refuse, route to teach.
    _err("Cannot execute a Cartesian ghost fold: the arm's kinematics are unknown and "
         "the bus-servo protocol is not confirmed. Record a joint-space ghost fold with "
         "`teach-ghost-fold`, then `replay-joint-demo` (very_slow, above the table).")
    log.log("refused", {"reason": "no_kinematics_or_protocol"})
    raise typer.Exit(1)


@app.command("teach-ghost-fold")
def teach_ghost_fold_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    out: str = typer.Option("data/real_demos/ghost_fold_001.json"),
    port: str = typer.Option("auto"),
):
    """Manually pose the arm at each waypoint and record joint positions."""
    from terafold.robot.real_motion import (GHOST_FOLD_POSES, motion_logger,
                                            record_teach_demo, save_joint_demo)

    cfg, adapter = _load_arm(robot, port)
    log = motion_logger("teach_ghost_fold")
    _echo("MANUAL TEACH MODE — move the arm BY HAND (torque off / loose).")
    _echo("Keep every pose ABOVE the table; do NOT touch the cloth.")
    _echo(f"Poses: {', '.join(GHOST_FOLD_POSES)}\n")
    demo = record_teach_demo(adapter, robot=cfg.robot_name, dof=cfg.dof,
                             joint_names=cfg.joint_names, log=_echo)
    save_joint_demo(out, demo)
    log.log("teach_saved", {"out": out, "has_positions": demo.has_positions()})
    _ok(f"Saved demo -> {out}")
    if not demo.has_positions():
        _echo("[warn] No servo positions were captured (protocol unconfirmed). This is a "
              "skeleton; confirm the protocol before it can be replayed.")
    _echo("Replay (dry-run):  terafold replay-joint-demo --robot "
          f"{robot} --demo {out} --speed very_slow --dry-run")


@app.command("replay-joint-demo")
def replay_joint_demo_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    demo: str = typer.Option(..., help="A joint-space demo JSON from teach-ghost-fold."),
    port: str = typer.Option("auto"),
    speed: str = typer.Option("very_slow"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Replay a taught joint-space ghost fold. Default = dry-run (prints only)."""
    from terafold.robot.real_motion import load_joint_demo, motion_logger
    from terafold.robot.safety import SafetyError, require_motion_enabled
    from terafold.robot.waveshare_bus_servo import PROTOCOL_REFUSAL

    cfg, adapter = _load_arm(robot, port)
    d = load_joint_demo(demo)
    log = motion_logger("replay_joint_demo")
    _echo(f"Demo: {d.robot}  {len(d.waypoints)} waypoints  has_positions={d.has_positions()}")
    for w in d.waypoints:
        _echo(f"  {w['name']:<16} positions={w.get('servo_positions')}")

    real = enable_motion or acknowledge or not dry_run
    if not real:
        _ok(f"DRY-RUN at speed={speed}: nothing moved.")
        return
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as e:
        _err(str(e))
        raise typer.Exit(1)
    if not d.has_positions():
        _err("Demo has no recorded servo positions — re-teach with a confirmed protocol.")
        raise typer.Exit(1)
    if not adapter.supports_motion:
        _err(PROTOCOL_REFUSAL)
        log.log("refused", {"reason": "protocol_unconfirmed"})
        raise typer.Exit(1)
    _err("(verified-backend replay path not reachable: no confirmed protocol).")
    raise typer.Exit(1)


@app.command("robot-estop")
def robot_estop_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    port: str = typer.Option("auto"),
):
    """EMERGENCY STOP: disable torque if possible, close the port, cut power by hand."""
    from terafold.robot.real_motion import print_estop_instructions

    _cfg, adapter = _load_arm(robot, port)
    res = adapter.emergency_stop()
    _echo(f"torque_off_attempted={res['torque_off_attempted']} "
          f"torque_off_ok={res['torque_off_ok']} port_closed={res['port_closed']}")
    if not res["protocol_confirmed"]:
        _echo("[warn] Protocol unconfirmed: torque-off bytes were NOT sent.")
    _err("CUT POWER MANUALLY NOW:")
    for line in res["manual_instructions"]:
        _echo(f"  - {line}")
    print_estop_instructions(_echo)


@app.command("real-image-fold")
def real_image_fold_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    image: str = typer.Option(..., help="Photo of the towel."),
    calibration: Optional[str] = typer.Option(None, help="Homography calibration JSON."),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Image/table-based REAL motion — refuses without full table calibration."""
    from terafold.robot.real_motion import image_real_motion_gate

    gate = image_real_motion_gate(
        homography_calibration=calibration, enable_motion=enable_motion, acknowledge=acknowledge,
    )
    if not gate["allowed"]:
        _err(gate["message"])
        _echo("Missing:")
        for m in gate["missing"]:
            _echo(f"  - {m}")
        raise typer.Exit(1)
    _err("(unreachable: image-based real motion still requires a confirmed protocol).")
    raise typer.Exit(1)


@app.command("map-servo-joints")
def map_servo_joints_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    port: str = typer.Option(..., help="Serial port (e.g. /dev/cu.usbmodem...)."),
    ids: str = typer.Option(..., help="Comma-separated servo IDs, e.g. 1,2,5,6."),
    delta_units: int = typer.Option(40, help="Tiny nudge in servo units."),
    speed: str = typer.Option("very_slow"),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Nudge one servo at a time, ask which joint moved, and save a joint map."""
    from terafold.robot.arm_config import load_arm_config
    from terafold.robot.real_image_ghost import run_map_servo_joints
    from terafold.robot.real_motion import motion_logger
    from terafold.robot.safety import SafetyError, require_motion_enabled
    from terafold.robot.waveshare_sms_sts_backend import WaveshareSmsStsBackend

    id_list = [int(x) for x in ids.split(",") if x.strip()]
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as e:
        _err(str(e))
        raise typer.Exit(1)
    cfg = load_arm_config(robot)
    logger = motion_logger("map_servo_joints")
    backend = WaveshareSmsStsBackend(
        port=port, baudrate=cfg.baud, active_ids=id_list,
        default_speed=cfg.default_speed_units, default_acc=cfg.default_acc_units,
        safe_min_units=cfg.safe_position_units[0], safe_max_units=cfg.safe_position_units[1],
        logger=logger,
    )
    backend.confirm()
    res = run_map_servo_joints(backend, robot=robot, ids=id_list, delta_units=delta_units,
                               speed=speed, log=_echo)
    backend.close()
    if not res["ok"]:
        _err(res["refusal"])
        raise typer.Exit(1)
    _ok(f"Saved joint map -> {res['path']}")
    _echo("Now `real-image-ghost-fold` can build a joint-space ghost (above the table).")


@app.command("real-image-ghost-fold")
def real_image_ghost_fold_cmd(
    image: str = typer.Option(..., help="Photo of the towel."),
    robot: str = typer.Option(..., help="Robot config name."),
    port: str = typer.Option(None, help="Serial port (defaults to the config's)."),
    height_clearance_m: float = typer.Option(0.10, help="Min height above the table (>=0.10)."),
    speed: str = typer.Option("very_slow"),
    calibration: Optional[str] = typer.Option(None, help="Homography calibration JSON (for table-space)."),
    task: str = typer.Option("configs/task_fold_towel_half.yaml"),
    mode: str = typer.Option("model", help="Perception: model (classical fallback) | markers | claude."),
    perception_backend: str = typer.Option(
        "classical", help="classical | yolo_towel_pose (trained towel corner-pose, dry-run only)."),
    weights: Optional[str] = typer.Option(None, help="YOLO towel-pose weights (perception-backend yolo_towel_pose)."),
    plan_critic: Optional[str] = typer.Option(None, help="Plan-critic model (optional; rule-based by default)."),
    min_perception_confidence: float = typer.Option(0.75, help="Refuse below this avg keypoint confidence."),
    min_plan_score: float = typer.Option(0.65, help="Refuse below this critic plan score."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Image -> plan -> GHOST fold above the table on the real arm. Default dry-run.

    With ``--perception-backend yolo_towel_pose`` the trained towel corner detector +
    rule-based plan critic run in DRY-RUN: corners, confidence, towel state, fold
    direction, grasp/place, plan score, risk reasons, and the allowed_for_* flags are
    printed; low-confidence / bad-view / multiple-towel / not-towel / already-folded
    cases are refused. Contact folding stays LOCKED and no hardware command is sent.
    """
    from terafold.robot.real_image_ghost import run_real_image_ghost_fold

    res = run_real_image_ghost_fold(
        image, robot=robot, port=port, height_clearance_m=height_clearance_m, speed=speed,
        calibration=calibration, task_path=task, mode=mode,
        perception_backend=perception_backend, weights=weights, plan_critic=plan_critic,
        min_perception_confidence=min_perception_confidence, min_plan_score=min_plan_score,
        enable_motion=enable_motion, acknowledge=acknowledge, log=_echo,
    )
    _echo(f"  log -> {res.get('log')}")
    if perception_backend == "yolo_towel_pose":
        _echo(f"  allowed_for_dry_run={res.get('allowed_for_dry_run')}  "
              f"allowed_for_real_ghost={res.get('allowed_for_real_ghost')}  "
              f"allowed_for_contact={res.get('allowed_for_contact')}")
    if res["status"] == "refused":
        _err(res.get("refusal", "refused"))
        raise typer.Exit(1)
    if res["status"] == "moved":
        _ok(f"Ghost sweep complete ({res['motor_commands_sent']} commands). Arm stayed above the table.")
    elif res["status"] == "dry_run":
        _ok("DRY-RUN complete (nothing moved).")


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
    level: str = typer.Option(
        "ee-only",
        help="ee-only | arm-ik | cloth-proxy | so101-real | cloth-physics-proxy | mujoco-cloth."),
    view: str = typer.Option(
        "iso",
        help="Camera: iso|top|side|follow-ee|gripper|demo|contact|cloth|split."),
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
    if res.get("mujoco_cloth"):
        mc = res["mujoco_cloth"]
        _echo(f"  MuJoCo cloth  : attempted={mc['attempted']} flex_available={mc['flex_available']} "
              f"-> using {mc['fell_back_to']}")
    if res.get("so101_assets"):
        a = res["so101_assets"]
        which = (f"official {a['type']} @ {a['path']}" if a["found"]
                 else "approximation (no official assets found)")
        _echo(f"  SO-101 assets : {which}")
    if "fold_visually_successful" in res:
        tag = "SUCCESS" if res["fold_visually_successful"] else "INCOMPLETE"
        _echo(f"  fold result   : {tag}  (edge error {res.get('final_edge_error_m')} m, "
              f"crossed_crease={res.get('crossed_crease')})")
    for w in (res.get("warnings") or []):
        _echo(f"  {w}")
    if res.get("all_views"):
        for v, p in res["views"].items():
            _echo(f"  {v:10s}: {p}")
    else:
        _echo(f"  video : {res.get('video') or res.get('out')}")
    for v, d in (res.get("frames_dirs") or {}).items():
        _echo(f"  frames[{v}]: {d}")
    _echo(f"  meta  : {res.get('meta_json')}")
    _echo(f"  {res.get('note')}")


# --------------------------------------------------------------------------
# Kinematics model status
# --------------------------------------------------------------------------


@app.command("robot-model-status")
def robot_model_status_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    save_json: Optional[str] = typer.Option(None, help="Also write the JSON status here."),
):
    """Kinematic-model status: joint map, raw→angle, FK/IK, contact posture.

    For the custom arm this reports the model as UNVALIDATED — autonomous Cartesian
    IK and contact stay refused until a validated model exists.
    """
    from terafold.kinematics.model_status import report_markdown, robot_model_status

    status = robot_model_status(robot)
    _echo(report_markdown(status))
    if save_json:
        from terafold.data.episode_schema import write_json

        write_json(save_json, status)
        _ok(f"Wrote status -> {save_json}")


# --------------------------------------------------------------------------
# Servo characterization (deadband / backlash / repeatability)
# --------------------------------------------------------------------------


@app.command("characterize-servos")
def characterize_servos_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    ids: str = typer.Option(..., help="Comma/range IDs, e.g. 1,2,5,6 or 1-6."),
    out: str = typer.Option("runs/servo_characterization/session_001"),
    port: str = typer.Option("auto"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Characterize servos (readback noise/deadband/backlash/step/repeatability).

    Default DRY-RUN prints the plan and moves nothing. Real mode needs BOTH motion
    flags AND a confirmed protocol (refuses otherwise).
    """
    from terafold.robot.characterization import run_characterization
    from terafold.robot.servo_scan import parse_id_range

    id_list = parse_id_range(ids)
    res = run_characterization(
        robot, ids=id_list, out=out, port=None if port == "auto" else port,
        dry_run=dry_run, enable_motion=enable_motion, acknowledge=acknowledge, log=_echo,
    )
    status = res.get("status")
    if status == "dry_run":
        _ok(f"DRY-RUN: nothing moved. Plan for IDs {id_list} written under {res.get('out')}.")
    elif status == "refused":
        _err(res.get("reason", res.get("refusal", "refused")))
        raise typer.Exit(1)
    elif status == "ok":
        _ok(f"Characterized {id_list}. metrics -> {res.get('metrics_json')}")
        if res.get("summary_md"):
            _echo(f"  summary : {res['summary_md']}")
    else:
        _echo(str(res))


# --------------------------------------------------------------------------
# Calibration: table homography, validation, camera intrinsics, robot↔table
# --------------------------------------------------------------------------


def _xy_pairs(spec: str):
    from terafold.camera.table_calibration import parse_xy_list

    return parse_xy_list(spec)


def _default_cal_out(robot: Optional[str], name: str, override: Optional[str]) -> str:
    if override:
        return override
    if robot:
        from terafold.robot.capabilities import artifact_paths

        return artifact_paths(robot).get(name, f"runs/calibration/{name}.yaml")
    return f"runs/calibration/{name}.yaml"


@app.command("calibrate-table")
def calibrate_table_cmd(
    points_image: str = typer.Option(..., help="4+ pixel pts 'u1,v1;u2,v2;...'."),
    points_table: str = typer.Option(..., help="4+ table metres 'x1,y1;x2,y2;...'."),
    robot: Optional[str] = typer.Option(None, help="Robot (defaults --out to its discoverable path)."),
    image: Optional[str] = typer.Option(None, help="Calibration image (reference only)."),
    out: Optional[str] = typer.Option(None, help="Output YAML (default runs/calibration/<robot>_table_homography.yaml)."),
    operator: str = typer.Option(""),
):
    """Solve + save the image→table homography (pure numpy) with validity flags."""
    import time as _time

    from terafold.calibration.table_homography import calibrate_table

    dest = _default_cal_out(robot, "table_homography", out)
    try:
        res = calibrate_table(_xy_pairs(points_image), _xy_pairs(points_table), dest,
                              image=image, operator=operator, timestamp=_time.time())
    except Exception as exc:
        _err(f"Calibration failed: {exc}")
        raise typer.Exit(1)
    _ok(f"Saved homography -> {res['out']}  (rms {res['rms_error_m']:.4f} m, "
        f"max {res['max_error_m']:.4f} m)")
    _echo(f"  valid_for_hover={res['valid_for_hover']}  valid_for_contact={res['valid_for_contact']}")


@app.command("validate-table-calibration")
def validate_table_calibration_cmd(
    calibration: str = typer.Option(..., help="Homography YAML from calibrate-table."),
    points_image: str = typer.Option(..., help="Held-out pixel pts 'u,v;...'."),
    points_table: str = typer.Option(..., help="Held-out table metres 'x,y;...'."),
):
    """Validate a saved homography against held-out correspondences."""
    from terafold.calibration.validation import validate_table_calibration

    res = validate_table_calibration(calibration, _xy_pairs(points_image), _xy_pairs(points_table))
    _echo(f"held-out rms {res['rms_error_m']:.4f} m  max {res['max_error_m']:.4f} m")
    _echo(f"valid_for_hover={res['valid_for_hover']}  valid_for_contact={res['valid_for_contact']}")
    if not res["valid_for_hover"]:
        _err("Calibration does NOT meet the hover gate — hover/contact stay locked.")
        raise typer.Exit(1)
    _ok("Calibration valid for hover.")


@app.command("calibrate-camera")
def calibrate_camera_cmd(
    images: str = typer.Option(..., help="Glob of checkerboard images, e.g. 'data/cal/*.png'."),
    robot: Optional[str] = typer.Option(None, help="Robot (defaults --out path)."),
    out: Optional[str] = typer.Option(None, help="Output YAML."),
    board: str = typer.Option("9,6", help="Inner corners 'cols,rows'."),
    square_m: float = typer.Option(0.025, help="Square size (m)."),
):
    """Estimate camera intrinsics from checkerboard images (OpenCV optional)."""
    from terafold.calibration.camera_intrinsics import calibrate_camera

    dest = _default_cal_out(robot, "camera_intrinsics", out)
    cols, rows = (int(x) for x in board.split(","))
    res = calibrate_camera(images, dest, board=(cols, rows), square_m=square_m)
    if res.get("status") != "ok":
        _err(res.get("reason", "camera calibration unavailable"))
        if res.get("install"):
            _err(f"Run:  {res['install']}")
        raise typer.Exit(1)
    _ok(f"Saved intrinsics -> {res['out']}  (rms {res['rms_px']:.3f} px, "
        f"valid_for_hover={res['valid_for_hover']})")


@app.command("robot-touch-calibration")
def robot_touch_calibration_cmd(
    robot: str = typer.Option(..., help="Robot config name."),
    joint_map: Optional[str] = typer.Option(None, help="Joint map YAML."),
    out: str = typer.Option("runs/calibration/robot_table_touch_points.json"),
    points_json: Optional[str] = typer.Option(None, help="A JSON of recorded {table_xy,robot_xyz} pairs to save."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
    enable_motion: bool = typer.Option(False, "--enable-motion"),
    acknowledge: bool = typer.Option(False, "--i-understand-this-moves-hardware"),
):
    """Plan/record manual table↔robot touch points (dry-run default; never auto-drives)."""
    from terafold.calibration.robot_table_transform import robot_touch_calibration
    from terafold.data.episode_schema import read_json

    points = None
    if points_json:
        d = read_json(points_json)
        points = d.get("points", d) if isinstance(d, dict) else d
    res = robot_touch_calibration(robot, joint_map=joint_map, out=out, dry_run=dry_run,
                                  enable_motion=enable_motion, acknowledge=acknowledge, points=points)
    status = res.get("status")
    if status == "saved":
        _ok(f"Saved {len(points or [])} touch points -> {res['out']}")
    elif status == "refused":
        _err(res.get("reason", "refused"))
        raise typer.Exit(1)
    else:
        _echo(f"[{status}] Manual touch-calibration plan (moves_hardware={res.get('moves_hardware')}):")
        for step in (res.get("procedure") or res.get("plan") or []):
            _echo(f"  - {step}")
        _echo(f"Output format: {res.get('output_format', 'JSON {points:[{table_xy,robot_xyz}]}')}")
        _echo("Record pairs then re-run with --points-json to save, or "
              "`fit-robot-table-transform` once you have >=3 points.")


@app.command("fit-robot-table-transform")
def fit_robot_table_transform_cmd(
    touch_points: str = typer.Option(..., help="Touch-points JSON (>=3 {table_xy,robot_xyz})."),
    robot: Optional[str] = typer.Option(None, help="Robot (defaults --out path)."),
    out: Optional[str] = typer.Option(None, help="Output YAML."),
):
    """Fit + save the robot↔table rigid transform (Kabsch/Umeyama) with validity flags."""
    import time as _time

    from terafold.calibration.robot_table_transform import fit_robot_table_transform

    dest = _default_cal_out(robot, "robot_table_transform", out)
    try:
        res = fit_robot_table_transform(touch_points, dest, timestamp=_time.time())
    except Exception as exc:
        _err(f"Fit failed: {exc}")
        raise typer.Exit(1)
    _ok(f"Saved robot↔table transform -> {res['out']}  (rms {res['rms_error_m']:.4f} m)")
    _echo(f"  valid_for_hover={res['valid_for_hover']}  valid_for_contact={res['valid_for_contact']}")


# --------------------------------------------------------------------------
# Data: episode summary + LeRobot export from motion logs
# --------------------------------------------------------------------------


@app.command("episode-summary")
def episode_summary_cmd(
    episode: str = typer.Option(..., help="A motion-log JSONL, episode dir, or episode JSON."),
):
    """Summarize an episode / motion-log (frames, task, robot, dry-run, writes)."""
    from terafold.data.summary import summary_markdown

    _echo(summary_markdown(episode))


@app.command("export-lerobot-logs")
def export_lerobot_logs_cmd(
    logs: str = typer.Option(..., help="Dir of motion-log JSONL files and/or episode dirs."),
    out: str = typer.Option(..., help="LeRobot dataset output dir."),
    fps: int = typer.Option(10),
    robot_type: str = typer.Option("custom_7dof_sms_sts"),
):
    """Export motion-log episodes to a LeRobot-compatible scaffold (no LeRobot needed)."""
    from terafold.data.lerobot_export import export_episodes_to_lerobot

    res = export_episodes_to_lerobot(logs, out, fps=fps, robot_type=robot_type)
    _ok(f"Exported {res.get('episodes')} episodes / {res.get('frames')} frames -> {res.get('out')} "
        f"({res.get('format')}, lerobot_installed={res.get('lerobot_installed')})")
    if res.get("note"):
        _echo(f"  note: {res['note']}")


# --------------------------------------------------------------------------
# Evaluation report
# --------------------------------------------------------------------------


@app.command("eval-run")
def eval_run_cmd(
    logs: str = typer.Option(..., help="A motion-log JSONL (or episode)."),
    out: Optional[str] = typer.Option(None, help="Write the Markdown report here."),
):
    """Compute evaluation metrics from logs and write a Markdown report."""
    from terafold.eval.report import eval_run

    res = eval_run(logs, out=out)
    if res.get("report_markdown"):
        _echo(res["report_markdown"])
    for w in res.get("warnings", []):
        _err(f"warning: {w}")
    if out:
        _ok(f"Wrote eval report -> {out}")


# --------------------------------------------------------------------------
# Sim: image-driven above-table ghost-fold preview (simulation only)
# --------------------------------------------------------------------------


@app.command("sim-real-ghost-fold")
def sim_real_ghost_fold_cmd(
    plan_json: str = typer.Option(..., help="A FoldPlan / demo result JSON."),
    joint_map: Optional[str] = typer.Option(None, help="Robot name or joint-map YAML (for active IDs)."),
    out: str = typer.Option("runs/sim/custom_arm_ghost_preview.png", help="Output image (PNG; metadata JSON beside it)."),
    height_clearance_m: float = typer.Option(0.10),
):
    """Visualize the image-derived above-table ghost fold. SIMULATION ONLY (no hardware)."""
    from terafold.sim.ghost_preview import run_sim_real_ghost_fold

    res = run_sim_real_ghost_fold(plan_json, joint_map=joint_map, out=out,
                                  height_clearance_m=height_clearance_m, on_log=_echo)
    _ok(f"sim-real-ghost-fold preview ({res['status']}): contact_disabled={res['contact_disabled']}, "
        f"touches_table={res['touches_table']}, rendered={res['rendered']}")
    _echo(f"  waypoints={res['num_waypoints']}  min_z={res['min_z']}m  direction={res.get('fold_direction')}")
    _echo(f"  meta : {res['meta_json']}")
    for w in res.get("warnings", []):
        _echo(f"  ⚠ {w}")


# --------------------------------------------------------------------------
# Towel real-image-first dataset + YOLO pose + plan critic pipeline
# --------------------------------------------------------------------------


@app.command("create-towel-real-dataset")
def create_towel_real_dataset_cmd(
    images: str = typer.Option(..., help="Folder of real towel images."),
    out: str = typer.Option(..., help="Output dataset directory."),
    source: str = typer.Option("user_photo", help="user_photo | external | synthetic."),
):
    """Build a REAL-image towel dataset (default path): copy + normalize + label templates."""
    from terafold.towel.dataset import create_towel_real_dataset

    res = create_towel_real_dataset(images, out, source=source, log=_echo)
    _ok(f"Created dataset: {res['num_images']} images -> {out} "
        f"({res['skipped_duplicates']} duplicates skipped)")
    _echo(f"  manifest: {res['manifest']}")
    _echo(f"  next: terafold label-towel-folder --dataset {out}")


@app.command("label-towel-image")
def label_towel_image_cmd(
    image: str = typer.Option(..., help="Image to label."),
    label: str = typer.Option(..., help="Output label JSON path."),
    state: Optional[str] = typer.Option(None, help="Towel state (default keeps/flat_unfolded)."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Force terminal coordinate input."),
):
    """Label one towel image: click/enter 4 corners (tl, tr, br, bl) + state."""
    from terafold.towel.labeling import label_towel_image

    res = label_towel_image(image, label, state=state, use_gui=not no_gui, log=_echo)
    _ok(f"Labeled -> {res['label_path']}  corners={res['corners']}")
    if res.get("overlay"):
        _echo(f"  overlay: {res['overlay']}")


@app.command("label-towel-folder")
def label_towel_folder_cmd(
    dataset: str = typer.Option(..., help="Towel dataset directory."),
):
    """Loop unlabeled images: label corners, or mark bad_view/multiple_towels/not_towel/skip."""
    from terafold.towel.labeling import label_towel_folder

    res = label_towel_folder(dataset, log=_echo)
    _ok(f"Done: {res['labeled']} labeled, {res['skipped']} skipped, "
        f"{res['marked_bad']} marked bad (of {res['total']}).")


@app.command("import-towel-web-dataset")
def import_towel_web_dataset_cmd(
    source: str = typer.Option(..., help="folder | roboflow | kaggle | huggingface | open_images."),
    out: str = typer.Option(..., help="Output dataset directory."),
    query: str = typer.Option("towel"),
    max_images: int = typer.Option(500),
    src_dir: Optional[str] = typer.Option(None, help="Local export folder (folder/roboflow/kaggle)."),
    hf_repo: Optional[str] = typer.Option(None, help="HuggingFace dataset repo id (source huggingface)."),
    license_note: str = typer.Option("", help="License/provenance note recorded in the manifest."),
):
    """Import external real towel images LEGALLY (no scraping). Manual instructions if needed."""
    from terafold.towel.web_import import import_towel_web_dataset

    res = import_towel_web_dataset(source, out, query=query, max_images=max_images,
                                   src_dir=src_dir, hf_repo=hf_repo, license_note=license_note,
                                   log=_echo)
    status = res.get("status")
    if status == "ok":
        _ok(f"Imported {res['num_images']} external images -> {res['out']} (needs labeling).")
    elif status == "manual_instructions":
        _err(f"Manual steps required for source '{source}':")
        for s in res.get("instructions", []):
            _echo(f"  - {s}")
        if res.get("install"):
            _echo(f"  install: {res['install']}")
        raise typer.Exit(2)
    else:
        _err(res.get("message", "import failed"))
        if res.get("supported"):
            _echo(f"  supported sources: {res['supported']}")
        raise typer.Exit(1)


@app.command("merge-towel-datasets")
def merge_towel_datasets_cmd(
    inputs: str = typer.Option(..., help="Comma-separated dataset dirs."),
    out: str = typer.Option(..., help="Merged output dataset dir."),
    real_only: bool = typer.Option(False, "--real-only", help="Drop synthetic samples."),
):
    """Merge usable labeled samples (dedup by hash) into a train/val/test split."""
    from terafold.towel.merge import merge_towel_datasets

    dirs = [d for d in inputs.split(",") if d.strip()]
    res = merge_towel_datasets(dirs, out, real_only=real_only, log=_echo)
    _ok(f"Merged {res['num_images']} usable samples -> {out}")
    _echo(f"  split train/val/test = {res['by_split']['train']}/{res['by_split']['val']}/"
          f"{res['by_split']['test']}   by source: {res['by_source']}")


@app.command("generate-towel-dataset")
def generate_towel_dataset_cmd(
    num: int = typer.Option(1000, help="Number of synthetic images."),
    out: str = typer.Option(..., help="Output dataset dir."),
    image_size: int = typer.Option(256),
    seed: int = typer.Option(0),
):
    """OPTIONAL synthetic towel augmentation (NOT the main path — real images are default)."""
    from terafold.towel.synthetic import generate_towel_dataset

    res = generate_towel_dataset(num, out, image_size=image_size, seed=seed, log=_echo)
    _ok(f"Generated {res['num_images']} SYNTHETIC (augmentation-only) images -> {res['out']}")


@app.command("export-yolo-towel-pose")
def export_yolo_towel_pose_cmd(
    data: str = typer.Option(..., help="A (merged) towel dataset dir."),
    out: str = typer.Option(..., help="YOLO pose dataset output dir."),
    include_pseudolabels: str = typer.Option(
        "none", "--include-pseudolabels",
        help="none (default, exclude Claude pseudo-labels) | approved_only."),
    include_critic_negatives: bool = typer.Option(
        False, "--include-critic-negatives",
        help="Include training_track=critic_negative samples (excluded by default)."),
):
    """Export labeled samples to YOLO pose (1 class towel, 4 corners tl/tr/br/bl)."""
    from terafold.towel.yolo_export import (INCLUDE_PSEUDOLABEL_MODES,
                                            export_yolo_towel_pose)

    if include_pseudolabels not in INCLUDE_PSEUDOLABEL_MODES:
        _err(f"--include-pseudolabels must be one of {list(INCLUDE_PSEUDOLABEL_MODES)}")
        raise typer.Exit(1)
    res = export_yolo_towel_pose(data, out, include_pseudolabels=include_pseudolabels,
                                 include_critic_negatives=include_critic_negatives, log=_echo)
    _ok(f"Exported {res['exported']} samples -> {out}  (rejected {res['rejected']})")
    bysrc = res["by_source"]
    _echo("  by source: " + " ".join(f"{g}={n}" for g, n in bysrc.items() if n))
    _echo(f"  by training_track: {res['by_training_track']}")
    excl = (res.get("skipped_pseudolabels", 0) or res.get("skipped_unapproved_pseudo", 0)
            or res.get("skipped_critic_negatives", 0))
    if excl:
        _echo(f"  excluded: {res['skipped_pseudolabels']} pseudo (policy='{include_pseudolabels}'), "
              f"{res['skipped_unapproved_pseudo']} unapproved, "
              f"{res['skipped_critic_negatives']} critic-negative")
    _echo(f"  yaml: {res['yaml']}  (kpt_shape={res['kpt_shape']})")


@app.command("print-towel-training-command")
def print_towel_training_command_cmd(
    data: str = typer.Option(..., help="YOLO pose dataset dir (has towel_pose.yaml)."),
    model: str = typer.Option("yolo26n-pose.pt", help="Base model (fallback yolo11n-pose.pt)."),
    epochs: int = typer.Option(100),
    imgsz: int = typer.Option(640),
    batch: int = typer.Option(16),
):
    """Print a copy-paste YOLO pose training command (local or RunPod)."""
    from terafold.towel.yolo_runtime import print_towel_training_command

    res = print_towel_training_command(data, model=model, epochs=epochs, imgsz=imgsz, batch=batch)
    _echo(res["command"])
    _echo("")
    _echo(f"# fallback (older model name): {res['fallback_command']}")
    if res.get("runpod_note"):
        _echo(f"# RunPod: {res['runpod_note']}")


@app.command("infer-towel-pose")
def infer_towel_pose_cmd(
    image: str = typer.Option(..., help="Image to run towel corner-pose detection on."),
    weights: str = typer.Option(..., help="Trained YOLO pose weights (best.pt)."),
    out: Optional[str] = typer.Option(None, help="JSON output path."),
    overlay_out: Optional[str] = typer.Option(None, help="Overlay image output path."),
    conf: float = typer.Option(0.25),
):
    """Run the trained towel corner-pose detector on one image (Ultralytics)."""
    from terafold.towel.yolo_runtime import infer_towel_pose

    res = infer_towel_pose(image, weights, out=out, overlay_out=overlay_out, conf=conf, log=_echo)
    status = res.get("status")
    if status == "unavailable":
        _err(res.get("message", "Ultralytics not installed."))
        _err(f"Run:  {res.get('install')}")
        raise typer.Exit(1)
    if status == "no_detection":
        _err("No towel detected.")
        raise typer.Exit(2)
    _ok(f"corners={res['corners']}  avg_conf={res.get('avg_keypoint_confidence'):.2f}")
    if res.get("out"):
        _echo(f"  json   : {res['out']}")
    if res.get("overlay"):
        _echo(f"  overlay: {res['overlay']}")


@app.command("evaluate-towel-pose")
def evaluate_towel_pose_cmd(
    data: str = typer.Option(..., help="Towel dataset dir (labeled)."),
    weights: str = typer.Option(..., help="Trained YOLO pose weights."),
    out: str = typer.Option(..., help="Eval output dir / Markdown path."),
):
    """Evaluate a trained towel pose model: corner error, IoU, error by source/state."""
    from terafold.towel.yolo_runtime import evaluate_towel_pose

    res = evaluate_towel_pose(data, weights, out, log=_echo)
    if res.get("status") == "unavailable":
        _err(res.get("message", "Ultralytics not installed."))
        _err(f"Run:  {res.get('install')}")
        raise typer.Exit(1)
    _ok(f"Evaluated. report -> {res.get('report')}")


@app.command("build-plan-critic-dataset")
def build_plan_critic_dataset_cmd(
    data: str = typer.Option(..., help="A towel dataset dir."),
    out: str = typer.Option(..., help="Plan-critic dataset output dir."),
):
    """Derive a fold-plan-critic feature table from a towel dataset."""
    from terafold.towel.plan_critic import build_plan_critic_dataset

    res = build_plan_critic_dataset(data, out, log=_echo)
    _ok(f"Built critic dataset: {res['rows']} rows "
        f"({res['positives']} pos / {res['negatives']} neg) -> {res['out']}")


@app.command("train-plan-critic")
def train_plan_critic_cmd(
    data: str = typer.Option(..., help="Plan-critic dataset dir (critic_dataset.jsonl)."),
    out: str = typer.Option(..., help="Output run dir (model.pt)."),
):
    """Train the fold-plan critic (scaffold: torch if present, else numpy logistic)."""
    from terafold.towel.plan_critic import train_plan_critic

    res = train_plan_critic(data, out, log=_echo)
    _ok(f"Trained plan critic ({res['backend']}): {res['model']}  train_acc={res.get('train_acc')}")


@app.command("eval-plan-critic")
def eval_plan_critic_cmd(
    data: str = typer.Option(..., help="Plan-critic dataset dir."),
    model: str = typer.Option(..., help="Trained model.pt."),
    out: str = typer.Option(..., help="Markdown report output path."),
):
    """Evaluate the fold-plan critic and write a Markdown report."""
    from terafold.towel.plan_critic import eval_plan_critic

    res = eval_plan_critic(data, model, out, log=_echo)
    _ok(f"Plan critic eval: accuracy={res.get('accuracy')}  report -> {res.get('report', out)}")


# --------------------------------------------------------------------------
# External real-image ingest + Claude pseudo-labeling pipeline
# --------------------------------------------------------------------------


def _print_lines(lines, prefix="  - "):
    for ln in lines or []:
        _echo(f"{prefix}{ln}" if ln else "")


@app.command("import-kaggle-towel-dataset")
def import_kaggle_towel_dataset_cmd(
    dataset: str = typer.Option(..., help="Kaggle dataset slug, e.g. owner/dataset-slug."),
    out: str = typer.Option(..., help="Output RAW image dataset directory."),
    max_images: int = typer.Option(2000, help="Max images to ingest."),
    min_size: int = typer.Option(64, help="Skip images whose smaller side is below this (px)."),
    src_dir: Optional[str] = typer.Option(None, help="Use an already-downloaded export (offline)."),
    license_note: str = typer.Option("", help="License/provenance note for the manifest."),
):
    """Ingest a Kaggle towel dataset into a RAW image dataset (no labels yet)."""
    from terafold.towel.raw_ingest import import_kaggle_towel_dataset

    res = import_kaggle_towel_dataset(dataset, out, max_images=max_images, min_size=min_size,
                                      src_dir=src_dir, license_note=license_note, log=_echo)
    status = res.get("status")
    if status == "ok":
        _ok(f"Ingested {res['num_images']} raw Kaggle images -> {out} (needs filtering/labeling).")
        _echo(f"  manifest: {res['manifest']}")
        _echo(f"  next: terafold filter-towel-images --input {out} --out data/towel_candidates_v0")
        return
    if status == "unavailable":
        _err("Kaggle client/credentials not available.")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    _err(res.get("message", "Kaggle import failed."))
    _print_lines(res.get("instructions"))
    raise typer.Exit(1)


@app.command("import-openimages-towels")
def import_openimages_towels_cmd(
    out: str = typer.Option(..., help="Output RAW image dataset directory."),
    max_images: int = typer.Option(2000, help="Max images to ingest."),
    min_size: int = typer.Option(64, help="Skip images whose smaller side is below this (px)."),
    src_dir: Optional[str] = typer.Option(None, help="Locally-downloaded Open Images Towel folder."),
    license_note: str = typer.Option("", help="License/provenance note for the manifest."),
):
    """Ingest Open Images 'Towel' images (manual download path) into a RAW dataset."""
    from terafold.towel.raw_ingest import import_openimages_towels

    res = import_openimages_towels(out, max_images=max_images, min_size=min_size,
                                   src_dir=src_dir, license_note=license_note, log=_echo)
    status = res.get("status")
    if status == "ok":
        _ok(f"Ingested {res['num_images']} raw Open Images towels -> {out}.")
        _echo(f"  manifest: {res['manifest']}")
        _echo(f"  next: terafold filter-towel-images --input {out} --out data/towel_candidates_v0")
        return
    if status == "manual_instructions":
        _err("Manual Open Images download required (boxes are not towel corners):")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    _err(res.get("message", "Open Images import failed."))
    raise typer.Exit(1)


@app.command("filter-towel-images")
def filter_towel_images_cmd(
    input: str = typer.Option(..., help="Raw/candidate manifest, a create-towel-real-dataset "
                              "dataset, an images/ dir, or a plain image folder."),
    out: str = typer.Option(..., help="Output candidate dataset dir."),
    mode: str = typer.Option("filename_or_vlm", help="filename | filename_or_vlm | all."),
    max_images: int = typer.Option(1000, help="Max candidates to keep."),
    min_size: int = typer.Option(0, help="Small-image cut (0 = default; 'all' uses a 16px floor)."),
    trust_source: bool = typer.Option(False, "--trust-source",
                                      help="Trust the source: keep all valid images (alias for --mode all)."),
):
    """Keep likely-towel images by keyword/VLM, or keep ALL valid images (--mode all / --trust-source)."""
    from terafold.towel.filtering import filter_towel_images

    if trust_source:
        mode = "all"
    res = filter_towel_images(input, out, mode=mode, max_images=max_images,
                              min_size=min_size, log=_echo)
    if res.get("status") != "ok":
        _err(res.get("message", "filter failed"))
        if res.get("modes"):
            _echo(f"  modes: {res['modes']}")
        raise typer.Exit(1)
    _ok(f"Kept {res['kept']} candidates (of {res['num_input']}) -> {out}  "
        f"[{res['rejected']} rejected; mode={res['mode']}]")
    _echo(f"  reasons: {res['reasons']}")
    _echo(f"  next: terafold claude-label-towel-folder --input {out} "
          "--out data/towel_pseudolabeled_v0")


@app.command("crop-real-towel-candidates")
def crop_real_towel_candidates_cmd(
    input: str = typer.Option(..., help="Raw/candidate manifest, a create-towel-real-dataset "
                              "dataset, an images/ dir, or a plain image folder of FULL photos."),
    out: str = typer.Option(..., help="Output crop dataset dir (tight crops + remap metadata)."),
    target_description: str = typer.Option(
        "beige and white striped towel",
        help="What Claude must localize (and ignore the bed/pillow/blanket/headboard/floor/body)."),
    max_images: int = typer.Option(500, help="Max images to send to Claude."),
    model: str = typer.Option("claude-opus-4-8", help="Anthropic model id."),
    min_confidence: float = typer.Option(0.5, help="Reject a localization below this confidence."),
    pad_frac: float = typer.Option(0.08, help="Padding around the detected box (fraction of box size)."),
    max_crop_size: int = typer.Option(1024, help="Downscale crops so the longest side <= this (px)."),
    resume: bool = typer.Option(False, "--resume", help="Skip images already cropped in --out."),
    force: bool = typer.Option(False, "--force", "--recrop",
                               help="Re-localize/crop every image, overwriting old crops."),
):
    """Crop-first: ask Claude for ONLY the target towel's box, then save a tight crop.

    Fixes the "Claude labeled the bed/headboard instead of the towel" problem by
    localizing the striped towel first; corners labeled on the crops remap back to the
    original full images via each crop's metadata.
    """
    from terafold.towel.crop_candidates import crop_real_towel_candidates

    res = crop_real_towel_candidates(
        input, out, target_description=target_description, max_images=max_images, model=model,
        min_confidence=min_confidence, pad_frac=pad_frac, max_crop_size=max_crop_size,
        resume=resume, force=force, log=_echo)
    status = res.get("status")
    if status == "no_api_key":
        _err("Anthropic API key not configured.")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    if status != "ok":
        _err(res.get("message", "Towel cropping failed."))
        raise typer.Exit(1)
    _ok(f"Cropped {res['num_crops']} / {res['num_input']} towel candidates -> {out}  "
        f"({res['num_rejected']} rejected, {res['errors']} errors)")
    usage = res.get("usage", {})
    _echo(f"  tokens in/out: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}  "
          f"est cost ~${res.get('cost_estimate_usd', 0)} (approx — verify Anthropic pricing)")
    _echo(f"  next: terafold claude-label-towel-folder --input {out} "
          "--label-policy outer_visible_corners_striped_towel_only --force "
          "--out data/towel_real_labeled_v1_crops --min-confidence 0.65")


@app.command("claude-label-towel-folder")
def claude_label_towel_folder_cmd(
    input: str = typer.Option(..., help="Raw/candidate manifest, a create-towel-real-dataset "
                              "dataset, a crop dataset (crop-real-towel-candidates), an images/ "
                              "dir, or a plain image folder."),
    out: str = typer.Option(..., help="Output pseudo-labeled towel dataset dir."),
    max_images: int = typer.Option(500, help="Max images to send to Claude."),
    model: str = typer.Option("claude-opus-4-8", help="Anthropic model id."),
    min_confidence: float = typer.Option(0.75, help="Below this -> review_needed."),
    label_policy: str = typer.Option(
        "default", help="default | outer_visible_corners_any_state "
        "(pixel-accurate visible outer corners, accepts folded states) | "
        "outer_visible_corners_striped_towel_only (crop-first striped-towel only; "
        "remaps corners back to full images)."),
    resume: bool = typer.Option(False, "--resume", help="Skip images already labeled in --out."),
    force: bool = typer.Option(False, "--force", "--relabel",
                               help="Re-label every image, overwriting old (loose) labels."),
):
    """Pseudo-label towel images with Claude Vision (geometry only; human review required)."""
    from terafold.towel.claude_pseudolabel import (LABEL_POLICIES,
                                                    claude_label_towel_folder)

    if label_policy not in LABEL_POLICIES:
        _err(f"--label-policy must be one of {list(LABEL_POLICIES)}")
        raise typer.Exit(1)
    res = claude_label_towel_folder(input, out, max_images=max_images, model=model,
                                    min_confidence=min_confidence, resume=resume, force=force,
                                    label_policy=label_policy, log=_echo)
    status = res.get("status")
    if status == "no_api_key":
        _err("Anthropic API key not configured.")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    if status != "ok":
        _err(res.get("message", "Claude labeling failed."))
        raise typer.Exit(1)
    _ok(f"Pseudo-labeled {res['num_images']} images -> {out}  "
        f"({res['review_needed']} need review, {res['errors']} errors)")
    usage = res.get("usage", {})
    _echo(f"  tokens in/out: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}  "
          f"est cost ~${res.get('cost_estimate_usd', 0)} (approx — verify Anthropic pricing)")
    _echo("  ⚠️ pseudo-labels can be WRONG — review before training:")
    _echo(f"  next: terafold review-pseudolabels --dataset {out}")


@app.command("openai-label-towel-folder")
def openai_label_towel_folder_cmd(
    input: str = typer.Option(..., help="Raw/candidate manifest, a create-towel-real-dataset "
                              "dataset, a crop dataset, an images/ dir, or a plain image folder."),
    out: str = typer.Option(..., help="Output pseudo-labeled towel dataset dir."),
    max_images: int = typer.Option(500, help="Max images to send to OpenAI (use a small N to test cheaply)."),
    model: str = typer.Option("gpt-5.5", help="OpenAI vision model id (Responses API)."),
    min_confidence: float = typer.Option(0.75, help="Below this -> review_needed."),
    label_policy: str = typer.Option(
        "striped_towel_visible_outer_corners",
        help="Labeling policy (default: striped-towel-only, visible outer corners)."),
    multipass: bool = typer.Option(
        False, "--multipass",
        help="HIGH-PRECISION multi-stage pipeline: localize -> crop+zoom -> corner-label "
             "-> remap -> verify -> geometry-gate (more calls per image, tighter corners)."),
    crop_padding: float = typer.Option(0.08, help="[multipass] padding around the detected towel bbox."),
    verify: bool = typer.Option(True, "--verify/--no-verify",
                                help="[multipass] run the verification pass on the remapped corners."),
    debug: bool = typer.Option(False, "--debug",
                               help="[multipass] log each stage (provider/model/image/extracted) and "
                                    "save raw-response snippets to <out>/debug/ on parse failure."),
    resume: bool = typer.Option(False, "--resume", help="Skip images already labeled in --out."),
    force: bool = typer.Option(False, "--force", "--relabel",
                               help="Re-label every image, overwriting old labels."),
):
    """Pseudo-label towel images with OpenAI Vision (second opinion; geometry only; human review required).

    A second labeler for when Claude labels the bed/background instead of the beige/
    white striped towel. Same towel schema, validation, overlays, and review gating.
    Add ``--multipass`` for the high-precision localize → crop → verify pipeline.
    """
    from terafold.towel.openai_pseudolabel import openai_label_towel_folder

    res = openai_label_towel_folder(input, out, max_images=max_images, model=model,
                                    min_confidence=min_confidence, resume=resume, force=force,
                                    label_policy=label_policy, multipass=multipass,
                                    crop_padding=crop_padding, verify=verify, debug=debug, log=_echo)
    status = res.get("status")
    if status == "no_api_key":
        _err("OpenAI API key not configured.")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    if status != "ok":
        _err(res.get("message", "OpenAI labeling failed."))
        raise typer.Exit(1)
    if multipass:
        _ok(f"Multipass-labeled {res['num_images']} images -> {out}  "
            f"({res['review_needed']} need review, {res['label_rejected']} label-rejected, "
            f"{res['rejected_localization']} localization-rejected, {res['errors']} errors)")
    else:
        _ok(f"Pseudo-labeled {res['num_images']} images -> {out}  "
            f"({res['review_needed']} need review, {res['errors']} errors)")
    usage = res.get("usage", {})
    _echo(f"  tokens in/out: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}  "
          "(verify current OpenAI pricing for this model)")
    _echo("  ⚠️ pseudo-labels can be WRONG — review before training:")
    _echo(f"  next: terafold review-pseudolabels --dataset {out}")


@app.command("review-pseudolabels")
def review_pseudolabels_cmd(
    dataset: str = typer.Option(..., help="Pseudo-labeled towel dataset dir."),
    all: bool = typer.Option(False, "--all", help="Review every pseudo-label, not just pending."),
    source: Optional[str] = typer.Option(None, help="Only review this provenance (e.g. openai_generated)."),
    category: Optional[str] = typer.Option(None, help="Only review this category_target (e.g. positive_flat)."),
):
    """Approve / reject / edit Claude pseudo-labels one by one; writes review_report.md."""
    from terafold.towel.review import review_pseudolabels

    res = review_pseudolabels(dataset, log=_echo, only_pending=not all,
                              source_filter=source, category_filter=category)
    if res.get("status") != "ok":
        _err(res.get("message", "review failed"))
        raise typer.Exit(1)
    _ok(f"Reviewed: {res['approved']} approved, {res['rejected']} rejected, "
        f"{res['needs_review']} still need review.")
    _echo(f"  report: {res['report']}")


@app.command("approve-high-confidence-pseudolabels")
def approve_high_confidence_pseudolabels_cmd(
    dataset: str = typer.Option(..., help="Pseudo-labeled towel dataset dir."),
    min_confidence: float = typer.Option(0.90, help="Minimum Claude confidence to auto-approve."),
    max_geometry_error: float = typer.Option(0.10, help="Max quad geometry error to auto-approve."),
    source: Optional[str] = typer.Option(None, help="Only consider this provenance (e.g. openai_generated)."),
    category: Optional[str] = typer.Option(None, help="Only consider this category_target."),
):
    """Auto-approve ONLY high-confidence, geometrically-sound pseudo-labels (rest stay for review)."""
    from terafold.towel.review import approve_high_confidence_pseudolabels

    res = approve_high_confidence_pseudolabels(
        dataset, min_confidence=min_confidence, max_geometry_error=max_geometry_error,
        source_filter=source, category_filter=category, log=_echo)
    if res.get("status") != "ok":
        _err(res.get("message", "auto-approve failed"))
        raise typer.Exit(1)
    _ok(f"Auto-approved {res['auto_approved']} / {res['considered']} pseudo-labels; "
        f"{res['needs_review']} left for human review.")
    _echo("  ⚠️ pseudo-labels can be wrong — spot-check overlays before training.")


# --------------------------------------------------------------------------
# OpenAI-generated towel image pipeline (Layer B — realism bridge)
# --------------------------------------------------------------------------


@app.command("preview-openai-towel-prompts")
def preview_openai_towel_prompts_cmd(
    task: str = typer.Option("mix_flat_wrinkled_negatives", help="Prompt curriculum task/mix."),
    num: int = typer.Option(20, help="How many sample prompts to show."),
    seed: int = typer.Option(0, help="Seed for reproducible prompt sampling."),
):
    """Preview the generated-towel prompt curriculum (no API calls, no cost)."""
    from terafold.towel.prompt_bank import (allocate_counts, build_prompt_curriculum,
                                            list_tasks)

    if task not in list_tasks():
        _err(f"unknown task {task!r}; known: {list_tasks()}")
        raise typer.Exit(1)
    recs = build_prompt_curriculum(task, num, seed=seed)
    _ok(f"Task '{task}' — {len(recs)} sample prompts (target mix: {allocate_counts(task, num)})")
    for i, r in enumerate(recs, 1):
        _echo(f"\n[{i}] ({r['category']} → target_state={r['target_state']}, "
              f"usable_hint={r['usable_hint']})")
        _echo(f"    {r['prompt']}")


@app.command("generate-openai-towel-dataset")
def generate_openai_towel_dataset_cmd(
    out: str = typer.Option(..., help="Output generated-dataset directory."),
    num_images: int = typer.Option(100, help="Number of images to generate."),
    model: str = typer.Option("gpt-image-1", help="OpenAI image model (e.g. gpt-image-1, dall-e-3)."),
    size: str = typer.Option("1024", help="Image size: 1024 | 1024x1536 | 1536x1024 | auto."),
    task: str = typer.Option("mix_flat_wrinkled_negatives", help="Prompt curriculum task/mix."),
    quality: str = typer.Option("medium", help="low | medium | high (gpt-image-1) / standard | hd (dall-e)."),
    seed: int = typer.Option(0, help="Seed for reproducible prompt sampling."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run",
                                 help="Preview prompts+cost without calling the API (default ON)."),
    max_cost_usd: Optional[float] = typer.Option(None, help="Refuse if predicted cost exceeds this."),
    confirm: bool = typer.Option(False, "--yes-i-understand-this-uses-paid-api",
                                 help="Required acknowledgement for REAL paid generation."),
    resume: bool = typer.Option(False, "--resume", help="Skip images already generated in --out."),
):
    """Generate photorealistic hotel-towel images via OpenAI (cost-gated; dry-run by default)."""
    from terafold.towel.openai_generation import generate_openai_towel_dataset

    res = generate_openai_towel_dataset(
        out, num_images, model=model, size=size, task=task, quality=quality, seed=seed,
        dry_run=dry_run, max_cost_usd=max_cost_usd, confirm=confirm, resume=resume, log=_echo)
    status = res.get("status")
    if status == "dry_run":
        est = res["cost_estimate"]
        _ok(f"[dry-run] would generate {res['num_requested']} images -> {out} "
            f"(est ~${est['total_usd']}, {'known' if est['known_pricing'] else 'UNKNOWN'} pricing).")
        _echo(f"  by category: {res['by_category']}")
        _echo(f"  preview {len(res['preview'])} prompts; full plan in {res['manifest']}")
        _echo("  to really generate: add --no-dry-run --max-cost-usd <cap> "
              "--yes-i-understand-this-uses-paid-api")
        return
    if status == "no_api_key":
        _err("OPENAI_API_KEY not configured.")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    if status == "needs_confirmation":
        _err(res["message"])
        raise typer.Exit(2)
    if status == "cost_exceeded":
        _err(res["message"])
        raise typer.Exit(3)
    if status == "sdk_missing":
        _err("OpenAI SDK not installed.")
        _print_lines(res.get("instructions"))
        raise typer.Exit(2)
    if status != "ok":
        _err(res.get("message", "generation failed"))
        raise typer.Exit(1)
    _ok(f"Generated {res['num_images']} images -> {out}  "
        f"(~${res['cost_actual_usd']} actual, {res['errors']} errors, {res['resumed']} resumed)")
    _echo(f"  by category: {res['by_category']}")
    _echo(f"  next: terafold filter-generated-towel-images --input {out} --out data/towel_openai_qc_v0")


@app.command("filter-generated-towel-images")
def filter_generated_towel_images_cmd(
    input: str = typer.Option(..., help="A generated (or raw) image dataset dir."),
    out: str = typer.Option(..., help="Output QC'd candidate dataset dir."),
    min_size: int = typer.Option(64, help="Drop images whose smaller side is below this (px)."),
    max_images: Optional[int] = typer.Option(None, help="Cap kept images."),
    near_dup_hamming: int = typer.Option(5, help="Perceptual near-duplicate Hamming threshold."),
    no_perceptual: bool = typer.Option(False, "--no-perceptual", help="Disable perceptual dedup."),
    require_metadata: bool = typer.Option(False, "--require-metadata",
                                          help="Reject samples missing generation lineage."),
):
    """QC generated images (validity, dims, exact+perceptual dedup, metadata) -> candidates."""
    from terafold.towel.qc import qc_filter_generated_images

    res = qc_filter_generated_images(
        input, out, min_size=min_size, max_images=max_images, near_dup_hamming=near_dup_hamming,
        dedup_perceptual=not no_perceptual, require_metadata=require_metadata, log=_echo)
    if res.get("status") != "ok":
        _err(res.get("message", "QC failed"))
        raise typer.Exit(1)
    _ok(f"QC kept {res['kept']} / {res['num_input']} -> {out}  ({res['rejected']} rejected)")
    _echo(f"  reasons: {res['reasons']}  incomplete_metadata: {res['incomplete_metadata']}")
    _echo(f"  next: terafold claude-label-towel-folder --input {out} "
          "--out data/towel_openai_labeled_v0 --min-confidence 0.75")


@app.command("summarize-openai-towel-dataset")
def summarize_openai_towel_dataset_cmd(
    data: str = typer.Option(..., help="A generated towel dataset dir."),
    out: Optional[str] = typer.Option(None, help="Optional Markdown report path."),
):
    """Summarize a generated dataset: counts by category / status / size + cost."""
    from terafold.towel.generated_dataset import summarize_generated_dataset

    res = summarize_generated_dataset(data, out=out, log=_echo)
    _ok(f"{res['num_images']} generated images (~${res['cost_actual_usd']}); "
        f"usable_for_training={res['usable_for_training']}")
    if res.get("report"):
        _echo(f"  report: {res['report']}")


@app.command("recommend-towel-training-plan")
def recommend_towel_training_plan_cmd(
    data: str = typer.Option(..., help="A towel dataset / merged dataset / YOLO pose dataset dir."),
    out: Optional[str] = typer.Option(None, help="Optional Markdown report path."),
):
    """Report dataset composition + print the recommended staged training strategy."""
    from terafold.towel.training_plan import recommend_towel_training_plan

    res = recommend_towel_training_plan(data, out=out, log=_echo)
    if res.get("status") != "ok":
        _err(res.get("message", "could not build plan"))
        raise typer.Exit(1)
    if res.get("report"):
        _echo(f"  report: {res['report']}")


@app.command("auto-triage-pseudolabels")
def auto_triage_pseudolabels_cmd(
    dataset: str = typer.Option(..., help="Pseudo-labeled towel dataset dir."),
    policy: str = typer.Option("pose_positive_strict",
                               help="pose_positive_strict (flat/wrinkled only) | "
                                    "outer_visible_corners_any_state (any state, 4 visible corners)."),
    apply: bool = typer.Option(False, "--apply/--dry-run",
                               help="Apply changes (default: dry-run, no labels modified)."),
    min_confidence_approve: float = typer.Option(0.88, help="Min confidence to auto-approve."),
    min_confidence_reject: float = typer.Option(0.75, help="Below this -> auto-reject."),
    source: Optional[str] = typer.Option(None, help="Only triage this provenance."),
    category: Optional[str] = typer.Option(None, help="Only triage this category_target."),
):
    """Auto-approve/reject/keep-for-review pseudo-labels by policy (does NOT blindly approve)."""
    from terafold.towel.review import TRIAGE_POLICIES, auto_triage_pseudolabels

    if policy not in TRIAGE_POLICIES:
        _err(f"--policy must be one of {list(TRIAGE_POLICIES)}")
        raise typer.Exit(1)
    res = auto_triage_pseudolabels(
        dataset, policy=policy, apply=apply, min_confidence_approve=min_confidence_approve,
        min_confidence_reject=min_confidence_reject, source_filter=source,
        category_filter=category, log=_echo)
    if res.get("status") != "ok":
        _err(res.get("message", "triage failed"))
        raise typer.Exit(1)
    mode = "applied" if apply else "dry-run (no labels modified)"
    _ok(f"Auto-triage [{policy}, {mode}]: {res['auto_approved']} approved, "
        f"{res['auto_rejected']} rejected, {res['review_needed']} need review "
        f"(of {res['considered']}).")
    _echo(f"  reasons: {res['reason_histogram']}")
    _echo(f"  report: {res['report']}")
    if not apply:
        _echo("  (dry-run) re-run with --apply to write the label changes.")


@app.command("build-towel-critic-from-review")
def build_towel_critic_from_review_cmd(
    dataset: str = typer.Option(..., help="A reviewed/triaged towel dataset dir."),
    out: str = typer.Option(..., help="Output critic dataset dir."),
):
    """Build a two-class scene-usability critic dataset (usable / not_usable) from review."""
    from terafold.towel.critic_dataset import build_towel_critic_from_review

    res = build_towel_critic_from_review(dataset, out, log=_echo)
    if res.get("status") != "ok":
        _err(res.get("message", "critic build failed"))
        raise typer.Exit(1)
    _ok(f"Built critic dataset: {res['num_images']} samples -> {out}")
    _echo(f"  by class: {res['by_class']}  (undecided skipped: {res['skipped_undecided']})")


# --------------------------------------------------------------------------
# YAM / MolmoAct2 (dual I2RT YAM arms — SHADOW MODE ONLY, no hardware commands)
# --------------------------------------------------------------------------


_YAM_STATUS_COLORS = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}


@app.command("yam-check-assets")
def yam_check_assets_cmd(
    config: str = typer.Option("configs/yam_dual_reference.yaml",
                               help="Dual-YAM/MolmoAct2 reference config."),
):
    """Validate YAM config + assets (repos/checkpoints/URDF). Never needs hardware."""
    from terafold.yam.assets import check_assets

    res = check_assets(config)
    for c in res["checks"]:
        tag = typer.style(f"[{c['status']:<4}]", fg=_YAM_STATUS_COLORS[c["status"]])
        typer.echo(f"{tag} {c['name']}: {c['detail']}")
        if c.get("fix"):
            _echo(f"       next: {c['fix']}")
    if "camera_order" in res:
        _echo(f"camera_order: {res['camera_order']}")
        _echo(f"norm_tag:     {res['norm_tag']}")
        _echo(f"model:        {res['model']}")
    status = res["status"]
    line = f"SUMMARY: {status}"
    if status == "FAIL":
        _err(line)
        raise typer.Exit(1)
    typer.echo(typer.style(line, fg=_YAM_STATUS_COLORS[status]))


@app.command("yam-molmoact2-smoke-test")
def yam_molmoact2_smoke_test_cmd(
    model: str = typer.Option("allenai/MolmoAct2-BimanualYAM", help="HF model id or local path."),
    dtype: str = typer.Option("bfloat16", help="bfloat16 (GPU) | float32 (CPU debug)."),
    out: str = typer.Option("runs/yam_molmoact2_smoke/actions.json",
                            help="Predicted-actions JSON (metadata.json lands beside it)."),
    top_image: Optional[str] = typer.Option(None, help="Optional real top image (else dummy)."),
    left_image: Optional[str] = typer.Option(None, help="Optional real left image (else dummy)."),
    right_image: Optional[str] = typer.Option(None, help="Optional real right image (else dummy)."),
    task: str = typer.Option("fold the towel in half neatly", help="Task text for the policy."),
    config: Optional[str] = typer.Option(None, help="Optional YAM config (camera order/norm tag)."),
    device: Optional[str] = typer.Option(None, help="Override device map (e.g. cuda:0)."),
    max_new_tokens: int = typer.Option(256, help="Generation budget for action tokens."),
    mock: bool = typer.Option(False, "--mock", help="Built-in mock adapter — no torch/GPU needed."),
    revision: Optional[str] = typer.Option(None, help="Pin the HF repo revision/commit "
                                           "(recommended: trust_remote_code runs hub code)."),
):
    """Load MolmoAct2-BimanualYAM, predict once on (dummy) images, save actions.json.

    Pure inference smoke test: NOTHING is ever sent to robot hardware.
    """
    from terafold.yam.molmoact2_smoke import run_smoke_test

    res = run_smoke_test(model_name=model, dtype=dtype, out=out, top_image=top_image,
                         left_image=left_image, right_image=right_image, task=task,
                         config_path=config, device=device, max_new_tokens=max_new_tokens,
                         mock=mock, revision=revision, log=_echo)
    if res["status"] == "deps_missing":
        _err("Missing dependencies for MolmoAct2 inference.")
        for line in res["instructions"]:
            _echo(f"  {line}")
        raise typer.Exit(2)
    if res["status"] != "ok":
        _err(f"{res['status']}: {res.get('message', '')}")
        if res.get("hint"):
            _echo(f"  hint: {res['hint']}")
        raise typer.Exit(1)
    _ok(f"Smoke test OK{' (mock)' if res['mock'] else ''}: action_shape={res['action_shape']}")
    _echo(f"  actions:  {res['actions_path']}")
    _echo(f"  metadata: {res['metadata_path']}")
    _echo("  hardware_commanded: false (shadow stack — no robot I/O exists here)")


@app.command("yam-create-dummy-episode")
def yam_create_dummy_episode_cmd(
    out: str = typer.Option(..., help="Episode dir, e.g. "
                            "data/yam_episodes/towel_fold_smoke/episode_000001"),
    frames: int = typer.Option(10, help="Number of timesteps."),
    task: str = typer.Option("fold the towel in half neatly", help="Task text."),
    seed: int = typer.Option(0, help="Determinism seed recorded in metadata."),
    fps: float = typer.Option(10.0, help="Nominal frame rate for timestamps."),
    config: Optional[str] = typer.Option(None, help="Optional YAM config (arms/cameras/dims)."),
):
    """Write a dummy YAM episode (states/actions/camera PNGs) for offline testing."""
    from terafold.yam.config import load_yam_config
    from terafold.yam.dataset import create_dummy_episode
    from terafold.yam.safety import ShadowModeViolation

    try:
        cfg = load_yam_config(config) if config else None
        res = create_dummy_episode(out, frames=frames, task=task, seed=seed, fps=fps, config=cfg)
    except ShadowModeViolation as e:
        _err(f"bad_config: {e}")
        raise typer.Exit(1)
    except (FileNotFoundError, ValueError) as e:
        _err(f"bad_config: {e}")
        raise typer.Exit(1)
    _ok(f"Dummy episode written: {res['out']}  ({res['frames']} frames, "
        f"cameras={res['cameras']}, action_dim={res['action_dim']})")
    _echo("  next: terafold yam-shadow-policy --episode " + res["out"] + " --mock")


@app.command("yam-build-isaac-scene")
def yam_build_isaac_scene_cmd(
    config: str = typer.Option("configs/yam_dual_reference.yaml",
                               help="Dual-YAM/MolmoAct2 reference config."),
    out: str = typer.Option("sim/yam_dual_towel_scene", help="Scene package output dir."),
):
    """Generate the Isaac Sim digital-twin scene package (visual only, no hardware).

    Writes scene_config.json + a standalone build_isaac_scene.py + README.md.
    Needs neither Isaac Sim nor a GPU — the USD build itself runs later, inside
    the Isaac Sim container (see yam-print-isaac-cloud-commands).
    """
    from terafold.yam.isaac_scene import generate_isaac_scene

    res = generate_isaac_scene(out, config_path=config)
    if res["status"] != "ok":
        _err(f"{res['status']}: {res.get('message', '')}")
        raise typer.Exit(1)
    _ok(f"Isaac scene package written: {res['out']}")
    for f in res["files"]:
        _echo(f"  {res['out']}/{f}")
    _echo(f"  cameras (order): {res['camera_order']}   arms: {res['arms']}")
    _echo("  visual digital twin only — no hardware, no cloth policy training")
    _echo("  next: terafold yam-print-isaac-cloud-commands")


@app.command("yam-print-isaac-cloud-commands")
def yam_print_isaac_cloud_commands_cmd(
    scene_dir: str = typer.Option("sim/yam_dual_towel_scene",
                                  help="Scene package dir to mount into the container."),
):
    """Print the exact docker/Isaac Sim container commands for a cloud GPU."""
    from terafold.yam.isaac_scene import isaac_cloud_commands

    for line in isaac_cloud_commands(scene_dir=scene_dir):
        _echo(line)


@app.command("yam-shadow-policy")
def yam_shadow_policy_cmd(
    episode: str = typer.Option(..., help="Recorded episode dir (see yam-create-dummy-episode)."),
    model: str = typer.Option("allenai/MolmoAct2-BimanualYAM", help="HF model id or local path."),
    dtype: str = typer.Option("bfloat16", help="bfloat16 (GPU) | float32 (CPU debug)."),
    out: str = typer.Option("runs/yam_shadow_smoke", help="Output dir for JSON (+ optional .rrd)."),
    first_frame_only: bool = typer.Option(False, "--first-frame-only",
                                          help="Smoke mode: predict only frame 0."),
    max_frames: Optional[int] = typer.Option(None, help="Cap frames processed."),
    max_new_tokens: int = typer.Option(256, help="Generation budget per frame."),
    mock: bool = typer.Option(False, "--mock", help="Built-in mock adapter — no torch/GPU needed."),
    no_rerun: bool = typer.Option(False, "--no-rerun", help="Skip Rerun logging even if installed."),
    revision: Optional[str] = typer.Option(None, help="Pin the HF repo revision/commit "
                                           "(recommended: trust_remote_code runs hub code)."),
):
    """Replay a recorded episode through the policy in SHADOW mode (predict + compare only).

    Predictions are saved and diffed against the recorded actions; no command
    ever reaches hardware (execution does not exist in this sprint).
    """
    from terafold.yam.shadow import run_shadow_policy

    res = run_shadow_policy(episode, model_name=model, dtype=dtype, out=out,
                            first_frame_only=first_frame_only, max_frames=max_frames,
                            mock=mock, use_rerun=not no_rerun, revision=revision,
                            max_new_tokens=max_new_tokens, log=_echo)
    if res["status"] == "deps_missing":
        _err("Missing dependencies for MolmoAct2 inference.")
        for line in res["instructions"]:
            _echo(f"  {line}")
        if res.get("hint"):
            _echo(f"  hint: {res['hint']}")
        raise typer.Exit(2)
    if res["status"] != "ok":
        _err(f"{res['status']}: {res.get('message', '')}")
        if res.get("hint"):
            _echo(f"  hint: {res['hint']}")
        raise typer.Exit(1)
    _ok(f"Shadow run OK: {res['frames_predicted']} frames predicted, "
        f"{res['frames_compared']} compared -> {res['out']}")
    _echo(f"  {res['rerun']}")
    _echo("  hardware_commanded: false (shadow mode — nothing was sent to the YAMs)")


def main() -> None:  # console-script friendly
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
