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
    from terafold.vision.infer_keypoints import get_keypoint_predictor

    img = imread(image)
    predictor = get_keypoint_predictor(checkpoint)
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


def _load_task(task_path: str):
    from terafold.config.load import load_task_config
    from terafold.config.validate import validate_task_config
    from terafold.planning.fold_task import FoldTask

    cfg = load_task_config(task_path)
    warnings = validate_task_config(cfg)
    for w in warnings:
        _err(f"[config warning] {w}")
    return FoldTask.from_config(cfg)


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
    viz: Optional[str] = typer.Option(None, help="Save plan overlay image here."),
):
    """Compute a fold plan from a camera image."""
    from terafold.vision.infer_keypoints import get_keypoint_predictor
    from terafold.planning.towel_half_fold import TowelHalfFoldPlanner

    ft = _load_task(task)
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

    residual = use_residual  # path; apply_residual_correction handles loading
    planner = TowelHalfFoldPlanner(
        ft, frames=frames, keypoint_predictor=get_keypoint_predictor(checkpoint),
        residual_model=residual,
    )
    plan = planner.plan_from_image(img)
    _echo(plan.summary())
    if out:
        plan.save(out)
        _ok(f"Saved plan -> {out}")
    if viz:
        from terafold.vision.visualization import draw_fold_plan, save_visualization

        save_visualization(viz, draw_fold_plan(img.copy(), plan, frames))
        _ok(f"Saved overlay -> {viz}")


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
    from terafold.vision.infer_keypoints import get_keypoint_predictor
    from terafold.robot.mock_robot import MockRobot

    ft = _load_task(task)
    frames = _maybe_frames(calibration)
    cam = build_camera(CameraConfig(type=camera))
    cam.connect()
    img = cam.read().image
    cam.disconnect()

    planner = TowelHalfFoldPlanner(ft, frames=frames, keypoint_predictor=get_keypoint_predictor(checkpoint))
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


@app.command("inspect-hf-dataset")
def inspect_hf_dataset_cmd(repo_id: str = typer.Option(..., help="HuggingFace dataset repo id.")):
    """Inspect a HuggingFace/LeRobot dataset and judge fit for our robot."""
    from terafold.data.inspect_hf_dataset import inspect_hf_dataset

    report = inspect_hf_dataset(repo_id)
    for k, v in report.items():
        _echo(f"  {k}: {v}")


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


def main() -> None:  # console-script friendly
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
