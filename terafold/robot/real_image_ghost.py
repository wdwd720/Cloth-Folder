"""Image -> plan -> real-robot GHOST fold (above the table) for the 7-DOF arm.

This runs the REAL perception/planning pipeline (no hardcoded fold), prints the
image-space and normalized table-space grasp/place, and only ever executes a
conservative, bounded **base-yaw sweep** ghost (no kinematics, no table contact).
Every safety gate must pass before any servo moves; default is dry-run.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from terafold.robot.arm_config import load_arm_config
from terafold.robot.joint_map import load_joint_map, save_joint_map, JointMap
from terafold.robot.real_motion import (countdown, ghost_fold_core, joint_space_ghost_fold, motion_logger,
                                        print_estop_instructions)
from terafold.robot.safety import SafetyError, require_motion_enabled

__all__ = ["run_real_image_ghost_fold", "run_map_servo_joints", "perceive_and_plan"]


def _noop(_m: str) -> None:
    pass


def perceive_and_plan(image_path: str, task_path: str, mode: str = "markers",
                      calibration: Optional[str] = None) -> Dict[str, Any]:
    """Run the actual perception + planning pipeline on an image."""
    from terafold.config.load import load_task_config
    from terafold.perception import perceive
    from terafold.physics.fold_geometry import grasp_place_from_keypoints
    from terafold.planning.fold_plan import plan_fold
    from terafold.planning.fold_task import FoldTask
    from terafold.vision.imageio import imread

    task = FoldTask.from_config(load_task_config(task_path))
    image = imread(image_path)
    h, w = image.shape[0], image.shape[1]
    pr = perceive(image, mode=mode, direction=task.direction)

    # Image-space grasp/place (PIXELS) from the detected corners.
    gp_img = grasp_place_from_keypoints(pr.fold_state.keypoints, direction=task.direction)
    gi = np.asarray(gp_img.grasp, dtype=float)[:2]
    pi = np.asarray(gp_img.place, dtype=float)[:2]
    norm_g = [round(float(gi[0] / w), 4), round(float(gi[1] / h), 4)]
    norm_p = [round(float(pi[0] / w), 4), round(float(pi[1] / h), 4)]

    # plan_fold(frames=None) maps the image into an ASSUMED (uncalibrated) table frame.
    plan_img = plan_fold(pr.fold_state, task, frames=None)
    table = {
        "calibrated": False,
        "assumed_grasp_m": [round(float(x), 4) for x in np.asarray(plan_img.grasp_place.grasp)[:2]],
        "assumed_place_m": [round(float(x), 4) for x in np.asarray(plan_img.grasp_place.place)[:2]],
    }
    if calibration:
        from terafold.camera.homography import load_homography

        frames = load_homography(calibration)
        plan_tbl = plan_fold(pr.fold_state, task, frames=frames)
        table.update({
            "calibrated": bool(plan_tbl.calibrated),
            "grasp_m": [round(float(x), 4) for x in np.asarray(plan_tbl.grasp_place.grasp)[:2]],
            "place_m": [round(float(x), 4) for x in np.asarray(plan_tbl.grasp_place.place)[:2]],
        })

    return {
        "task": task,
        "direction": task.direction,
        "perception": pr.perception,
        "confidence": round(float(pr.confidence), 3),
        "corners_image": [[round(float(c[0]), 1), round(float(c[1]), 1)]
                          for c in pr.fold_state.keypoints.corners],
        "image_shape": [int(w), int(h)],
        "grasp_image": [round(float(gi[0]), 1), round(float(gi[1]), 1)],
        "place_image": [round(float(pi[0]), 1), round(float(pi[1]), 1)],
        "grasp_normalized": norm_g,
        "place_normalized": norm_p,
        "table_space": table,
        "plan_image": plan_img,
    }


def run_real_image_ghost_fold(
    image_path: str,
    robot: str,
    port: Optional[str] = None,
    height_clearance_m: float = 0.10,
    speed: str = "very_slow",
    calibration: Optional[str] = None,
    task_path: str = "configs/task_fold_towel_half.yaml",
    mode: str = "model",
    enable_motion: bool = False,
    acknowledge: bool = False,
    backend: object = None,
    log: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    perception_backend: str = "classical",
    weights: Optional[str] = None,
    plan_critic: Optional[str] = None,
    min_perception_confidence: float = 0.75,
    min_plan_score: float = 0.65,
) -> Dict[str, Any]:
    """Image -> plan -> (dry-run | real) ghost fold above the table. Returns a report.

    ``perception_backend='yolo_towel_pose'`` routes to the trained towel corner-pose
    detector + rule-based plan critic (dry-run only): it detects corners, computes the
    fold plan, scores it, and refuses low-confidence / bad-view / multiple-towel /
    not-towel / already-folded cases. It NEVER sends a hardware command and contact
    stays locked.
    """
    log = log or _noop
    if perception_backend == "yolo_towel_pose":
        return _towel_pose_ghost_dryrun(
            image_path, robot, port=port, weights=weights, plan_critic=plan_critic,
            task_path=task_path, calibration=calibration,
            min_perception_confidence=min_perception_confidence, min_plan_score=min_plan_score,
            height_clearance_m=height_clearance_m, enable_motion=enable_motion,
            acknowledge=acknowledge, log=log,
        )
    cfg = load_arm_config(robot)
    port = port or cfg.port
    clearance = max(float(height_clearance_m), float(cfg.table_clearance_m))
    logger = motion_logger("real_image_ghost_fold")

    # 1. Real perception + planning.
    pp = perceive_and_plan(image_path, task_path, mode=mode, calibration=calibration)
    plan_img = pp["plan_image"]
    traj = plan_img.trajectory
    ghost = ghost_fold_core(traj.waypoints, list(traj.gripper), list(traj.phases),
                            height_clearance_m=clearance)

    joint_map = load_joint_map(robot)
    calibration_present = bool(pp["table_space"].get("calibrated"))
    report: Dict[str, Any] = {
        "status": "dry_run",
        "robot": robot, "port": port, "direction": pp["direction"],
        "perception": pp["perception"], "confidence": pp["confidence"],
        "grasp_image": pp["grasp_image"], "place_image": pp["place_image"],
        "grasp_normalized": pp["grasp_normalized"], "place_normalized": pp["place_normalized"],
        "table_space": pp["table_space"],
        "calibration_present": calibration_present,
        "contact_fold_allowed": False,  # NEVER allowed here (ghost only)
        "joint_map_present": joint_map is not None,
        "ghost": {k: ghost[k] for k in ("min_z", "touches_table", "num_waypoints", "note")},
        "ghost_waypoints": ghost["waypoints"],
        "height_clearance_m": clearance,
        "motor_commands_sent": 0,
        "simulation_only": False,
        "log": logger.path,
    }
    logger.log("image_ghost_request", {"robot": robot, "port": port, "clearance_m": clearance,
                                       "enable_motion": enable_motion, "acknowledge": acknowledge,
                                       "perception": pp["perception"], "direction": pp["direction"]})

    # 2. Print plan.
    log(f"perception     : {pp['perception']} (conf {pp['confidence']})  direction={pp['direction']}")
    log(f"grasp (image)  : {pp['grasp_image']} px   normalized {pp['grasp_normalized']}")
    log(f"place (image)  : {pp['place_image']} px   normalized {pp['place_normalized']}")
    ts = pp["table_space"]
    if ts.get("calibrated"):
        log(f"grasp/place (table m): {ts['grasp_m']} / {ts['place_m']}  [CALIBRATED]")
    else:
        log(f"table-space    : NOT calibrated (image coords are NOT robot coords). "
            f"assumed grasp/place(m)={ts['assumed_grasp_m']}/{ts['assumed_place_m']}")
    log(f"ghost fold     : {ghost['num_waypoints']} air waypoints, min_z={ghost['min_z']}m "
        f"(>= {clearance}m), touches_table={ghost['touches_table']}")
    log("CONTACT FOLD   : refused (no table calibration / contact disabled). Ghost-only.")

    # 3. Dry-run is the default.
    real = enable_motion or acknowledge
    if not real:
        log("DRY-RUN: nothing moved. To execute the ghost sweep above the table, add "
            "--enable-motion --i-understand-this-moves-hardware (needs a joint map).")
        return report

    # 4. Real-motion gates.
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as e:
        report["status"] = "refused"
        report["refusal"] = str(e)
        log(str(e))
        logger.log("refused", {"reason": "flags"})
        return report

    if joint_map is None:
        msg = ("No joint map for this robot. Run `map-servo-joints` first so the fold "
               "plan can be turned into a joint-space ghost (refusing autonomous IK; "
               "kinematics are unknown).")
        report["status"] = "refused"
        report["refusal"] = msg
        log(msg)
        logger.log("refused", {"reason": "no_joint_map"})
        return report

    # 5. Confirm the verified backend (ping + read on configured IDs).
    if backend is None:
        from terafold.robot.waveshare_sms_sts_backend import WaveshareSmsStsBackend

        backend = WaveshareSmsStsBackend(
            port=port, baudrate=cfg.baud, active_ids=cfg.active_servo_ids,
            default_speed=cfg.default_speed_units, default_acc=cfg.default_acc_units,
            safe_min_units=cfg.safe_position_units[0], safe_max_units=cfg.safe_position_units[1],
            logger=logger,
        )
        backend.confirm()
    if not backend.protocol_confirmed:
        msg = ("Protocol not confirmed: no ping/read on the configured servo IDs "
               f"{cfg.active_servo_ids}. Check power (DC 9-12.6V), the USB-C cable, the "
               "port, and that pyserial + the vendor scservo_sdk are installed.")
        report["status"] = "refused"
        report["refusal"] = msg
        log(msg)
        logger.log("refused", {"reason": "protocol_unconfirmed"})
        return report

    # 6. Read current positions (refuse if any cannot be read).
    positions = backend.read_all_positions(cfg.active_servo_ids)
    report["current_positions"] = positions
    if any(v is None for v in positions.values()):
        msg = f"Cannot read current positions for all configured IDs: {positions}. Refusing."
        report["status"] = "refused"
        report["refusal"] = msg
        log(msg)
        logger.log("refused", {"reason": "no_positions"})
        backend.close()
        return report

    # 7. Build the conservative joint-space ghost (derived; refuse if unmapped/out-of-range).
    speed_units = cfg.default_speed_units if speed == "very_slow" else int(cfg.default_speed_units * 1.5)
    js = joint_space_ghost_fold(pp["direction"], joint_map, positions,
                                max_sweep_units=cfg.max_sweep_units,
                                safe_units=tuple(cfg.safe_position_units),
                                speed=speed_units, acc=cfg.default_acc_units)
    report["joint_space"] = js
    if not js.get("ok"):
        report["status"] = "refused"
        report["refusal"] = js["refusal"]
        log(js["refusal"])
        logger.log("refused", {"reason": "joint_space", "detail": js})
        backend.close()
        return report

    # 8. Execute the bounded sweep (countdown, logging, Ctrl+C safe).
    print_estop_instructions(log)
    countdown(5, log, sleep_fn=sleep_fn)
    sent = 0
    try:
        for wp in js["waypoints"]:
            backend.write_position(wp["servo_id"], wp["target_units"], wp["speed"], wp["acc"])
            logger.log("moved", {"label": wp["label"], "id": wp["servo_id"],
                                 "target": wp["target_units"]})
            sent += 1
            log(f"  {wp['label']}: servo {wp['servo_id']} -> {wp['target_units']} units")
            sleep_fn(0.6)
        report["status"] = "moved"
    except KeyboardInterrupt:
        log("\n[ESTOP] interrupted — stopping and closing.")
        backend.emergency_stop()
        log("CUT POWER NOW (DC 9-12.6V).")
        report["status"] = "interrupted"
    finally:
        report["motor_commands_sent"] = sent
        backend.close()
    return report


def _infer_joint_role(joint: str) -> str:
    """Best-effort role from a joint name (base/shoulder/elbow/wrist/gripper)."""
    j = (joint or "").lower()
    for key in ("base", "shoulder", "elbow", "wrist", "gripper"):
        if key in j:
            return key
    return "unknown"


def run_map_servo_joints(
    backend,
    robot: str,
    ids: List[int],
    delta_units: int = 40,
    speed: str = "very_slow",
    input_fn: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
    now: float = 0.0,
    save: bool = True,
) -> Dict[str, Any]:
    """Nudge one servo at a time, ask which joint moved, and save a RICH joint map.

    The operator is asked exactly one question per servo (which joint moved). The
    sign is *inferred* from the readback delta (no extra prompt), and raw home /
    conservative raw limits / last-observed / confidence / role / notes are recorded
    so the YAML is a usable actuator description, not just ``{joint, sign}``.
    """
    if not getattr(backend, "protocol_confirmed", False):
        return {"ok": False, "refusal": "protocol not confirmed; cannot nudge servos to map them."}
    cfg = load_arm_config(robot)
    safe_lo, safe_hi = int(cfg.safe_position_units[0]), int(cfg.safe_position_units[1])
    speed_units = cfg.default_speed_units if speed == "very_slow" else int(cfg.default_speed_units * 1.5)
    mapping: Dict[int, Dict[str, Any]] = {}
    for sid in ids:
        pos = backend.read_position(sid)
        if pos is None:
            log(f"[warn] servo {sid}: cannot read position; skipping.")
            continue
        home = int(pos)
        log(f"Nudging servo {sid} by {delta_units} units (very slow)...")
        target = max(safe_lo, min(safe_hi, home + int(delta_units)))
        backend.write_position(sid, target, speed_units, cfg.default_acc_units)
        observed = backend.read_position(sid)
        observed = int(observed) if observed is not None else home
        answer = input_fn(f"  Which joint moved when servo {sid} nudged? (e.g. base_yaw) ").strip()
        backend.write_position(sid, home, speed_units, cfg.default_acc_units)  # return
        if not answer:
            continue
        delta_obs = observed - home
        sign = 1 if delta_obs >= 0 else -1
        moved = abs(delta_obs) >= max(1, int(delta_units) // 4)
        confidence = "high" if moved else "low"
        mapping[sid] = {
            "joint": answer,
            "role": _infer_joint_role(answer),
            "sign": sign,
            "raw_home": home,
            "raw_min": max(safe_lo, home - 300),   # conservative placeholder window
            "raw_max": min(safe_hi, home + 300),   # widen later via characterization
            "last_observed": observed,
            "confidence": confidence,
            "motion_notes": (f"nudged +{delta_units}u -> observed delta {delta_obs}u "
                             f"(sign {sign:+d}); raw_min/max are conservative placeholders "
                             "until characterize-servos refines them"),
            "operator_notes": "",
        }
        log(f"   servo {sid} -> {answer} (role={mapping[sid]['role']}, sign={sign:+d}, "
            f"home={home}, confidence={confidence})")
    expected = sorted(set(range(1, int(cfg.dof) + 1)))
    disabled = [i for i in expected if i not in mapping and i not in cfg.active_servo_ids]
    jm = JointMap(robot=robot, servo_joint_map=mapping, created=float(now),
                  disabled_ids=disabled,
                  note="built by map-servo-joints (nudge + operator-identified joints; "
                       "sign inferred from readback; raw limits are placeholders)")
    path = save_joint_map(jm, None) if save else None
    return {"ok": True, "path": path, "joint_map": jm, "mapped": mapping,
            "disabled_ids": disabled}


# ----------------------------------------------------------------------
# YOLO towel-pose perception backend (DRY-RUN only): detect corners -> plan ->
# rule-based critic -> gated decision. Never sends a hardware command; contact
# folding stays LOCKED. This is the perception/planning integration (no policy).
# ----------------------------------------------------------------------

# Risk reasons that hard-fail the dry-run decision (image is not a foldable towel).
_HARD_FAIL_RISKS = {"bad_view", "multiple_towels", "not_towel", "already_folded"}


def _aspect_from_bbox(bbox) -> Optional[float]:
    if not bbox or len(bbox) != 4:
        return None
    w = abs(float(bbox[2]) - float(bbox[0]))
    h = abs(float(bbox[3]) - float(bbox[1]))
    if h <= 1e-6:
        return None
    return round(w / h, 3)


def _plan_from_corners(corners: Dict[str, Any], direction: str):
    """Build grasp/place/fold-axis from 4 detected corners (image pixels)."""
    from terafold.physics.cloth_state import ClothKeypoints, FoldLine
    from terafold.physics.fold_geometry import compute_fold_line, grasp_place_from_keypoints

    def v(k):
        p = corners[k]
        return np.array([float(p[0]), float(p[1])], dtype=float)

    kp = ClothKeypoints(top_left=v("tl"), top_right=v("tr"),
                        bottom_right=v("br"), bottom_left=v("bl"))
    gp = grasp_place_from_keypoints(kp, direction=direction)
    fl: FoldLine = compute_fold_line(kp, direction)
    grasp = [round(float(x), 1) for x in np.asarray(gp.grasp)[:2]]
    place = [round(float(x), 1) for x in np.asarray(gp.place)[:2]]
    if fl.a is not None and fl.b is not None:
        a, b = np.asarray(fl.a)[:2], np.asarray(fl.b)[:2]
    else:  # derive endpoints from point ± direction
        pt, d = np.asarray(fl.point)[:2], np.asarray(fl.direction)[:2]
        a, b = pt - d * 100.0, pt + d * 100.0
    fold_axis = [[round(float(a[0]), 1), round(float(a[1]), 1)],
                 [round(float(b[0]), 1), round(float(b[1]), 1)]]
    return grasp, place, fold_axis


def _towel_pose_ghost_dryrun(
    image_path: str,
    robot: str,
    *,
    port: Optional[str] = None,
    weights: Optional[str] = None,
    plan_critic: Optional[str] = None,
    task_path: str = "configs/task_fold_towel_half.yaml",
    calibration: Optional[str] = None,
    min_perception_confidence: float = 0.75,
    min_plan_score: float = 0.65,
    height_clearance_m: float = 0.10,
    enable_motion: bool = False,
    acknowledge: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Detect towel corners with YOLO pose, plan a fold, critique it. DRY-RUN only."""
    from terafold.config.load import load_task_config
    from terafold.planning.fold_task import FoldTask
    from terafold.towel.plan_critic import critique_plan
    from terafold.towel.yolo_runtime import infer_towel_pose

    log = log or _noop
    cfg = load_arm_config(robot)
    port = port or cfg.port
    task = FoldTask.from_config(load_task_config(task_path))
    direction = task.direction
    joint_map = load_joint_map(robot)
    logger = motion_logger("towel_pose_ghost_fold")
    logger.log("towel_pose_request", {"robot": robot, "image": image_path, "weights": weights,
                                      "min_conf": min_perception_confidence,
                                      "min_plan_score": min_plan_score})

    report: Dict[str, Any] = {
        "status": "dry_run", "perception_backend": "yolo_towel_pose", "robot": robot,
        "port": port, "image": image_path, "direction": direction, "weights": weights,
        "joint_map_present": joint_map is not None, "contact_fold_allowed": False,
        "allowed_for_contact": False, "motor_commands_sent": 0, "simulation_only": False,
        "log": logger.path,
    }

    infer = infer_towel_pose(image_path, weights, log=log)
    report["perception"] = infer
    if infer.get("status") != "ok":
        report["status"] = "refused"
        report["refusal"] = infer.get("message") or f"towel pose detection: {infer.get('status')}"
        report["allowed_for_dry_run"] = False
        report["allowed_for_real_ghost"] = False
        if infer.get("install"):
            log(f"towel pose unavailable: {report['refusal']}  ({infer['install']})")
        else:
            log(f"REFUSED: {report['refusal']} (nothing moved)")
        logger.log("refused", {"reason": infer.get("status")})
        return report

    corners = infer["corners"]
    conf = float(infer.get("avg_keypoint_confidence") or 0.0)
    state = infer.get("predicted_state")
    bbox = infer.get("bbox")
    grasp, place, fold_axis = _plan_from_corners(corners, direction)

    features = {
        "corners": [corners[k] for k in ("tl", "tr", "br", "bl")],
        "aspect_ratio": _aspect_from_bbox(bbox), "state": state,
        "keypoint_confidence": conf, "grasp_point": grasp, "place_point": place,
        "fold_direction": direction, "fold_axis": fold_axis,
        "visible": infer.get("visible"),
    }
    critic = critique_plan(features, min_confidence=min_perception_confidence)
    plan_score = float(critic.get("success_probability", 0.0))
    risk_reasons = list(critic.get("risk_reasons", []))
    hard_fail = sorted(set(risk_reasons) & _HARD_FAIL_RISKS)

    conf_ok = conf >= min_perception_confidence
    score_ok = plan_score >= min_plan_score
    allowed_for_dry_run = bool(conf_ok and score_ok and not hard_fail)
    allowed_for_real_ghost = bool(allowed_for_dry_run and joint_map is not None)

    report.update({
        "corners": corners, "normalized_corners": infer.get("normalized_corners"),
        "bbox": bbox, "confidence": round(conf, 3), "towel_state": state,
        "fold_direction": direction, "grasp": grasp, "place": place, "fold_axis": fold_axis,
        "plan_score": round(plan_score, 3), "risk_score": round(float(critic.get("risk_score", 0.0)), 3),
        "risk_reasons": risk_reasons, "hard_fail_reasons": hard_fail,
        "min_perception_confidence": min_perception_confidence, "min_plan_score": min_plan_score,
        "allowed_for_dry_run": allowed_for_dry_run,
        "allowed_for_real_ghost": allowed_for_real_ghost, "allowed_for_contact": False,
    })

    log(f"image          : {image_path}")
    log(f"perception     : yolo_towel_pose (weights={weights})  conf={conf:.2f}")
    log(f"corners        : {corners}")
    log(f"towel_state    : {state}")
    log(f"fold_direction : {direction}")
    log(f"grasp / place  : {grasp} / {place}")
    log(f"plan_score     : {plan_score:.2f} (>= {min_plan_score})  risk_reasons={risk_reasons}")
    log(f"allowed_for_dry_run    : {allowed_for_dry_run}")
    log(f"allowed_for_real_ghost : {allowed_for_real_ghost}")
    log("allowed_for_contact    : False  (contact folding stays LOCKED)")
    logger.log("towel_pose_result", {"conf": conf, "plan_score": plan_score,
                                     "risk_reasons": risk_reasons,
                                     "allowed_for_dry_run": allowed_for_dry_run})

    if not allowed_for_dry_run:
        reasons = []
        if not conf_ok:
            reasons.append(f"low perception confidence {conf:.2f} < {min_perception_confidence}")
        if not score_ok:
            reasons.append(f"plan score {plan_score:.2f} < {min_plan_score}")
        if hard_fail:
            reasons.append("risk: " + ",".join(hard_fail))
        report["status"] = "refused"
        report["refusal"] = "; ".join(reasons)
        log(f"REFUSED: {report['refusal']}  (dry-run printed; nothing moved)")
        logger.log("refused", {"reasons": reasons})
        return report

    if enable_motion or acknowledge:
        # The YOLO backend is perception-only/dry-run in this version. Real ghost
        # execution uses the classical joint-space path (never contact).
        report["status"] = "refused"
        report["refusal"] = (
            "yolo_towel_pose is a DRY-RUN perception backend; real ghost execution uses "
            "the classical joint-space path (map-servo-joints + real-image-ghost-fold "
            "without --perception-backend). Nothing moved; contact stays locked.")
        log(report["refusal"])
        logger.log("refused", {"reason": "yolo_dryrun_only"})
        return report

    log("DRY-RUN: plan accepted by the critic; nothing moved. Contact stays locked.")
    return report
