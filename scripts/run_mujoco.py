#!/usr/bin/env python3
"""Run the authored RoboLab cube/bowl task with the original Show-Harness VLM loop."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import camera_contract, deep_merge, load_secrets_env, load_yaml, resolve_vlm_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=ROOT / "configs/robot_mujoco.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--model")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-vlm", action="store_true", help="Check physics/cameras without API calls.")
    parser.add_argument("--probe-axes", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cube-size", type=float, help="Cube maximum side length in metres (default 0.04).")
    parser.add_argument("--grip-force", type=float, help="Closing force in newtons per finger (default 80).")
    parser.add_argument("--video-fps", type=float, help="Continuous simulation video frame rate (default 30).")
    parser.add_argument("--video-layout", choices=["front", "multiview"], help="Video layout; VLM input views are unaffected")
    parser.add_argument("--extra-view", choices=["side"],
                        help="Camera-only variant: append one fixed side image to every VLM request.")
    parser.add_argument("--describe-side-camera", action="store_true",
                        help="Separate prompt variant: describe the added side camera's orientation.")
    parser.add_argument("--side-grasp-check", action="store_true",
                        help="Separate prompt variant: require the VLM to confirm grasp alignment in the side view.")
    args = parser.parse_args(argv)
    if args.describe_side_camera and args.extra_view != "side":
        parser.error("--describe-side-camera requires --extra-view side")
    if args.side_grasp_check and not (args.extra_view == "side" and args.describe_side_camera):
        parser.error("--side-grasp-check requires --extra-view side --describe-side-camera")
    return args


def resolve_config(args):
    import yaml
    from core.launch import DEFAULTS

    overrides = load_yaml(args.robot_config)
    # Inherit the original POLICY body, not its real-robot site/secret overlays.
    original = yaml.safe_load((ROOT / overrides["harness_config"]).read_text())
    for key in ("defaults", "overlays", "robot", "poses", "z_floors"):
        original.pop(key, None)
    cfg = deep_merge(deep_merge(DEFAULTS, original), overrides)
    if args.cube_size is not None:
        cfg["cube_size_m"] = args.cube_size
    if args.grip_force is not None:
        cfg["gripper_control"]["close_force_n"] = args.grip_force
    if args.video_fps is not None:
        cfg["recording"]["fps"] = args.video_fps
    if args.video_layout is not None:
        cfg["recording"]["layout"] = args.video_layout
    if args.extra_view == "side":
        from core.sim.mujoco_ablation import side_camera_spec
        cfg["observer_cameras"] = [side_camera_spec()]
    cfg.update(camera_contract(load_yaml(ROOT / overrides["camera_config"])))
    for key in ("max_steps", "log_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if int(cfg["max_steps"]) <= 0 or args.episodes <= 0:
        raise ValueError("--max-steps and --episodes must be positive")
    if cfg["task_class"] != "RubiksCubeTask":
        raise ValueError("Only the authored RubiksCubeTask is converted so far")
    # Read the task's instruction as data, without importing Isaac-only Python code.
    source = ROOT / cfg["task_source"]
    if not source.exists():
        raise FileNotFoundError("Original RoboLab assets are missing. Run bash scripts/setup.sh mujoco.")
    tree = ast.parse(source.read_text())
    task_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cfg["task_class"])
    instruction = next(n.value for n in task_class.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "instruction")
    cfg["task"] = ast.literal_eval(instruction)["default"]
    # Enforce the agreed raw-image policy contract, including when a config is edited.
    for key in ("coords", "affordance", "video_ref"):
        if cfg["plugins"].get(key):
            raise ValueError(f"plugins.{key} must be disabled for this raw-image evaluation")
    load_secrets_env()
    cfg["vlm"] = resolve_vlm_config(cfg)
    if args.model:
        cfg["vlm"]["model"] = args.model
    if not args.no_vlm and cfg["vlm"]["api_key"] in ("", "EMPTY"):
        raise ValueError("GEMINI_API_KEY is not set; export it or use --no-vlm")
    return cfg


def make_controller(task, cfg):
    from types import SimpleNamespace
    from core.launch import make_controller as original_make_controller
    from core.sim.mujoco_task import MujocoSession

    session = MujocoSession(task)
    args = SimpleNamespace(mock_robot=False, no_z_floor=False, z_floor_m=None, z_floor=False)
    controller = original_make_controller(cfg, load_yaml(ROOT / "configs/primitives_franka.yaml"),
                                          session, args, hardware="franka")
    controller.sync_from_robot()
    task.bind_controller(controller)
    return session, controller


def record_runtime_failure(task, logger, cfg, error):
    """Record independent scoring after an aborted episode; never feeds the policy."""
    from core.sim.mujoco_task import evaluate_task

    message = str(error)
    key = str(cfg.get("vlm", {}).get("api_key", ""))
    if key:
        message = message.replace(key, "[REDACTED]")
    evaluation = evaluate_task(task)
    evaluation.update(model_declared_success=False, episode_status="runtime_error", error=message)
    (logger.run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    path = logger.run_dir / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update(success=evaluation["success"], model_declared_success=False,
                   control_mode="mujoco", end_reason="runtime_error", error=message)
    logger.write_summary(summary)
    return evaluation


def finalize_motion_recording(logger, recorder, cfg, success, *, check_only=False):
    """Preserve the legacy decision montage and publish the continuous video."""
    if not logger._steps_file.closed:
        logger.close(success=success, fps=cfg["v0"]["video_fps"])
    if recorder is None:
        return {}
    if not recorder.closed:
        for name in ("rollout_success.mp4", "rollout_failure.mp4"):
            old = logger.run_dir / name
            if old.exists():
                old.replace(logger.run_dir / "decision_frames.mp4")
        filename = "motion_check.mp4" if check_only else ("rollout_success.mp4" if success else "rollout_failure.mp4")
        recorder.close(final_path=logger.run_dir / filename)
    return {"video_path": str(recorder.video_path),
            "decision_video_path": str(logger.run_dir / "decision_frames.mp4"),
            "trajectory_manifest": str(logger.run_dir / "trajectory_manifest.json"),
            "action_endpoints": str(logger.run_dir / "action_endpoints.jsonl"),
            "video_clock": "simulation time; API waits omitted"}


def run_episode(args, cfg, index, instrumentation=None):
    from core.action_units import MOVE_ATOMS
    from core.prompting.prompt_loader import load_prompt_dir
    from core.launch import make_runner, make_vlm_client
    from core.record.episode_logger import EpisodeLogger
    from core.record.images import save_png
    from core.sim.mujoco_task import MujocoTask, evaluate_task
    from core.sim.mujoco_recording import MujocoRecorder
    from core.sim.mujoco_ablation import AblationInstrumentation

    task = MujocoTask(cfg, gui=args.gui)
    logger = client = recorder = None
    audit = instrumentation
    try:
        session, controller = make_controller(task, cfg)
        logger = EpisodeLogger(cfg["log_dir"], task_id=index,
                               variant="physics_check" if args.no_vlm else cfg["vlm"]["model"],
                               video_fps=cfg["v0"]["video_fps"])
        logger.write_metadata({"simulator": "mujoco", "simulator_version": task.mj.__version__,
                               "config": cfg, "provenance": task.provenance,
                               "policy": "core.launch.make_runner / RealEpisodeRunner",
                               "observations": "unannotated RGB + robot proprioception"})
        print(f"[mujoco] {cfg['task_class']}: {cfg['task']}\n[mujoco] Logs: {logger.run_dir}", flush=True)
        if args.no_vlm and cfg.get("recording", {}).get("enabled", True):
            recorder = MujocoRecorder(task, logger.run_dir, cfg)
            recorder.install_controller(controller)
        if args.probe_axes:
            measurements = {}
            for token in MOVE_ATOMS:
                result = controller.step(token, step_override_m=0.02)
                measurements[token] = (result.post_pose[:3] - result.pre_pose[:3]).tolist()
                print(f"[probe] {token}: {measurements[token]}", flush=True)
            logger.write_calibration(measurements)
            if not args.no_vlm:
                task.close()
                task = MujocoTask(cfg, gui=args.gui)
                session, controller = make_controller(task, cfg)
        if recorder is None and cfg.get("recording", {}).get("enabled", True):
            recorder = MujocoRecorder(task, logger.run_dir, cfg)
            recorder.install_controller(controller)
        front, wrist = task.render()
        save_png(logger.run_dir / "initial_agentview.png", front)
        save_png(logger.run_dir / "initial_wrist.png", wrist)
        if args.no_vlm:
            task.advance(0.2)
            for camera in cfg.get("observer_cameras", []):
                save_png(logger.run_dir / f"initial_{camera['name']}.png",
                         task.render_observer(camera["name"]))
            logger.log_step(0, front, wrist, {"step_idx": 0, "task": cfg["task"], "mode": "physics_check"})
            result = {"mode": "physics_check", "evaluated": False, "run_dir": str(logger.run_dir)}
            result.update(finalize_motion_recording(logger, recorder, cfg, False, check_only=True))
            logger.write_summary(result)
            return result
        client = make_vlm_client(args, cfg)
        audit = audit or AblationInstrumentation(send_side=False)
        session = audit.install(task, session, controller, logger, client)
        if recorder is not None:
            recorder.install_client(client)
        runner = make_runner(cfg, load_prompt_dir(ROOT / "prompts"), client, session,
                             controller, logger, args.debug)
        started = time.monotonic()
        result = runner.run()
        # Scoring happens after model completion/timeout; it cannot affect policy decisions.
        task.advance(float(cfg["evaluation_settle_s"]))
        evaluation = evaluate_task(task)
        evaluation.update({"model_declared_success": result.success, "model_result": asdict(result),
                           "wall_time_s": time.monotonic() - started})
        # The existing runner reports the VLM's conclusion. Label user-facing
        # artifacts by the independent score while preserving that conclusion.
        video = logger.run_dir / ("rollout_success.mp4" if evaluation["success"] else "rollout_failure.mp4")
        if recorder is not None:
            evaluation.update(finalize_motion_recording(logger, recorder, cfg, evaluation["success"]))
        else:
            old_video = Path(result.video_path)
            if old_video.exists() and old_video != video:
                old_video.replace(video)
        evaluation["model_result"]["video_path"] = str(video)
        evaluation["video_path"] = str(video)
        summary_path = logger.run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(success=evaluation["success"], model_declared_success=result.success,
                       control_mode="mujoco", video_path=str(video))
        if recorder is not None:
            summary.update({k: evaluation[k] for k in ("trajectory_manifest", "action_endpoints", "decision_video_path", "video_clock")})
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        (logger.run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
        front, wrist = task.render()
        save_png(logger.run_dir / "final_agentview.png", front)
        save_png(logger.run_dir / "final_wrist.png", wrist)
        print(f"[mujoco] model_complete={result.success}; evaluated_success={evaluation['success']}", flush=True)
        print(f"[mujoco] Continuous video: {video}", flush=True)
        return evaluation
    except Exception as exc:
        if logger is not None:
            evaluation = record_runtime_failure(task, logger, cfg, exc)
            if recorder is not None:
                evaluation.update(finalize_motion_recording(logger, recorder, cfg, False))
                (logger.run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
                summary = json.loads((logger.run_dir / "summary.json").read_text())
                summary.update({k: evaluation[k] for k in ("video_path", "trajectory_manifest", "action_endpoints", "decision_video_path", "video_clock")})
                logger.write_summary(summary)
        raise
    finally:
        try:
            if audit is not None:
                audit.close()
            try:
                if logger is not None and not logger._steps_file.closed:
                    logger.close(success=False, fps=cfg["v0"]["video_fps"])
            finally:
                if recorder is not None and not recorder.closed:
                    finalize_motion_recording(logger, recorder, cfg, False, check_only=args.no_vlm)
        finally:
            if client is not None:
                client.session.close()
            task.close()


def main(argv=None):
    args = parse_args(argv)
    if not args.gui:
        os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = resolve_config(args)
    results = []
    for i in range(args.episodes):
        if args.extra_view == "side":
            from core.sim.mujoco_ablation import AblationInstrumentation
            result = run_episode(args, cfg, i, instrumentation=AblationInstrumentation(
                True, describe_side=args.describe_side_camera, check_side_grasp=args.side_grasp_check))
        else:
            result = run_episode(args, cfg, i)
        results.append(result)
        if result.get("model_result", {}).get("end_reason") == "interrupted":
            print("[mujoco] Interrupted; remaining episodes cancelled.", flush=True)
            return 130
    if args.no_vlm:
        return 0
    passed = sum(bool(r["success"]) for r in results)
    print(f"[mujoco] Evaluated successes: {passed}/{len(results)}", flush=True)
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"[mujoco] {exc}", file=sys.stderr)
        raise SystemExit(1)
