#!/usr/bin/env python3
"""Run the zero-shot Show-Harness planner/controller in a MuJoCo Panda scene."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.config import load_secrets_env, load_yaml, resolve_vlm_config
from core.launch import coarse_step_m, high_above_table_m, make_runner
from core.prompting.prompt_loader import load_prompt_dir
from core.record.episode_logger import EpisodeLogger
from core.record.images import save_png
from core.record.mujoco_recorder import CAMERAS, MujocoRecorder
from core.sim.launch import make_vlm_client
from core.sim.mujoco_session import MujocoSession
from interpreters.real_atomic_controller import RealAtomicController
from plugins.variable_step import VariableStepPlugin


def make_controller(session, cfg, *, variable_step=False):
    plugin = None
    if variable_step:
        fine, coarse, height = float(cfg["fine_step_m"]), coarse_step_m(cfg), high_above_table_m(cfg)
        if not np.isfinite([fine, coarse, height]).all() or not 0 < fine <= coarse or height < 0:
            raise ValueError("Require finite 0 < fine_step_m <= coarse_step_m and high_above_table_m >= 0")
        plugin = VariableStepPlugin(
            enabled=True, coarse_step_m=coarse, high_above_table_m=height,
            large_step_m=cfg.get("large_step_m"),
            large_above_table_m=cfg.get("large_above_table_m", 0.20),
        )
    controller = RealAtomicController.from_primitives_config(
        session.robot, load_yaml(ROOT / "configs/primitives_franka.yaml"),
        step_m=float(cfg["fine_step_m"]), z_floor_m=float(cfg["z_floor_m"]),
        settle_steps=1, settle_dt_s=0, gripper_settle_s=0,
        grasp_min_width_m=float(cfg["empty_width_m"]),
        grasp_open_width_m=float(cfg["open_width_m"]),
        gripper_close_threshold_m=float(cfg["open_width_m"]),
        variable_step_plugin=plugin,
        table_height_m=float(cfg["z_floor_m"]) if variable_step else None,
    )
    session.motion_reference_m = controller.step_m if variable_step else None
    controller.sync_from_robot()
    return controller


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default=str(ROOT / "configs/robot_mujoco.yaml"))
    parser.add_argument("--model", help="Override the configured Ollama model")
    parser.add_argument("--vlm-url", help="Override the configured API base URL")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-dir")
    parser.add_argument("--gui", action="store_true", help="On macOS launch with .venv/bin/mjpython")
    parser.add_argument("--smoke-test", action="store_true", help="Check physics/cameras without an API call")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record", action="store_true", help="Enable video recording (disabled by default)")
    parser.add_argument("--variable-step", action="store_true", help="Enable automatic 2/5/10 cm movement steps")
    args = parser.parse_args(argv)
    load_secrets_env()
    cfg = load_yaml(args.robot_config)
    cfg["plugins"] = {**(cfg.get("plugins") or {}), "variable_step": args.variable_step}
    for key in ("max_steps", "log_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if int(cfg["max_steps"]) <= 0:
        parser.error("--max-steps must be positive")
    cfg["vlm"] = resolve_vlm_config(cfg)
    if args.model:
        cfg["vlm"]["model"] = args.model
    if args.vlm_url:
        cfg["vlm"]["base_url"] = args.vlm_url
    if not args.smoke_test and cfg["vlm"]["api_key"] == "EMPTY":
        parser.error("Set OLLAMA_API_KEY in your environment or configs/secrets.env")
    session = MujocoSession(cfg, gui=args.gui)
    client = None
    try:
        controller = make_controller(session, cfg, variable_step=args.variable_step)
        if args.variable_step:
            large = controller.variable_step_plugin.large_step_m
            print(f"[mujoco] variable step: fine {controller.step_m * 100:g} cm, "
                  f"coarse {controller.variable_step_plugin.coarse_step_m * 100:g} cm"
                  + (f", large {large * 100:g} cm" if large is not None else ""))
        if args.smoke_test:
            directory = Path(cfg["log_dir"]) / "smoke_test"
            if args.record:
                session.recorder = MujocoRecorder(session, directory / "videos",
                                                  **cfg.get("recording", {}), task="Smoke test")
            obs = session.get_observation()
            for view in CAMERAS:
                save_png(directory / f"{view}.png", obs[view])
            for step_idx, token in enumerate((*controller.move_vectors, "GRASP")):
                if session.recorder is not None:
                    session.recorder.begin_step(
                        step_idx=step_idx, stage="Smoke test", token=token, source="smoke_test",
                        annotation=("Scripted far-target calibration; target_in_wrist=False; no VLM call."
                                    if args.variable_step else "Scripted motion and camera check; no VLM call."),
                    )
                before = session.get_ee_pose()[:3]
                result = controller.step(token, target_in_wrist=False if args.variable_step else None)
                if token in controller.move_vectors:
                    delta = session.get_ee_pose()[:3] - before
                    expected = controller.move_vectors[token] * result.step_m
                    if not np.allclose(delta, expected, atol=0.003):
                        raise RuntimeError(f"{token}: measured {delta}, expected {expected}")
                if session.recorder is not None:
                    session.recorder.end_step({
                        "act": token, "grasp_fail": result.grasp_empty,
                        "step_kind": result.step_kind, "step_cm": result.step_m * 100,
                    })
            if controller.gripper_closed:
                raise RuntimeError("Empty-grasp detection failed")
            if session.recorder is not None:
                session.recorder.finish(True, "smoke_test_passed")
            print(f"MuJoCo smoke test passed: six axes, gripper, three cameras. Output: {directory}")
            return 0
        client = make_vlm_client(args, cfg)
        client.health_check()
        logger = EpisodeLogger(cfg["log_dir"], task_id=0, variant="ollama",
                               video_fps=cfg.get("v0", {}).get("video_fps", 2),
                               record_video=False, primary_camera="side")
        logger.write_metadata({"config": cfg, "simulator": "mujoco", "recording_enabled": args.record})
        if args.record:
            session.recorder = MujocoRecorder(
                session, logger.run_dir / "videos", **cfg.get("recording", {}),
                model=cfg["vlm"]["model"], task=cfg["task"],
            )
        print(f"[mujoco] {cfg['vlm']['model']} at {cfg['vlm']['base_url']}")
        print(f"[mujoco] Rollout: {logger.run_dir}")
        prompts = load_prompt_dir(ROOT / "prompts")
        prompts["controller_prompt"] = prompts["controller_mujoco_prompt"]
        prompts["common_context"] += "\n" + cfg.get("observation_context", "")
        runner = make_runner(cfg, prompts, client,
                             session, controller, logger, args.debug)
        result = runner.run()
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.success else 1
    except BaseException as exc:
        if session.recorder is not None and not session.recorder._closed:
            session.recorder.event("error", annotation=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        session.close()
        if client is not None:
            client.session.close()


if __name__ == "__main__":
    raise SystemExit(main())
