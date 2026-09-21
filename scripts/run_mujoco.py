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
from core.cartesian_actions import CartesianActions
from core.launch import coarse_step_m, high_above_table_m, make_runner
from core.prompting.prompt_loader import load_prompt_dir
from core.record.episode_logger import EpisodeLogger
from core.record.images import save_png
from core.record.mujoco_recorder import MujocoRecorder
from core.sim.launch import make_vlm_client
from core.sim.mujoco_session import MujocoSession
from core.sim.mujoco_success import (
    PLUG_RETREAT_MARGIN_M,
    plug_retreat_clearance_m,
    plug_success_prompt,
)
from interpreters.real_atomic_controller import RealAtomicController
from interpreters.mujoco_atomic_controller import MujocoAtomicController
from plugins.variable_step import VariableStepPlugin


def make_controller(session, cfg, *, variable_step=False):
    actions_cfg = cfg.get("cartesian_motion")
    actions = CartesianActions(actions_cfg) if actions_cfg is not None else None
    if actions is not None and variable_step:
        raise ValueError("cartesian_motion uses explicit sizes; omit --variable-step")
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
    controller_type = MujocoAtomicController if actions is not None else RealAtomicController
    controller = controller_type.from_primitives_config(
        session.robot, load_yaml(ROOT / "configs/primitives_franka.yaml"),
        step_m=float(cfg["fine_step_m"]), z_floor_m=float(cfg["z_floor_m"]),
        settle_steps=1, settle_dt_s=0, gripper_settle_s=0,
        grasp_min_width_m=float(cfg["empty_width_m"]),
        grasp_open_width_m=float(cfg["open_width_m"]),
        gripper_close_threshold_m=float(cfg["open_width_m"]),
        variable_step_plugin=plugin,
        table_height_m=float(cfg["z_floor_m"]) if variable_step else None,
        **({"cartesian_actions": actions} if actions is not None else {}),
    )
    session.motion_reference_m = (actions.translation_steps_m["medium"] if actions is not None
                                  else controller.step_m if variable_step else None)
    session.motion_reference_rad = (np.deg2rad(actions.rotation_steps_deg["medium"])
                                    if actions is not None else None)
    controller.sync_from_robot()
    return controller


def render_plug_retreat_text(text, cfg):
    """Use the same clearance setting in task, planner, controller, and evaluator."""
    clearance_mm = plug_retreat_clearance_m(cfg) * 1000
    return str(text).replace("{retreat_clearance_mm}", f"{clearance_mm:g}").replace(
        "{retreat_target_mm}", f"{clearance_mm + PLUG_RETREAT_MARGIN_M * 1000:g}",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default=str(ROOT / "configs/robot_mujoco.yaml"))
    parser.add_argument("--vlm-backend", help="Select a vlm_backends profile (openai, gemini, or ollama)")
    parser.add_argument("--model", help="Override the selected backend's model")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        help="Override reasoning effort (OpenAI defaults to medium)")
    parser.add_argument("--vlm-url", help="Override the configured API base URL")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-dir")
    parser.add_argument("--gui", action="store_true", help="On macOS launch with .venv/bin/mjpython")
    parser.add_argument("--smoke-test", action="store_true", help="Check physics/cameras without an API call")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record", action="store_true", help="Enable video recording (disabled by default)")
    parser.add_argument("--record-global", action="store_true",
                        help="Also save a separate 1920x1080 global.mp4 (implies --record)")
    parser.add_argument("--variable-step", action="store_true", help="Enable automatic 2/5/10 cm movement steps")
    parser.add_argument("--cartesian-motion", action="store_true",
                        help="Enable VLM-selected small/medium/large translations and XYZ rotations")
    args = parser.parse_args(argv)
    args.record = args.record or args.record_global
    load_secrets_env()
    cfg = load_yaml(args.robot_config)
    if args.record_global:
        cfg.setdefault("recording", {})["global_video"] = True
    if args.cartesian_motion:
        cfg.setdefault("cartesian_motion", {})
    if args.variable_step and cfg.get("cartesian_motion") is not None:
        parser.error("explicit cartesian_motion sizes cannot be combined with --variable-step")
    cfg["plugins"] = {**(cfg.get("plugins") or {}), "variable_step": args.variable_step}
    for key in ("max_steps", "log_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if int(cfg["max_steps"]) <= 0:
        parser.error("--max-steps must be positive")
    if cfg.get("scene") == "plug_insert":
        cfg["task"] = render_plug_retreat_text(cfg["task"], cfg)
    cfg["vlm"] = resolve_vlm_config(cfg, backend=args.vlm_backend)
    cfg["vlm_backend"] = cfg["vlm"]["backend"]
    if args.model:
        cfg["vlm"]["model"] = args.model
    if args.reasoning_effort:
        cfg["vlm"]["reasoning_effort"] = args.reasoning_effort
    if args.vlm_url:
        cfg["vlm"]["base_url"] = args.vlm_url
    if not args.smoke_test and cfg["vlm"]["api_key"] == "EMPTY":
        parser.error(f"Set {cfg['vlm'].get('api_key_env', 'VLLM_API_KEY')} in your environment or configs/secrets.env")
    session = MujocoSession(cfg, gui=args.gui)
    client = None
    try:
        controller = make_controller(session, cfg, variable_step=args.variable_step)
        actions = getattr(controller, "cartesian_actions", None)
        if actions is not None:
            print(f"[mujoco] explicit Cartesian steps: {actions.translation_steps_m} m; "
                  f"XYZ rotations: {actions.rotation_steps_deg} deg")
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
            for view in session.cameras:
                save_png(directory / f"{view}.png", obs[view])
            smoke_tokens = list(controller.move_vectors)
            if actions is not None:
                # Inverse pairs return to the clear initial workspace after each check.
                smoke_tokens = [f"{token}_{size.upper()}" for size in ("small", "medium", "large")
                                for token in controller.move_vectors]
                smoke_tokens += [f"ROT_{axis}_{sign}_{size.upper()}"
                                 for size in ("small", "medium", "large")
                                 for axis in "XYZ" for sign in ("POS", "NEG")]
            for step_idx, token in enumerate((*smoke_tokens, "GRASP")):
                if session.recorder is not None:
                    session.recorder.begin_step(
                        step_idx=step_idx, stage="Smoke test", token=token, source="smoke_test",
                        annotation=("Scripted far-target calibration; target_in_wrist=False; no VLM call."
                                    if args.variable_step else "Scripted motion and camera check; no VLM call."),
                    )
                before = session.get_ee_pose()
                result = controller.step(token, target_in_wrist=False if args.variable_step else None)
                if result.kind == "move":
                    delta = session.get_ee_pose()[:3] - before[:3]
                    expected = result.intended_delta_m
                    if not np.allclose(delta, expected, atol=0.003):
                        raise RuntimeError(f"{token}: measured {delta}, expected {expected}")
                elif result.kind == "rotate":
                    from scipy.spatial.transform import Rotation
                    post = session.get_ee_pose()
                    rotation_error = (Rotation.from_quat(result.target_pose[3:])
                                      * Rotation.from_quat(post[3:]).inv()).magnitude()
                    if rotation_error > np.deg2rad(1) or not np.allclose(post[:3], before[:3], atol=0.003):
                        raise RuntimeError(f"{token}: rotation/position tracking failed")
                if session.recorder is not None:
                    session.recorder.end_step({
                        "act": token, "grasp_fail": result.grasp_empty,
                        "step_kind": result.step_kind, "step_cm": result.step_m * 100,
                        **({"rotation_deg": result.rotation_deg} if result.kind == "rotate" else {}),
                    })
            if controller.gripper_closed:
                raise RuntimeError("Empty-grasp detection failed")
            if session.recorder is not None:
                session.recorder.finish(True, "smoke_test_passed")
            print(f"MuJoCo smoke test passed: {len(smoke_tokens)} movements, gripper, "
                  f"{len(session.cameras)} camera views. Output: {directory}")
            return 0
        client = make_vlm_client(args, cfg)
        client.health_check()
        logger = EpisodeLogger(cfg["log_dir"], task_id=0, variant=cfg["vlm_backend"],
                               video_fps=cfg.get("v0", {}).get("video_fps", 2),
                               record_video=False, primary_camera="front")
        logger.write_metadata({"config": cfg, "simulator": "mujoco", "recording_enabled": args.record})
        if args.record:
            session.recorder = MujocoRecorder(
                session, logger.run_dir / "videos", **cfg.get("recording", {}),
                model=cfg["vlm"]["model"], task=cfg["task"],
            )
        print(f"[mujoco] {cfg['vlm']['model']} at {cfg['vlm']['base_url']}")
        if cfg["vlm"].get("reasoning_effort"):
            print(f"[mujoco] reasoning effort: {cfg['vlm']['reasoning_effort']}")
        print(f"[mujoco] Rollout: {logger.run_dir}")
        prompts = load_prompt_dir(ROOT / "prompts")
        prompt_name = "controller_mujoco_cartesian_prompt" if actions is not None else "controller_mujoco_prompt"
        if cfg.get("scene") == "plug_insert":
            prompt_name = "controller_mujoco_plug_prompt"
            prompts[prompt_name] = render_plug_retreat_text(prompts[prompt_name], cfg)
        prompts["controller_prompt"] = prompts[prompt_name]
        # CoT backends must use the same camera/frame contract.
        prompts[f"controller_{cfg['vlm_backend']}_prompt"] = prompts[prompt_name]
        context = cfg.get("observation_context", "")
        if actions is not None:
            context = cfg.get("cartesian_observation_context", "") or (
                "Images are Wrist (A), Front (B), Right Side (C). Wrist is attached to the gripper and "
                "rotates with it. Front is on +X and Right Side on +Y, both orthographic "
                "and tilted downward 30 degrees. Prioritize Wrist for object/fingertip "
                "alignment; use Front to check left/right alignment and Right Side to "
                "check forward/back alignment. Cross-check all views for height clearance."
            )
            context += ("\nThe controller can translate along all three fixed world axes and "
                        "rotate around each of them, choosing a small, medium, or large "
                        "increment in one command. Orient held objects before contact.")
        if cfg.get("scene") == "plug_insert":
            context = render_plug_retreat_text(context, cfg) + "\n" + plug_success_prompt(cfg)
        prompts["common_context"] += "\n" + context
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
