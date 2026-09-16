#!/usr/bin/env python3
"""Move through recorded robot endpoints without a VLM or an API key.

Examples:
  .venv/bin/python scripts/replay_mujoco.py <run-directory> --gui --linear-speed 0.10
  .venv/bin/python scripts/replay_mujoco.py <run-directory> --gui --playback-type no_fail
  .venv/bin/python scripts/replay_mujoco.py <run-directory> --mode controls --no-video
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.record.images import StreamingVideoWriter
from core.sim.mujoco_recording import NativeReplayCameras, motion_frame, trajectory_rows, write_json
from core.sim.mujoco_playback import EndpointPlayer, PlaybackSettings, load_actions


def replay_endpoints(run_dir, model, data, manifest, *, settings, fps, gui, output, record_video,
                     playback_type="full"):
    import mujoco

    rows = trajectory_rows(run_dir, manifest)
    try:
        initial = next(rows, None)
    finally:
        rows.close()
    if initial is None:
        raise ValueError("The recording contains no initial simulation state")
    # This is the only restoration of a saved world state. All subsequent
    # motion is generated from endpoints and physically simulated afresh.
    mujoco.mj_setState(model, data, initial[1], manifest["state_spec"])
    mujoco.mj_forward(model, data)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    gripper = manifest.get("gripper_control", metadata.get("config", {}).get("gripper_control", {}))
    actions = load_actions(run_dir)
    planning_start = time.monotonic()
    player = EndpointPlayer(model, data, manifest, actions, settings, gripper, playback_type=playback_type)
    planning_time = time.monotonic() - planning_start
    cameras = NativeReplayCameras(model, data)
    writer = StreamingVideoWriter(output, fps) if record_video else None
    viewer = None
    sampled_frames, next_frame, last_frame = 0, 0.0, None
    wall_start = time.monotonic()
    interrupted, failure = False, None
    trace_path = output.with_name(output.stem + "_trajectory.npz")

    def frame(label):
        nonlocal viewer, sampled_frames, last_frame, wall_start
        when = float(data.time-player.start_time)
        if gui and viewer is None:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [.35, .02, .15]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth, viewer.cam.elevation = 135, -25
            wall_start = time.monotonic() - when
        if writer is not None:
            writer.append(motion_frame(cameras.render, manifest["camera_config"], when, label))
        sampled_frames += 1
        last_frame = when
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            remaining = wall_start + when - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

    try:
        frame("INITIAL")
        next_frame = 1/fps
        for label in player.execute():
            when = float(data.time-player.start_time)
            if when + 1e-9 >= next_frame:
                frame(label)
                while next_frame <= when + 1e-9:
                    next_frame += 1/fps
        if last_frame is None or data.time-player.start_time-last_frame > 1e-9:
            frame("FINAL")
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        failure = exc
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()
        cameras.close()
        player.save_trace(trace_path)
    cruise_speeds = [r["tool_speed"] for r in player.trace if r["cruise"]]
    reference_speeds = [r["target_tool_speed"] for r in player.trace if r["cruise"]]
    report = {"mode": "endpoints", "playback_type": playback_type,
              "skipped_actions": player.skipped_actions,
              "skipped_action_count": len(player.skipped_actions),
              "settings": asdict(settings), "api_requests": 0,
              "source_endpoint_records": len(actions), "source_physics_states_read": 1,
              "source_timing_used": False, "source_controls_used": False,
              "simulated_duration_s": float(data.time-player.start_time),
              "planning_time_s": planning_time, "wall_time_s": time.monotonic()-wall_start,
              "fps": fps, "frames": writer.frame_count if writer is not None else 0,
              "sampled_frames": sampled_frames, "interrupted": interrupted,
              "status": "failed" if failure else "interrupted" if interrupted else "completed",
              "output": str(output) if record_video else None, "trajectory": str(trace_path),
              "events": player.event_reports,
              "cruise_tool_speed_m_s": {"median": float(np.median(cruise_speeds)),
                                         "p05": float(np.percentile(cruise_speeds, 5)),
                                         "p95": float(np.percentile(cruise_speeds, 95))} if cruise_speeds else None,
              "target_cruise_tool_speed_m_s": {"min": float(min(reference_speeds)),
                                                "max": float(max(reference_speeds))} if reference_speeds else None,
              "max_tool_tracking_error_m": max((float(np.linalg.norm(r["eef_xyz"]-r["target_eef_xyz"]))
                                                  for r in player.trace), default=0.0),
              "final_qpos": data.qpos.tolist()}
    if "provenance" in metadata:
        from core.sim.mujoco_task import evaluate_task
        report["evaluation"] = evaluate_task(SimpleNamespace(
            provenance=metadata["provenance"], model=model, data=data, mj=mujoco))
    if failure:
        report["error"] = str(failure)
    write_json(output.with_suffix(".json"), report)
    if failure:
        raise failure
    return report


def replay(run_dir, *, mode="endpoints", speed=1.0, fps=None, gui=False, output=None, layout=None,
           record_video=True, settings=None, playback_type="full"):
    if speed <= 0 or not np.isfinite(speed):
        raise ValueError("Replay speed must be positive and finite")
    if mode not in ("endpoints", "states", "controls"):
        raise ValueError("Replay mode must be endpoints, states or controls")
    if playback_type not in ("full", "no_fail"):
        raise ValueError("Playback type must be full or no_fail")
    if mode != "endpoints" and playback_type != "full":
        raise ValueError("--playback-type no_fail requires --mode endpoints")
    if mode == "endpoints" and speed != 1:
        raise ValueError("Endpoint playback uses physical travel speeds; use --linear-speed, not --speed")
    if not gui:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    run_dir = Path(run_dir).resolve()
    path = run_dir / "trajectory_manifest.json"
    if not path.exists():
        raise FileNotFoundError("No continuous trajectory in this directory. Record a new run with the updated MuJoCo runner.")
    manifest = json.loads(path.read_text())
    if manifest["format"] != "show-harness-mujoco-trajectory-v1":
        raise ValueError("Unsupported MuJoCo trajectory format")
    if layout is not None:
        if layout not in ("front", "multiview"):
            raise ValueError("Replay layout must be front or multiview")
        manifest["camera_config"]["video_layout"] = layout
    fps = float(manifest["video_fps"] if fps is None else fps)
    if not np.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError("Replay fps must be between 1 and 60")
    model_path = run_dir / manifest["model"]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Recorded model checksum mismatch")
    try:
        if model_path.suffix == ".gz":
            with tempfile.TemporaryDirectory(prefix="mujoco_replay_") as temporary:
                decoded = Path(temporary) / "model.mjb"
                with gzip.open(model_path, "rb") as source, decoded.open("wb") as target:
                    shutil.copyfileobj(source, target)
                model = mujoco.MjModel.from_binary_path(str(decoded))
        else:
            model = mujoco.MjModel.from_binary_path(str(model_path))
    except ValueError as exc:
        raise ValueError(f"Cannot load model recorded by MuJoCo {manifest['mujoco_version']}; use a matching MuJoCo version") from exc
    data, reference = mujoco.MjData(model), mujoco.MjData(model)
    suffix = "_no_fail" if playback_type == "no_fail" else ""
    output = Path(output).resolve() if output else run_dir / f"replay_{mode}{suffix}.mp4"
    if output == run_dir / manifest["video"]:
        raise ValueError("Choose a replay output name different from the original recorded video")
    if mode == "endpoints":
        return replay_endpoints(run_dir, model, data, manifest, settings=settings or PlaybackSettings(),
                                fps=fps, gui=gui, output=output, record_video=record_video,
                                playback_type=playback_type)
    cameras = NativeReplayCameras(model, data)
    writer = StreamingVideoWriter(output, fps) if record_video else None
    sampled_frames = 0
    viewer = None
    actions_path = run_dir / "action_endpoints.jsonl"
    actions = [json.loads(line) for line in actions_path.read_text().splitlines()] if actions_path.exists() else []
    action_index = 0
    start = float(manifest["start_sim_time_s"])
    next_frame_time = start
    last_frame_time = None
    max_qpos_error = 0.0
    samples = 0
    wall_start = time.monotonic()
    stopped = False
    last_state = None
    last_time = start
    try:
        for index, (when, state, controls, inputs) in enumerate(trajectory_rows(run_dir, manifest)):
            samples += 1
            last_state, last_time = state, float(when)
            if mode == "controls":
                if index == 0:
                    mujoco.mj_setState(model, data, state, manifest["state_spec"])
                else:
                    if inputs is not None:
                        mujoco.mj_setState(model, data, inputs, manifest["input_spec"])
                    else:
                        data.ctrl[:] = controls
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                mujoco.mj_setState(model, reference, state, manifest["state_spec"])
                max_qpos_error = max(max_qpos_error, float(np.max(abs(data.qpos - reference.qpos))))
            if when + 1e-9 < next_frame_time:
                continue
            if mode == "states":
                mujoco.mj_setState(model, data, state, manifest["state_spec"])
                mujoco.mj_forward(model, data)
            if gui and viewer is None:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(model, data)
                viewer.cam.lookat[:] = [0.35, 0.02, 0.15]
                viewer.cam.distance = 1.5
                viewer.cam.azimuth, viewer.cam.elevation = 135, -25
            while action_index + 1 < len(actions) and actions[action_index]["end_sample"] < index:
                action_index += 1
            action = actions[action_index]["token"] if actions else "REPLAY"
            if writer is not None:
                writer.append(motion_frame(cameras.render, manifest["camera_config"], float(when), action))
            sampled_frames += 1
            last_frame_time = float(when)
            if viewer is not None:
                if not viewer.is_running():
                    stopped = True
                    break
                viewer.sync()
                remaining = wall_start + (when - start) / speed - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            while next_frame_time <= when + 1e-9:
                next_frame_time += speed / fps
        if last_state is None:
            raise ValueError("The recording contains no completed trajectory chunks")
        if not stopped and (last_frame_time is None or last_time - last_frame_time > 1e-9):
            if mode == "states":
                mujoco.mj_setState(model, data, last_state, manifest["state_spec"])
                mujoco.mj_forward(model, data)
            if writer is not None:
                writer.append(motion_frame(cameras.render, manifest["camera_config"], last_time, "FINAL"))
            sampled_frames += 1
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()
        cameras.close()
    report = {"mode": mode, "speed": speed, "fps": fps, "frames": writer.frame_count if writer is not None else 0,
              "sampled_frames": sampled_frames,
              "samples_read": samples, "interrupted": stopped, "output": str(output) if record_video else None,
              "simulated_duration_s": last_time - start, "wall_time_s": time.monotonic() - wall_start,
              "api_requests": 0, "max_qpos_error_vs_recording": max_qpos_error if mode == "controls" else None,
              "final_qpos": data.qpos.tolist()}
    write_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=["endpoints", "states", "controls"], default="endpoints",
                        help="endpoints (default): plan new smooth motion; states/controls: diagnostic recording replay")
    parser.add_argument("--playback-type", choices=["full", "no_fail"], default="full",
                        help="full (default): retain all actions; no_fail: omit recorded empty-grasp attempts and their automatic reopens (endpoints mode)")
    parser.add_argument("--linear-speed", type=float, default=.10, help="Endpoint tool cruise speed in m/s (default: 0.10)")
    parser.add_argument("--angular-speed", type=float, default=.5, help="Endpoint angular speed limit in rad/s")
    parser.add_argument("--joint-speed", type=float, default=1.0, help="Endpoint joint speed limit in rad/s")
    parser.add_argument("--joint-acceleration", type=float, default=4.0, help="Endpoint joint acceleration limit in rad/s²")
    parser.add_argument("--ramp-time", type=float, default=.25, help="Minimum acceleration ramp time in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="Time multiplier for diagnostic states/controls modes only")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=["front", "multiview"], help="Override the recorded video layout")
    parser.add_argument("--no-video", action="store_true", help="Skip offscreen rendering/encoding for fast GUI playback or physics verification")
    args = parser.parse_args()
    if args.mode == "endpoints" and args.speed != 1:
        parser.error("--speed is for diagnostic states/controls modes; use --linear-speed for endpoint playback")
    if args.mode != "endpoints" and args.playback_type != "full":
        parser.error("--playback-type no_fail requires --mode endpoints")
    settings = PlaybackSettings(args.linear_speed, args.angular_speed, args.joint_speed,
                                args.joint_acceleration, args.ramp_time)
    report = replay(args.run_dir, mode=args.mode, speed=args.speed, fps=args.fps,
                    gui=args.gui, output=args.output, layout=args.layout, record_video=not args.no_video,
                    settings=settings, playback_type=args.playback_type)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
