"""Simulation-clock video, endpoint telemetry, and replayable physics traces.

These records are never inputs to the VLM. Recording every physics integration
step preserves object motion as well as robot motion, without API wait periods.
"""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from core.record.images import StreamingVideoWriter, prepare_view


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def motion_frame(render_native, camera_config, sim_time, action=""):
    """A video-only layout; no labels or resized video panels enter the policy."""
    front = render_native("front_cam")
    panels = [("Front", Image.fromarray(front))]
    if camera_config.get("video_layout", "multiview") == "multiview":
        wrist = prepare_view(render_native("wrist_cam"), **{
            key: camera_config.get(f"wrist_{key}")
            for key in ("rotation_degrees", "flip", "crop_aspect", "square_size")})
        wrist_panel = Image.new("RGB", (320, 480))
        wrist_panel.paste(Image.fromarray(wrist).resize((320, 320)), (0, 80))
        panels.append(("Wrist", wrist_panel))
        if any(c["name"] == "ablation_side" for c in camera_config.get("observer_cameras", [])):
            panels.append(("Side", Image.fromarray(render_native("ablation_side"))))
    canvas = Image.new("RGB", (sum(p.width for _, p in panels), 512), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    offset = 0
    for label, panel in panels:
        canvas.paste(panel, (offset, 32))
        draw.text((offset + 8, 9), label, fill="white")
        offset += panel.width
    draw.text((80, 9), f"simulation t={sim_time:.3f}s   {action}", fill="white")
    return np.asarray(canvas)


class NativeReplayCameras:
    def __init__(self, model, data):
        import mujoco
        self.mj, self.model, self.data = mujoco, model, data
        self.renderers = {}

    def render(self, name):
        if name not in self.renderers:
            height, width = (256, 256) if name == "wrist_cam" else (480, 640)
            self.renderers[name] = self.mj.Renderer(self.model, height=height, width=width)
        renderer = self.renderers[name]
        renderer.update_scene(self.data, camera=name)
        return renderer.render().copy()

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()


class MujocoRecorder:
    def __init__(self, task, run_dir, cfg):
        self.task, self.run_dir = task, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir = self.run_dir / "trajectory"
        self.trajectory_dir.mkdir()
        settings = cfg.get("recording", {})
        self.fps = float(settings.get("fps", 30))
        if not 1 <= self.fps <= 60:
            raise ValueError("recording.fps must be between 1 and 60")
        self.camera_config = {key: value for key, value in cfg.items()
                              if key.startswith(("agentview_", "wrist_")) or key == "observer_cameras"}
        self.camera_config["video_layout"] = settings.get("layout", "front")
        if self.camera_config["video_layout"] not in ("front", "multiview"):
            raise ValueError("recording.layout must be front or multiview")
        self.spec = int(task.mj.mjtState.mjSTATE_INTEGRATION)
        self.state_size = task.mj.mj_stateSize(task.model, self.spec)
        self.input_spec = int(task.mj.mjtState.mjSTATE_USER)
        self.input_size = task.mj.mj_stateSize(task.model, self.input_spec)
        self.states, self.controls, self.times = [], [], []
        self.inputs = []
        self.chunks = []
        self.sample_count = 0
        self.action_count = self.request_count = 0
        self.last_request = None
        self.active_action = "INITIAL"
        self.closed = False
        self.video_error = None
        self.secret = ""
        self.video_path = self.run_dir / "motion_live.mp4"
        self.video = StreamingVideoWriter(self.video_path, self.fps)
        self.endpoints = (self.run_dir / "action_endpoints.jsonl").open("w")
        self.requests = (self.run_dir / "request_timing.jsonl").open("w")
        self.frames = (self.run_dir / "video_frames.jsonl").open("w")
        model_raw = self.run_dir / "model.mjb"
        self.model_path = self.run_dir / "model.mjb.gz"
        task.mj.mj_saveModel(task.model, str(model_raw))
        with model_raw.open("rb") as source, gzip.open(self.model_path, "wb", compresslevel=3) as target:
            shutil.copyfileobj(source, target)
        model_raw.unlink()
        self.model_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.start_time = float(task.data.time)
        self.next_frame_time = self.start_time
        self.last_frame_time = None
        self.before_step()
        self._capture_state(task.data.ctrl.copy())
        self._capture_video()
        self.next_frame_time += 1 / self.fps
        task.step_callbacks.append(self.on_step)
        task.before_step_callbacks.append(self.before_step)
        self.flush()

    def pose_record(self):
        task = self.task
        pose = task.ee_pose
        rpy = Rotation.from_quat(pose[3:]).as_euler("xyz")
        return {"sim_time_s": float(task.data.time),
                "eef_pose6d_xyz_rpy": np.r_[pose[:3], rpy].tolist(),
                "eef_pose_xyz_quat_xyzw": pose.tolist(),
                "arm_joint_positions_rad": task.data.qpos[task.arm_qpos].tolist(),
                "arm_joint_velocities_rad_s": task.data.qvel[task.arm_dofs].tolist(),
                "finger_joint_positions_m": task.data.qpos[task.finger_qpos].tolist(),
                "gripper_width_m": task.gripper_width,
                "finger_actuator_forces_n": task.data.actuator_force[task.finger_actuators].tolist()}

    def _capture_state(self, controls):
        state = np.empty(self.state_size)
        self.task.mj.mj_getState(self.task.model, self.task.data, state, self.spec)
        self.states.append(state)
        self.controls.append(np.asarray(controls).copy())
        self.inputs.append(self.pending_inputs.copy())
        self.times.append(float(self.task.data.time))
        self.sample_count += 1

    def before_step(self):
        self.pending_inputs = np.empty(self.input_size)
        self.task.mj.mj_getState(self.task.model, self.task.data, self.pending_inputs, self.input_spec)

    def _capture_video(self):
        when = float(self.task.data.time)
        self.video.append(motion_frame(self.task.render_native, self.camera_config, when, self.active_action))
        self.frames.write(json.dumps({"frame": self.video.frame_count - 1, "sim_time_s": when,
                                     "trajectory_sample": self.sample_count - 1,
                                     "action": self.active_action}) + "\n")
        self.frames.flush()
        self.last_frame_time = when

    def on_step(self, controls):
        self._capture_state(controls)
        if self.task.data.time + 1e-9 >= self.next_frame_time:
            self._capture_video()
            while self.next_frame_time <= self.task.data.time + 1e-9:
                self.next_frame_time += 1 / self.fps
        if len(self.states) >= 2048:
            self.flush()

    def flush(self):
        if self.states:
            name = f"chunk_{len(self.chunks):05d}.npz"
            path = self.trajectory_dir / name
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f, states=np.stack(self.states), controls=np.stack(self.controls),
                                    inputs=np.stack(self.inputs), times=np.asarray(self.times))
            temporary.replace(path)
            self.chunks.append({"file": f"trajectory/{name}", "samples": len(self.states),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            self.states.clear()
            self.controls.clear()
            self.inputs.clear()
            self.times.clear()
        self._write_manifest()

    def _write_manifest(self):
        task = self.task
        write_json(self.run_dir / "trajectory_manifest.json", {
            "format": "show-harness-mujoco-trajectory-v1", "complete": self.closed,
            "mujoco_version": task.mj.__version__, "model": self.model_path.name, "model_sha256": self.model_hash,
            "physics_timestep_s": float(task.model.opt.timestep), "state_spec": self.spec,
            "state_size": self.state_size, "samples": self.sample_count, "chunks": self.chunks,
            "input_spec": self.input_spec, "input_size": self.input_size,
            "start_sim_time_s": self.start_time, "end_sim_time_s": float(task.data.time),
            "control_convention": "Row i controls integrate state i-1 to state i; row 0 controls are unused.",
            "pose_convention": "Robot-base XYZ in metres; extrinsic xyz roll/pitch/yaw in radians; quaternion xyzw.",
            "arm_joint_names": [f"joint{i}" for i in range(1, 8)],
            "finger_joint_names": ["finger_joint1", "finger_joint2"],
            "gripper_control": task.cfg.get("gripper_control", {}),
            "camera_config": self.camera_config, "video_fps": self.fps,
            "video_frames": self.video.frame_count, "video": self.video_path.name,
            "video_error": self.video_error,
            "actions": self.action_count, "http_requests": self.request_count,
            "api_waits_in_video": False,
        })

    def install_controller(self, controller):
        original = controller.step

        def step(token, *args, **kwargs):
            self.active_action = str(token)
            record = {"action_id": self.action_count, "request_id": self.last_request,
                      "token": str(token), "arguments": kwargs,
                      "start_sample": self.sample_count - 1, "before": self.pose_record()}
            self.action_count += 1
            motion_start = len(self.task.motion_records)
            try:
                result = original(token, *args, **kwargs)
                record.update(status="completed", grasp_empty=bool(result.grasp_empty), note=result.note,
                              controller_target_pose_xyz_xyzw=(result.target_pose.tolist()
                                  if result.target_pose is not None else None))
                return result
            except BaseException as exc:
                record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error=self.redact(str(exc)))
                raise
            finally:
                record.update(end_sample=self.sample_count - 1, after=self.pose_record(),
                              motions=self.task.motion_records[motion_start:])
                self.flush()
                self.endpoints.write(json.dumps(record) + "\n")
                self.endpoints.flush()

        controller.step = step

    def install_client(self, client):
        self.secret = str(client._api_key)
        original = client.session.post

        def post(url, *args, **kwargs):
            # A concurrent motion holds this lock until it has settled. Reject a
            # stale payload rather than silently sending an image from before it.
            with self.task.lock:
                observed = self.task.last_observation_time
                if observed is None or abs(observed - self.task.data.time) > 1e-9:
                    raise RuntimeError("Refusing a VLM request with a stale camera observation; capture after movement completion")
                if self.task.motion_in_progress:
                    raise RuntimeError("VLM request attempted before movement completion")
                self.flush()
                self.request_count += 1
                self.last_request = self.request_count
                record = {"request_id": self.request_count, "observation_id": self.task.observation_id,
                          "observation_sim_time_s": observed, "motion_complete": True,
                          "trajectory_sample": self.sample_count - 1}
            started = time.monotonic()
            try:
                response = original(url, *args, **kwargs)
                record["http_status"] = response.status_code
                return response
            except Exception as exc:
                record["error"] = self.redact(str(exc))
                raise
            finally:
                record.update(wall_elapsed_s=time.monotonic() - started,
                              response_sim_time_s=float(self.task.data.time))
                self.requests.write(json.dumps(record) + "\n")
                self.requests.flush()

        client.session.post = post

    def redact(self, text):
        return text.replace(self.secret, "[REDACTED]") if self.secret else text

    def close(self, final_path=None):
        if self.closed:
            return self.video_path
        if self.on_step in self.task.step_callbacks:
            self.task.step_callbacks.remove(self.on_step)
        if self.before_step in self.task.before_step_callbacks:
            self.task.before_step_callbacks.remove(self.before_step)
        try:
            if self.last_frame_time is None or self.task.data.time - self.last_frame_time > 1e-9:
                self._capture_video()
        except Exception as exc:
            self.video_error = self.redact(str(exc))
            raise
        finally:
            try:
                self.video_path = self.video.close(final_path=final_path) or self.video_path
            except Exception as exc:
                self.video_error = self.redact(str(exc))
                raise
            finally:
                self.closed = True
                try:
                    self.flush()
                finally:
                    for stream in (self.endpoints, self.requests, self.frames):
                        stream.close()
        return self.video_path


def trajectory_rows(run_dir, manifest):
    for chunk in manifest["chunks"]:
        path = Path(run_dir) / chunk["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
            raise ValueError(f"Trajectory chunk checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            inputs = data["inputs"] if "inputs" in data else [None] * len(data["times"])
            yield from zip(data["times"], data["states"], data["controls"], inputs)
