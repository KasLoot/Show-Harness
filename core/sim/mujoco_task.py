"""MuJoCo execution backend for the authored RoboLab RubiksCubeTask.

Only RGB and robot proprioception cross the policy interface. Object geometry
is used by the separate evaluator, never to select or gate a model action.
"""
from __future__ import annotations

import json
import copy
from collections import deque
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from core.record.images import prepare_view

ROOT = Path(__file__).resolve().parents[2]


def numbers(value) -> str:
    return " ".join(map(str, np.asarray(value).ravel()))


def _configure_cube(scene, provenance, size_m):
    """Scale visuals and collision together about the authored mesh centroid.

    Mass is preserved. The cube settles onto the table under gravity at reset.
    Cached source meshes/XML are never overwritten.
    """
    if size_m is None:
        return
    size_m = float(size_m)
    if not np.isfinite(size_m) or not 0.01 <= size_m <= 0.10:
        raise ValueError("cube_size_m must be between 0.01 and 0.10 m, or null for the source size")
    obj = provenance["objects"]["rubiks_cube"]
    bounds = np.asarray(obj["local_bounds"])
    center = np.asarray(obj["hull_centroid"])
    scale = size_m / np.max(bounds[1] - bounds[0])
    body = scene.find(".//body[@name='rubiks_cube']")
    assets = scene.find("asset")
    for geom in body.findall("geom"):
        original = assets.find(f"mesh[@name='{geom.get('mesh')}']")
        mesh = copy.deepcopy(original)
        mesh.set("name", original.get("name") + "_runtime_scaled")
        mesh.set("scale", numbers(np.fromstring(original.get("scale", "1 1 1"), sep=" ") * scale))
        assets.append(mesh)
        geom.set("mesh", mesh.get("name"))
        position = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        geom.set("pos", numbers(center + scale * (position - center)))
    obj["source_local_bounds"] = obj["local_bounds"]
    obj["local_bounds"] = (center + scale * (bounds - center)).tolist()
    provenance.setdefault("runtime_overrides", {})["cube"] = {
        "uniform_scale": float(scale), "dimensions_m": (scale * (bounds[1] - bounds[0])).tolist(),
        "mass_kg": obj["authored_mass_kg"], "mass_policy": "unchanged from source",
    }


def build_model_xml(cfg: dict[str, Any]) -> tuple[str, dict]:
    directory = ROOT / cfg["asset_dir"]
    if not (directory / "panda.xml").exists():
        raise FileNotFoundError("MuJoCo assets are not prepared. Run "
                                "scripts/mujoco/download_assets.py, then scripts/mujoco/prepare_assets.py.")
    provenance = json.loads((directory / "provenance.json").read_text())
    root = ET.parse(directory / "panda.xml").getroot()
    scene = ET.parse(directory / "scene.xml").getroot()
    _configure_cube(scene, provenance, cfg.get("cube_size_m"))
    root.find("asset").extend(scene.find("asset"))
    root.find("worldbody").extend(scene.find("worldbody"))
    root.find("option").set("timestep", str(cfg["physics_timestep_s"]))
    root.find("option").set("cone", "elliptic")
    root.find("option").set("iterations", "100")
    grip = cfg.get("gripper_control", {})
    if grip.get("enabled", False):
        force = float(grip.get("close_force_n", 80))
        open_force = float(grip.get("open_force_n", 40))
        if not (0 < force <= 200 and 0 < open_force <= 200):
            raise ValueError("Gripper force targets must be positive and at most 200 N per finger")
        actuators = root.find("actuator")
        for i in (1, 2):
            joint = root.find(f".//joint[@name='finger_joint{i}']")
            joint.set("solreflimit", "0.004 1")
            joint.set("solimplimit", "0.99 0.999 0.0001")
            old = actuators.find(f"*[@name='finger_actuator{i}']")
            actuators.remove(old)
            limit = max(force, open_force)
            # MuJoCo integrates actuator velocity damping implicitly. Computing
            # that damping as a Python motor force would make it explicit and
            # can cause chatter at a stiff contact with the 2 ms timestep.
            ET.SubElement(actuators, "general", name=f"finger_actuator{i}", joint=f"finger_joint{i}",
                          gear="1", dyntype="none", gaintype="fixed", gainprm="1",
                          biastype="affine", biasprm=f"0 0 {-float(grip.get('kd', 100))}",
                          ctrlrange=f"{-limit} {limit}", forcerange=f"{-limit} {limit}")
        provenance.setdefault("runtime_overrides", {})["gripper_control"] = dict(grip)
    camera = provenance["franka"]
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.5 0.5 0.5")
    ET.SubElement(visual, "map", znear="0.001")
    world = root.find("worldbody")
    ET.SubElement(world, "light", name="fill", pos="1 -1 2", dir="-0.5 0 -1",
                  diffuse="0.4 0.4 0.4", castshadow="false")
    rotation = np.asarray(camera["FRONT_CAM_R"]) @ np.diag([1, -1, -1])
    fx, _, cx, _, fy, cy, *_ = camera["FRONT_CAM_K"]
    width, height = camera["FRONT_CAM_W"], camera["FRONT_CAM_H"]
    ET.SubElement(world, "camera", name="front_cam", pos=numbers(camera["FRONT_CAM_POS"]),
                  xyaxes=numbers(rotation[:, :2].T), resolution=f"{width} {height}",
                  sensorsize="0.0064 0.0048", focalpixel=numbers([fx, fy]),
                  principalpixel=numbers([width/2 - cx, height/2 - cy]))
    hand = root.find(".//body[@name='hand']")
    # Original camera: ROS identity relative to panda_hand. OpenGL looks along -Z.
    ET.SubElement(hand, "camera", name="wrist_cam", pos=numbers(camera["WRIST_CAM_POS"]),
                  quat="0 1 0 0", fovy="90")
    # Optional passive observers for controlled camera experiments. They do not
    # add bodies, geoms, forces or controller logic to the simulation.
    for observer in cfg.get("observer_cameras", []):
        ET.SubElement(world, "camera", name=observer["name"],
                      pos=numbers(observer["position"]), xyaxes=numbers(observer["xyaxes"]),
                      fovy=str(observer["fovy"]))
    return ET.tostring(root, encoding="unicode"), provenance


class MujocoTask:
    def __init__(self, cfg: dict[str, Any], *, gui: bool = False):
        import mujoco

        self.mj, self.cfg = mujoco, cfg
        xml, self.provenance = build_model_xml(cfg)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.eef_id = self.model.site("eef").id
        self.arm_qpos = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dofs = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_actuators = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.limits = np.array([self.model.joint(f"joint{i}").range for i in range(1, 8)])
        self.finger_qpos = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.finger_dofs = [self.model.joint(f"finger_joint{i}").dofadr[0] for i in (1, 2)]
        self.finger_actuators = [self.model.actuator(f"finger_actuator{i}").id for i in (1, 2)]
        self.renderer = self.wrist_renderer = self.viewer = None
        self.observer_renderers = {}
        self.lock = threading.RLock()
        self.motion_in_progress = False
        self.last_observation_time = None
        self.observation_id = 0
        self.motion_records = []
        self.step_callbacks = []
        self.before_step_callbacks = []
        self._pacing = None
        self.gripper_closed_command = False
        self.gripper_reference = np.full(2, 0.04)
        self.arm_hold_bias = np.zeros(7)
        self.gripper_control = cfg.get("gripper_control", {})
        home = self.provenance["franka"]["FRANKA_HOME_QPOS"]
        self.data.qpos[self.arm_qpos] = [home[f"panda_joint{i}"] for i in range(1, 8)]
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.arm_actuators] = self.data.qpos[self.arm_qpos]
        self.data.ctrl[self.finger_actuators] = 0 if self.gripper_control.get("enabled") else 0.04
        self.mj.mj_forward(self.model, self.data)
        self.advance(float(cfg["initial_settle_s"]))
        self.arm_goal_pose = self.ee_pose.copy()
        if gui:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.35, 0.02, 0.15]
            self.viewer.cam.distance = 1.5
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
            self.viewer.sync()

    @property
    def ee_pose(self):
        rotation = Rotation.from_matrix(self.data.site_xmat[self.eef_id].reshape(3, 3))
        return np.r_[self.data.site_xpos[self.eef_id], rotation.as_quat()]

    @property
    def gripper_width(self):
        return max(0.0, float(np.sum(self.data.qpos[self.finger_qpos])))

    def advance(self, seconds):
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self._update_gripper_control()
            for callback in tuple(self.before_step_callbacks):
                callback()
            controls = self.data.ctrl.copy()
            self.mj.mj_step(self.model, self.data)
            # mj_step integrates qpos after computing kinematics. Refresh before
            # feedback, pose logging, or images so all refer to this exact state.
            self.mj.mj_forward(self.model, self.data)
            for callback in tuple(self.step_callbacks):
                callback(controls)
            if self.viewer is not None and round(self.data.time / self.model.opt.timestep) % 20 == 0:
                self.viewer.sync()
            if self._pacing is not None:
                wall, sim = self._pacing
                remaining = wall + (self.data.time - sim) / float(self.cfg.get("gui_speed", 1.0)) - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("MuJoCo produced a non-finite state")

    def _update_gripper_control(self):
        grip = self.gripper_control
        if not grip.get("enabled", False):
            return
        kp = float(grip.get("kp", 6000))
        closing = float(grip.get("close_force_n", 80))
        opening = float(grip.get("open_force_n", 40))
        goal = -closing / kp if self.gripper_closed_command else 0.04
        increment = float(grip.get("speed_m_s", 0.04)) * self.model.opt.timestep
        self.gripper_reference += np.clip(goal - self.gripper_reference, -increment, increment)
        positions = self.data.qpos[self.finger_qpos]
        reference = self.gripper_reference.copy()
        if self.gripper_closed_command:
            # At the mechanical closed stop, remove preload rather than driving
            # the empty fingers through their travel limits. Object grasps still
            # receive the configured force. Only joint encoders are used here.
            reference = np.where(positions < 0.001, np.maximum(reference, 0.0), reference)
        force = kp * (reference - positions)
        self.data.ctrl[self.finger_actuators] = np.clip(force, -closing, opening)

    @contextmanager
    def _movement(self):
        with self.lock:
            previous_busy, previous_pacing = self.motion_in_progress, self._pacing
            self.motion_in_progress = True
            if not previous_busy:
                self._pacing = (time.monotonic(), self.data.time) if self.viewer is not None else None
            try:
                yield
            finally:
                self.motion_in_progress = previous_busy
                self._pacing = previous_pacing

    def pose_error(self, target):
        actual = self.ee_pose
        return (float(np.linalg.norm(actual[:3] - target[:3])),
                float((Rotation.from_quat(target[3:]) * Rotation.from_quat(actual[3:]).inv()).magnitude()))

    def bind_controller(self, controller):
        """Keep a whole atomic action (including an empty-close reopen) atomic."""
        original = controller.step

        def step(token, *args, **kwargs):
            with self._movement():
                first = len(self.motion_records)
                try:
                    result = original(token, *args, **kwargs)
                except BaseException:
                    controller.sync_from_robot()
                    raise
                if any(m.get("workspace_clipped") for m in self.motion_records[first:]):
                    controller.sync_from_robot()
                    result.target_pose = controller.target_pose
                    result.note = (result.note + "; workspace target clipped; setpoint resynchronized").strip("; ")
                return result

        controller.step = step

    def solve_ik(self, pose):
        work = self.ik_data
        work.qpos[:] = self.data.qpos
        goal_rotation = Rotation.from_quat(pose[3:]).as_matrix()
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(100):
            self.mj.mj_forward(self.model, work)
            rotation = work.site_xmat[self.eef_id].reshape(3, 3)
            error = np.r_[pose[:3] - work.site_xpos[self.eef_id],
                          Rotation.from_matrix(goal_rotation @ rotation.T).as_rotvec()]
            if np.linalg.norm(error[:3]) < 0.0001 and np.linalg.norm(error[3:]) < 0.001:
                return work.qpos[self.arm_qpos].copy()
            self.mj.mj_jacSite(self.model, work, jacp, jacr, self.eef_id)
            jac = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            work.qpos[self.arm_qpos] = np.clip(
                work.qpos[self.arm_qpos] + np.clip(delta, -0.12, 0.12),
                self.limits[:, 0] + 0.001, self.limits[:, 1] - 0.001)
        raise RuntimeError(f"MuJoCo IK cannot reach {np.round(pose[:3], 4).tolist()}")

    def command_pose(self, pose):
        requested = np.asarray(pose, dtype=float).copy()
        target = requested.copy()
        target[:3] = np.clip(target[:3], self.cfg["workspace_min"], self.cfg["workspace_max"])
        settings = self.cfg.get("motion_control", {})
        if not settings.get("enabled", False):
            self.data.ctrl[self.arm_actuators] = self.solve_ik(target)
            self.advance(float(self.cfg["control_tick_s"]))
            return
        record = {"kind": "pose", "requested_pose_xyz_xyzw": requested.tolist(),
                  "applied_pose_xyz_xyzw": target.tolist(), "start_time_s": float(self.data.time),
                  "workspace_clipped": bool(np.any(requested[:3] != target[:3])), "status": "running"}
        with self._movement():
            try:
                start = self.data.qpos[self.arm_qpos].copy()
                goal = self.solve_ik(target)
                self.arm_goal_pose = target.copy()
                distance, angle = self.pose_error(target)
                duration = max(float(settings.get("min_duration_s", 0.25)),
                               1.875 * distance / float(settings.get("linear_speed_m_s", 0.10)),
                               1.875 * angle / float(settings.get("angular_speed_rad_s", 0.5)),
                               1.875 * np.max(abs(goal - start)) / float(settings.get("joint_speed_rad_s", 1.0)))
                count = max(1, int(np.ceil(duration / self.model.opt.timestep)))
                # Smooth bounded joint references drive position servos. Live
                # qpos is only changed by mj_step, never by this trajectory.
                for i in range(1, count + 1):
                    u = i / count
                    blend = 10*u**3 - 15*u**4 + 6*u**5
                    self.data.ctrl[self.arm_actuators] = np.clip(
                        start + blend * (goal - start) + self.arm_hold_bias,
                        self.limits[:, 0], self.limits[:, 1])
                    self.advance(self.model.opt.timestep)
                self.data.ctrl[self.arm_actuators] = np.clip(goal + self.arm_hold_bias,
                                                           self.limits[:, 0], self.limits[:, 1])
                stable = 0.0
                deadline = self.data.time + float(settings.get("settle_timeout_s", 3.0))
                while self.data.time < deadline:
                    # Encoder feedback removes steady payload sag. This is robot
                    # joint feedback, not object-state/target-coordinate control.
                    self.arm_hold_bias = np.clip(
                        self.arm_hold_bias + float(settings.get("integral_gain_s", 2.0))
                        * (goal - self.data.qpos[self.arm_qpos]) * self.model.opt.timestep,
                        -0.05, 0.05)
                    self.data.ctrl[self.arm_actuators] = np.clip(goal + self.arm_hold_bias,
                                                               self.limits[:, 0], self.limits[:, 1])
                    self.advance(self.model.opt.timestep)
                    pe, re = self.pose_error(target)
                    speed = np.max(abs(self.data.qvel[self.arm_dofs]))
                    reached = (pe <= float(settings.get("position_tolerance_m", 0.0005))
                               and re <= float(settings.get("rotation_tolerance_rad", 0.005))
                               and speed <= float(settings.get("velocity_tolerance_rad_s", 0.02)))
                    stable = stable + self.model.opt.timestep if reached else 0.0
                    if stable >= float(settings.get("settle_duration_s", 0.1)):
                        break
                else:
                    raise RuntimeError(f"MuJoCo motion did not settle: position error {pe:.4f} m, rotation error {re:.4f} rad")
                record.update(status="reached", position_error_m=pe, rotation_error_rad=re,
                              hold_bias_rad=self.arm_hold_bias.tolist())
            except BaseException as exc:
                record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                raise
            finally:
                record["end_time_s"] = float(self.data.time)
                self.motion_records.append(record)

    def command_gripper(self, close):
        with self._movement():
            started = self.data.time
            record = {"kind": "gripper", "close": bool(close), "start_time_s": float(started), "status": "running"}
            try:
                # Begin a new opening/closing ramp at the measured position.
                # In particular, release must discard the closing preload now,
                # rather than ramping it back through the closed travel stop.
                self.gripper_reference = np.clip(self.data.qpos[self.finger_qpos], 0.0, 0.04).copy()
                self.gripper_closed_command = bool(close)
                if not self.gripper_control.get("enabled", False):
                    self.data.ctrl[self.finger_actuators] = 0.0 if close else 0.04
                    self.advance(float(self.cfg["gripper_duration_s"]))
                else:
                    stable = 0.0
                    widths = deque()
                    timeout = float(self.gripper_control.get("timeout_s", 3.0))
                    while self.data.time - started < timeout:
                        self.advance(self.model.opt.timestep)
                        speed = np.max(abs(self.data.qvel[self.finger_dofs]))
                        widths.append((float(self.data.time), self.gripper_width))
                        while len(widths) > 1 and widths[0][0] < self.data.time - 0.1:
                            widths.popleft()
                        if close:
                            ready = (np.all(self.data.actuator_force[self.finger_actuators] <=
                                            -0.95 * float(self.gripper_control.get("close_force_n", 80)))
                                     or np.all(self.data.qpos[self.finger_qpos] < 0.0005))
                            # Contact can move the two fingers together slightly
                            # while their separation is stable. A force-controlled
                            # grasp is complete when the jaw width has settled.
                            span = max(w for _, w in widths) - min(w for _, w in widths)
                            quiet = (self.data.time - widths[0][0] >= 0.095
                                     and span < float(self.gripper_control.get("width_tolerance_m", 0.0002)))
                        else:
                            ready = np.all(abs(self.data.qpos[self.finger_qpos] - 0.04) < 0.0005)
                            quiet = speed < float(self.gripper_control.get("velocity_tolerance_m_s", 0.003))
                        settled = ready and quiet and self.data.time - started >= float(self.cfg["gripper_duration_s"])
                        stable = stable + self.model.opt.timestep if settled else 0.0
                        if stable >= 0.1:
                            break
                    else:
                        raise RuntimeError("MuJoCo gripper did not settle before timeout")
                if self.cfg.get("motion_control", {}).get("enabled", False):
                    self.command_pose(self.arm_goal_pose)
                record.update(status="settled", width_m=self.gripper_width,
                              actuator_force_n=self.data.actuator_force[self.finger_actuators].tolist())
            except BaseException as exc:
                record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                raise
            finally:
                record["end_time_s"] = float(self.data.time)
                self.motion_records.append(record)

    def render_native(self, camera):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=480, width=640)
            self.wrist_renderer = self.mj.Renderer(self.model, height=256, width=256)
        if camera == "front_cam":
            renderer = self.renderer
        elif camera == "wrist_cam":
            renderer = self.wrist_renderer
        else:
            if camera not in self.observer_renderers:
                self.observer_renderers[camera] = self.mj.Renderer(self.model, height=480, width=640)
            renderer = self.observer_renderers[camera]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def render(self):
        views = []
        for prefix, camera in (("agentview", "front_cam"), ("wrist", "wrist_cam")):
            views.append(prepare_view(self.render_native(camera), **{
                key: self.cfg.get(f"{prefix}_{key}")
                for key in ("rotation_degrees", "flip", "crop_aspect", "square_size")}))
        return tuple(views)

    def close(self):
        for resource in (self.viewer, self.renderer, self.wrist_renderer,
                         *self.observer_renderers.values()):
            if resource is not None:
                resource.close()

    def render_observer(self, name):
        """An unannotated fixed camera; called identically in both ablation arms."""
        return prepare_view(self.render_native(name), square_size=256)


class MujocoRobot:
    """The actuator/encoder interface used by FrankaAtomicController on hardware."""
    def __init__(self, task):
        self._task = task

    def get_ee_pose(self):
        return self._task.ee_pose.copy()

    def get_gripper_position(self):
        return [self._task.gripper_width]

    def update_desired_ee_pose(self, pose):
        self._task.command_pose(pose)

    def control_gripper(self, close):
        self._task.command_gripper(close)


class MujocoSession:
    def __init__(self, task):
        from types import SimpleNamespace

        self._task = task
        self.robot = MujocoRobot(task)
        self.config = SimpleNamespace(start_impedance=False)

    def get_observation(self):
        with self._task.lock:
            front, wrist = self._task.render()
            self._task.last_observation_time = float(self._task.data.time)
            self._task.observation_id += 1
            return {"agentview": front, "wrist": wrist, "ee_pose": self.robot.get_ee_pose(),
                    "gripper_width": self.robot.get_gripper_position()[0]}


def evaluate_task(task: MujocoTask) -> dict:
    """Independent RoboLab object-in-container/contact/detachment scoring.

    Called only after the VLM episode ends, never by the runner or action interpreter.
    """
    bounds = task.provenance["objects"]
    cube_local = np.asarray(bounds["rubiks_cube"]["hull_centroid"])
    cube = task.data.body("rubiks_cube")
    bowl = task.data.body("bowl")
    center = cube.xpos + cube.xmat.reshape(3, 3) @ cube_local
    in_bowl_frame = bowl.xmat.reshape(3, 3).T @ (center - bowl.xpos)
    planes = np.asarray(bounds["bowl"]["open_top_planes"])
    inside = bool(np.all(planes[:, :3] @ in_bowl_frame + planes[:, 3] <= 0))
    cube_id, bowl_id = task.model.body("rubiks_cube").id, task.model.body("bowl").id
    finger_ids = {task.model.body("left_finger").id, task.model.body("right_finger").id}
    forces = {body: np.zeros(3) for body in {bowl_id, *finger_ids}}
    for contact_id, contact in enumerate(task.data.contact):
        body1, body2 = (int(task.model.geom_bodyid[contact.geom1]),
                        int(task.model.geom_bodyid[contact.geom2]))
        if cube_id in (body1, body2):
            other = body1 if body2 == cube_id else body2
            if other not in forces:
                continue
            force = np.zeros(6)
            task.mj.mj_contactForce(task.model, task.data, contact_id, force)
            world_force = contact.frame.reshape(3, 3).T @ force[:3]
            forces[other] += world_force if body2 == cube_id else -world_force
    touching_bowl = bool(np.linalg.norm(forces[bowl_id]) > 0.1)
    touching_gripper = any(np.linalg.norm(forces[body]) > 0.1 for body in finger_ids)
    return {"success": bool(inside and touching_bowl and not touching_gripper),
            "object_in_container": inside, "contact_with_bowl": touching_bowl,
            "gripper_detached": not touching_gripper,
            "cube_center_world": center.tolist(), "cube_center_in_bowl": in_bowl_frame.tolist()}
