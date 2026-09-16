"""MuJoCo execution backend for the authored RoboLab RubiksCubeTask.

Only RGB and robot proprioception cross the policy interface. Object geometry
is used by the separate evaluator, never to select or gate a model action.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from core.record.images import prepare_view

ROOT = Path(__file__).resolve().parents[2]


def numbers(value) -> str:
    return " ".join(map(str, np.asarray(value).ravel()))


def build_model_xml(cfg: dict[str, Any]) -> tuple[str, dict]:
    directory = ROOT / cfg["asset_dir"]
    if not (directory / "panda.xml").exists():
        raise FileNotFoundError("MuJoCo assets are not prepared. Run "
                                "scripts/mujoco/download_assets.py, then scripts/mujoco/prepare_assets.py.")
    provenance = json.loads((directory / "provenance.json").read_text())
    root = ET.parse(directory / "panda.xml").getroot()
    scene = ET.parse(directory / "scene.xml").getroot()
    root.find("asset").extend(scene.find("asset"))
    root.find("worldbody").extend(scene.find("worldbody"))
    root.find("option").set("timestep", str(cfg["physics_timestep_s"]))
    root.find("option").set("cone", "elliptic")
    root.find("option").set("iterations", "100")
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
        self.finger_actuators = [self.model.actuator(f"finger_actuator{i}").id for i in (1, 2)]
        self.renderer = self.wrist_renderer = self.viewer = None
        self.observer_renderers = {}
        home = self.provenance["franka"]["FRANKA_HOME_QPOS"]
        self.data.qpos[self.arm_qpos] = [home[f"panda_joint{i}"] for i in range(1, 8)]
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.arm_actuators] = self.data.qpos[self.arm_qpos]
        self.data.ctrl[self.finger_actuators] = 0.04
        self.mj.mj_forward(self.model, self.data)
        self.advance(float(cfg["initial_settle_s"]))
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
        return float(np.sum(self.data.qpos[self.finger_qpos]))

    def advance(self, seconds):
        for i in range(max(1, round(seconds / self.model.opt.timestep))):
            self.mj.mj_step(self.model, self.data)
            if self.viewer is not None and i % 20 == 0:
                self.viewer.sync()
        self.mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("MuJoCo produced a non-finite state")

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
        pose = np.asarray(pose, dtype=float).copy()
        pose[:3] = np.clip(pose[:3], self.cfg["workspace_min"], self.cfg["workspace_max"])
        self.data.ctrl[self.arm_actuators] = self.solve_ik(pose)
        self.advance(float(self.cfg["control_tick_s"]))

    def command_gripper(self, close):
        self.data.ctrl[self.finger_actuators] = 0.0 if close else 0.04
        self.advance(float(self.cfg["gripper_duration_s"]))

    def render(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=480, width=640)
            self.wrist_renderer = self.mj.Renderer(self.model, height=256, width=256)
        views = []
        for prefix, camera, renderer in (("agentview", "front_cam", self.renderer),
                                         ("wrist", "wrist_cam", self.wrist_renderer)):
            renderer.update_scene(self.data, camera=camera)
            views.append(prepare_view(renderer.render().copy(), **{
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
        if name not in self.observer_renderers:
            self.observer_renderers[name] = self.mj.Renderer(self.model, height=480, width=640)
        renderer = self.observer_renderers[name]
        renderer.update_scene(self.data, camera=name)
        return prepare_view(renderer.render().copy(), square_size=256)


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
        front, wrist = self._task.render()
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
