"""MuJoCo Panda session implementing the existing Cartesian robot/session contract."""
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from plugins.smooth import SmoothPlugin


ROOT = Path(__file__).resolve().parents[2]
PANDA = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"


class MujocoSession:
    control_mode = "mujoco"

    def __init__(self, cfg: dict, *, gui: bool = False) -> None:
        if not PANDA.is_file():
            raise FileNotFoundError("Panda assets missing. Run: bash scripts/setup.sh mujoco")
        panda = ET.parse(PANDA).getroot()
        panda.find("compiler").set("meshdir", str(PANDA.parent / "assets"))
        panda.remove(panda.find("keyframe"))
        scene = ET.parse(ROOT / "assets/mujoco/pick_place.xml").getroot()
        for element in scene:
            existing = panda.find(element.tag)
            if existing is None:
                panda.append(element)
            else:
                existing.extend(element)
        hand = panda.find(".//body[@name='hand']")
        ET.SubElement(hand, "site", name="tcp", pos="0 0 0.103", size="0.003", rgba="0 0 0 0")
        # Center the optical axis on the TCP so "between the fingers" is a grasp.
        ET.SubElement(hand, "camera", name="wrist", pos="0 0 0.055",
                      xyaxes="0 -1 0 -1 0 0", fovy="80")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(panda, encoding="unicode"))
        table = self.model.geom("table")
        self.table_height_m = float(table.pos[2] + table.size[2])
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        self.robot = self
        self.recorder = None
        self._renderer = None
        self._viewer = None
        self.resolution = int(cfg.get("camera_resolution", 512))
        self.motion_s = float(cfg.get("motion_s", 0.35))
        # Set by the variable-step controller to preserve speed on longer translations.
        self.motion_reference_m = None
        self.settle_s = float(cfg.get("settle_s", 0.15))
        self.ik_damping = float(cfg.get("ik_damping", 0.03))
        durations = (self.motion_s, self.settle_s, self.ik_damping)
        if not 16 <= self.resolution <= 640 or not np.isfinite(durations).all() or min(durations) <= 0:
            raise ValueError("Require camera_resolution in [16, 640] and finite positive motion_s, settle_s, ik_damping")
        joints = [self.model.joint(f"joint{i}") for i in range(1, 8)]
        self._qpos = np.array([int(j.qposadr[0]) for j in joints])
        self._dofs = np.array([int(j.dofadr[0]) for j in joints])
        self._limits = np.array([j.range for j in joints])
        self._actuators = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._tcp = self.model.site("tcp").id
        self._gripper = self.model.actuator("actuator8").id
        self._jac_pos = np.zeros((3, self.model.nv))
        self._jac_rot = np.zeros((3, self.model.nv))
        home = np.array(cfg["reset_qpos"], dtype=float)
        if home.shape != (7,) or not np.isfinite(home).all():
            raise ValueError("reset_qpos must contain seven finite joint angles")
        if np.any(home < self._limits[:, 0]) or np.any(home > self._limits[:, 1]):
            raise ValueError("reset_qpos exceeds Panda joint limits")
        self.data.qpos[self._qpos] = home
        self.data.ctrl[self._actuators] = home
        for name in ("finger_joint1", "finger_joint2"):
            self.data.joint(name).qpos[0] = 0.04
        self.data.ctrl[self._gripper] = 255
        mujoco.mj_forward(self.model, self.data)
        self._advance(1.0)
        if gui:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.cam.lookat[:] = [0.4, 0, 0.2]
            self._viewer.cam.distance = 1.5
            self._viewer.cam.azimuth = 0
            self._viewer.cam.elevation = -35
            self._viewer.sync()

    def _advance(self, seconds: float, target_ctrl: np.ndarray | None = None) -> None:
        # Decision-synchronous physics: cloud latency does not change simulated time.
        dt = self.model.opt.timestep
        count = max(2 if target_ctrl is not None else 1, int(np.ceil(seconds / dt)))
        start_ctrl = self.data.ctrl.copy()
        fractions = (SmoothPlugin(enabled=True, substeps=count, dt_s=dt).fractions()
                     if target_ctrl is not None else None)
        sim_started = float(self.data.time)
        wall_started = time.monotonic() if self._viewer is not None else 0.0
        viewer_stride = max(1, round(1 / (60 * dt)))
        for i in range(count):
            if self._viewer is not None and not self._viewer.is_running():
                raise KeyboardInterrupt
            if fractions is not None:
                self.data.ctrl[:] = start_ctrl + fractions[i] * (target_ctrl - start_ctrl)
            self.data.qfrc_applied[self._dofs] = self.data.qfrc_bias[self._dofs]
            mujoco.mj_step(self.model, self.data)
            if self.recorder is not None:
                self.recorder.capture()
            if self._viewer is not None and ((i + 1) % viewer_stride == 0 or i + 1 == count):
                delay = float(self.data.time) - sim_started - (time.monotonic() - wall_started)
                if delay > 0:
                    time.sleep(delay)
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_camlight(self.model, self.data)
                self._viewer.sync()
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("MuJoCo state became non-finite")

    def get_ee_pose(self) -> np.ndarray:
        site = self.data.site(self._tcp)
        quat = Rotation.from_matrix(site.xmat.reshape(3, 3)).as_quat()
        return np.concatenate([site.xpos, quat])

    def get_gripper_position(self) -> np.ndarray:
        return np.array([sum(float(self.data.joint(n).qpos[0])
                             for n in ("finger_joint1", "finger_joint2"))])

    def control_gripper(self, close: bool) -> None:
        if self.recorder is not None:
            self.recorder.gripper_command(close)
        target = self.data.ctrl.copy()
        target[self._gripper] = 0 if close else 255
        self._advance(self.motion_s, target_ctrl=target)
        self._advance(self.settle_s)

    def update_desired_ee_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite [x, y, z, qx, qy, qz, qw] pose")
        target_rotation = Rotation.from_quat(pose[3:]).as_matrix()
        self._ik.qpos[:] = self.data.qpos
        # Damped Jacobian IK uses a scratch state; the live arm moves via actuators.
        for _ in range(100):
            mujoco.mj_kinematics(self.model, self._ik)
            mujoco.mj_comPos(self.model, self._ik)
            site = self._ik.site(self._tcp)
            error = np.concatenate([
                pose[:3] - site.xpos,
                Rotation.from_matrix(target_rotation @ site.xmat.reshape(3, 3).T).as_rotvec(),
            ])
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, self._ik, self._jac_pos, self._jac_rot, self._tcp)
            jac = np.vstack([self._jac_pos[:, self._dofs], self._jac_rot[:, self._dofs]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + self.ik_damping**2 * np.eye(6), error)
            self._ik.qpos[self._qpos] = np.clip(
                self._ik.qpos[self._qpos] + np.clip(delta, -0.1, 0.1),
                self._limits[:, 0], self._limits[:, 1],
            )
        else:
            raise RuntimeError(f"Panda IK could not reach target {pose[:3].round(4).tolist()}")
        target = self.data.ctrl.copy()
        target[self._actuators] = self._ik.qpos[self._qpos]
        duration = self.motion_s
        if self.motion_reference_m is not None:
            distance = float(np.linalg.norm(pose[:3] - self.data.site(self._tcp).xpos))
            duration *= max(1.0, distance / self.motion_reference_m)
        self._advance(duration, target_ctrl=target)
        self._advance(self.settle_s)

    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.resolution, width=self.resolution)
        # Refresh geometry without running the force solver or changing its warm start.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def get_observation(self) -> dict:
        frames = {camera: self.render(camera) for camera in ("side", "wrist", "front")}
        # The shared single-arm runner calls its primary external input "agentview".
        # It now contains Side; the removed AgentView camera is never rendered.
        return {**frames, "agentview": frames["side"], "primary_camera": "Side",
                "extra_views": {"front": frames["front"]}, "ee_pose": self.get_ee_pose(),
                "gripper_width": float(self.get_gripper_position()[0])}

    def check_success(self) -> bool:
        cube = self.data.body("cube").xpos
        target = self.data.body("target").xpos
        extents = (np.abs(self.data.geom("cube").xmat.reshape(3, 3))
                   @ self.model.geom("cube").size)
        pad = self.model.geom("target").size
        # The whole 4 cm cube must rest on the 12 cm pad, with the fingers open.
        return bool(np.all(np.abs(cube[:2] - target[:2]) + extents[:2] < pad[:2])
                    and abs(cube[2] - extents[2] - target[2] - pad[2]) < 0.006
                    and self.get_gripper_position()[0] > 0.06
                    and self.get_ee_pose()[2] > cube[2] + 0.06
                    and np.linalg.norm(self.data.joint("cube_joint").qvel) < 0.05)

    def close(self) -> None:
        try:
            if self.recorder is not None:
                self.recorder.close()
        finally:
            try:
                if self._renderer is not None:
                    self._renderer.close()
            finally:
                if self._viewer is not None:
                    self._viewer.close()
