"""MuJoCo Panda session implementing the existing Cartesian robot/session contract."""
from pathlib import Path
from collections.abc import Mapping
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from core.sim.mujoco_socket import add_round_socket
from core.sim.mujoco_success import plug_retreat_clearance_m
from core.sim.mujoco_depth import (
    depth_to_grayscale, depth_to_text, validate_depth_grid, validate_depth_range,
)
from plugins.smooth import SmoothPlugin


ROOT = Path(__file__).resolve().parents[2]
PANDA = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"


class MujocoSession:
    control_mode = "mujoco"

    def __init__(self, cfg: dict, *, gui: bool = False) -> None:
        self.scene = cfg.get("scene", "pick_place")
        if self.scene not in ("pick_place", "plug_insert"):
            raise ValueError("scene must be 'pick_place' or 'plug_insert'")
        self.plug_retreat_clearance_m = plug_retreat_clearance_m(cfg)
        if not PANDA.is_file():
            raise FileNotFoundError("Panda assets missing. Run: bash scripts/setup.sh mujoco")
        panda = ET.parse(PANDA).getroot()
        panda.find("compiler").set("meshdir", str(PANDA.parent / "assets"))
        panda.remove(panda.find("keyframe"))
        # Balance the Panda's local task spotlight with the lab's broad fill.
        panda.find("./worldbody/light[@name='top']").set("diffuse", "0.35 0.35 0.35")
        scene = ET.parse(ROOT / "assets/mujoco" / f"{self.scene}.xml").getroot()
        if self.scene == "plug_insert":
            add_round_socket(scene)
        # Shared lab furnishings are visual-only; task surfaces and contacts stay
        # in the individual scene. Append them after task geometry for stable IDs.
        lab = ET.parse(ROOT / "assets/mujoco/lab.xml").getroot()
        for source in (scene, lab):
            for element in source:
                existing = panda.find(element.tag)
                if existing is None:
                    panda.append(element)
                else:
                    existing.extend(element)
        # Camera clipping scales with model extent. Keep it tied to the robot
        # workspace so adding a room cannot clip nearby wrist-camera objects.
        ET.SubElement(panda, "statistic", center="0.35 0 0.17", extent="2")
        hand = panda.find(".//body[@name='hand']")
        ET.SubElement(hand, "site", name="tcp", pos="0 0 0.103", size="0.003", rgba="0 0 0 0")
        # Center the optical axis on the TCP so "between the fingers" is a grasp.
        ET.SubElement(hand, "camera", name="wrist", pos="0 0 0.055",
                      xyaxes="0 -1 0 -1 0 0", fovy="80")
        self.cameras = ("side", "wrist", "front")
        if self.scene == "plug_insert":
            # Rigid hand mount: oblique view of the shaft and socket rim below TCP.
            ET.SubElement(hand, "camera", name="wrist_insert", pos="0.1 0 0.045",
                          xyaxes="0 -1 0 -0.7488700551657237 0 -0.6627168629785166",
                          fovy="65")
            self.cameras += ("wrist_insert", "wrist_depth")
        depth_cfg = cfg.get("wrist_depth") or {}
        if not isinstance(depth_cfg, Mapping):
            raise ValueError("wrist_depth must be a mapping")
        self.depth_representation = depth_cfg.get("representation", "grayscale")
        if self.depth_representation not in ("grayscale", "text"):
            raise ValueError("wrist_depth.representation must be 'grayscale' or 'text'")
        self.depth_grid_rows = depth_cfg.get("grid_rows", 32)
        self.depth_grid_cols = depth_cfg.get("grid_cols", 32)
        validate_depth_grid(self.depth_grid_rows, self.depth_grid_cols)
        self.depth_near_m = float(depth_cfg.get("near_m", 0.0))
        self.depth_far_m = float(depth_cfg.get("far_m", 0.30))
        validate_depth_range(self.depth_near_m, self.depth_far_m)
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
        # The opt-in Cartesian controller also scales duration for larger turns.
        self.motion_reference_rad = None
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
            # Interactive overview includes the bench frame and lab; the named
            # cameras supplied to the VLM retain their task-focused calibration.
            self._viewer.cam.lookat[:] = [0.3, 0, 0.05]
            self._viewer.cam.distance = 2.6
            self._viewer.cam.azimuth = 35
            self._viewer.cam.elevation = -25
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
        duration_scale = 1.0
        if self.motion_reference_m is not None:
            distance = float(np.linalg.norm(pose[:3] - self.data.site(self._tcp).xpos))
            duration_scale = max(duration_scale, distance / self.motion_reference_m)
        if self.motion_reference_rad is not None:
            current_rotation = self.data.site(self._tcp).xmat.reshape(3, 3)
            angle = float(Rotation.from_matrix(target_rotation @ current_rotation.T).magnitude())
            duration_scale = max(duration_scale, angle / self.motion_reference_rad)
        duration *= duration_scale
        self._advance(duration, target_ctrl=target)
        self._advance(self.settle_s)

    def render(self, camera: str) -> np.ndarray:
        if camera == "wrist_depth":
            return depth_to_grayscale(self.render_depth("wrist"),
                                      near_m=self.depth_near_m, far_m=self.depth_far_m)
        self._prepare_render(camera)
        return self._renderer.render().copy()

    def _prepare_render(self, camera: str) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.resolution, width=self.resolution)
        # Refresh geometry without running the force solver or changing its warm start.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self._renderer.update_scene(self.data, camera=camera)

    def render_depth(self, camera: str = "wrist") -> np.ndarray:
        """Metric optical-axis depth from the same projection as the RGB camera."""
        self._prepare_render(camera)
        self._renderer.enable_depth_rendering()
        try:
            return self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()

    def get_observation(self) -> dict:
        frames = {camera: self.render(camera) for camera in self.cameras if camera != "wrist_depth"}
        raw_depth = None
        if "wrist_depth" in self.cameras:
            # A single raw capture supplies text, saved metric data, and the local
            # grayscale diagnostic, so they cannot represent different snapshots.
            raw_depth = self.render_depth("wrist").copy()
            raw_depth.setflags(write=False)
            frames["wrist_depth"] = depth_to_grayscale(
                raw_depth, near_m=self.depth_near_m, far_m=self.depth_far_m,
            )
        # Plug appends Angled Wrist and registered Wrist Depth after the three RGB views.
        # Preserve physical camera names for recordings and the runner's external slot.
        obs = {**frames, "agentview": frames["front"], "primary_camera": "Front",
               "wrist_first": True, "extra_views": {"side": frames["side"]},
               "ee_pose": self.get_ee_pose(),
               "gripper_width": float(self.get_gripper_position()[0])}
        if "wrist_insert" in frames:
            obs["extra_views"]["wrist_insert"] = frames["wrist_insert"]
            rotation = self.data.camera("wrist_insert").xmat.reshape(3, 3)
            # Robot camera calibration only; no target/object ground truth.
            obs["insertion_camera_axes"] = {
                "image_right": rotation[:, 0].tolist(),
                "image_down": (-rotation[:, 1]).tolist(),
                "sightline": (-rotation[:, 2]).tolist(),
            }
        if "wrist_depth" in frames:
            obs["extra_views"]["wrist_depth"] = frames["wrist_depth"]
            camera = self.data.camera("wrist")
            sightline = -camera.xmat.reshape(3, 3)[:, 2]
            obs["wrist_depth_calibration"] = {
                "near_m": self.depth_near_m, "far_m": self.depth_far_m,
                "tcp_depth_m": float(np.dot(self.data.site(self._tcp).xpos - camera.xpos, sightline)),
            }
            obs["wrist_depth_m"] = raw_depth
            if self.depth_representation == "text":
                obs["wrist_depth_calibration"]["representation"] = "text"
                obs["wrist_depth_text"] = depth_to_text(
                    raw_depth, obs["wrist_depth_calibration"],
                    grid_rows=self.depth_grid_rows, grid_cols=self.depth_grid_cols,
                )
        return obs

    def check_success(self) -> bool:
        if self.scene == "plug_insert":
            return self._check_plug_insert_success()
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

    def _check_plug_insert_success(self) -> bool:
        """Require the whole cylindrical shaft to rest inside the physical bore."""
        return self.success_diagnostics()["success"]

    def success_diagnostics(self) -> dict:
        """Report physical completion criteria for logs/replay, never VLM observations.

        The plug checker uses the same evaluation, so diagnostic pass/fail values
        cannot disagree with episode success. Pick/place retains its original check.
        """
        if self.scene != "plug_insert":
            return {"scene": self.scene, "success": self.check_success(), "criteria": {}}
        tip = self.data.site("plug_tip")
        seat = self.data.site("socket_seat")
        socket_rotation = seat.xmat.reshape(3, 3)
        offset = socket_rotation.T @ (tip.xpos - seat.xpos)
        relative_rotation = socket_rotation.T @ tip.xmat.reshape(3, 3)
        peg = self.model.geom("plug_peg")
        peg_center = socket_rotation.T @ (self.data.geom("plug_peg").xpos - seat.xpos)
        # The radial envelope includes both shaft ends, so a centered tip alone
        # cannot pass a tilted shaft. Rotation around the pin axis is irrelevant.
        half_axis = peg.size[1] * relative_rotation[:, 2]
        ends = np.array([peg_center - half_axis, peg_center + half_axis])
        radial_extent = np.max(np.linalg.norm(ends[:, :2], axis=1)) + peg.size[0]
        aperture = self.model.site("socket_seat").size[0]
        plug_top = float(self.data.site("plug_top").xpos[2])
        tcp_z = float(self.get_ee_pose()[2])
        velocity = self.data.joint("plug_joint").qvel

        def criterion(value, operator, threshold, unit):
            value, threshold = float(value), float(threshold)
            passed = value > threshold if operator == ">" else value < threshold
            return {"value": value, "operator": operator, "threshold": threshold,
                    "unit": unit, "passed": bool(passed)}

        criteria = {
            "upright": criterion(relative_rotation[2, 2], ">", np.cos(np.deg2rad(6)), "cosine"),
            "shaft_fits": criterion(radial_extent, "<", aperture, "m"),
            "tip_centered": criterion(np.linalg.norm(offset[:2]), "<", 0.004, "m"),
            "insertion_depth": criterion(abs(offset[2]), "<", 0.002, "m"),
            "gripper_open": criterion(self.get_gripper_position()[0], ">", 0.06, "m"),
            "retreat_clearance": criterion(tcp_z - plug_top, ">", self.plug_retreat_clearance_m, "m"),
            "linear_rest": criterion(np.linalg.norm(velocity[:3]), "<", 0.02, "m/s"),
            "angular_rest": criterion(np.linalg.norm(velocity[3:]), "<", 0.1, "rad/s"),
        }
        # Retain the original world-height comparison exactly. At equality this
        # avoids subtraction roundoff accidentally turning 50 mm into >50 mm.
        criteria["retreat_clearance"].update(
            passed=bool(tcp_z > plug_top + self.plug_retreat_clearance_m),
            tcp_height_m=tcp_z, plug_top_height_m=plug_top,
            required_tcp_height_m=plug_top + self.plug_retreat_clearance_m,
        )
        failed = [name for name, item in criteria.items() if not item["passed"]]
        return {"scene": self.scene, "success": not failed,
                "failed_criteria": failed, "criteria": criteria}

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
