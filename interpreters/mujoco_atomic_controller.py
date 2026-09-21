"""MuJoCo single-shot Cartesian controller with explicit translation/rotation sizes.

Orientation is a quaternion setpoint. World-axis increments are left-composed
onto it, so all three axes work at arbitrary orientations without Euler-angle
singularities or the legacy yaw realignment on lifting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import DONE_ATOM, GRIPPER_ATOMS, MOVE_ATOMS, ROTATE_ATOMS, STOP_ATOM
from core.cartesian_actions import CartesianActions
from interpreters.real_atomic_controller import AtomicStepResult, RealAtomicController


@dataclass
class CartesianStepResult(AtomicStepResult):
    intended_rotation_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_deg: float = 0.0


class MujocoAtomicController(RealAtomicController):
    """Opt-in controller; the original pick/place and real controllers are unchanged."""

    LOG_TAG = "mujoco-atomic"

    def __init__(self, *args: Any, cartesian_actions: CartesianActions | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.motion_frame != "base":
            raise ValueError("MuJoCo Cartesian actions require the fixed base/world motion frame")
        for name in ("variable_step_plugin", "rotation_plugin", "smooth_plugin"):
            if bool(getattr(getattr(self, name), "enabled", False)):
                raise ValueError(f"MuJoCo explicit Cartesian actions cannot combine with {name}")
        self.cartesian_actions = cartesian_actions if cartesian_actions is not None else CartesianActions()
        self._target_quat: np.ndarray | None = None
        # Explicit magnitudes are commands, not suggestions to a legacy yaw clamp.
        self.max_position_delta_m = max(
            self.max_position_delta_m, *self.cartesian_actions.translation_steps_m.values(),
        )
        self.max_rotation_delta_rad = max(
            self.max_rotation_delta_rad,
            np.deg2rad(max(self.cartesian_actions.rotation_steps_deg.values())),
            abs(self.yaw_step_rad),
        )

    def sync_from_robot(self) -> np.ndarray:
        pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite [x,y,z,qx,qy,qz,qw] measured pose")
        self._target_pos = pose[:3].copy()
        self._target_quat = Rotation.from_quat(pose[3:]).as_quat()
        self._stream_hot = False
        self._stream_dir = None
        self.gripper_closed = float(self.robot.get_gripper_position()[0]) < self.gripper_close_threshold_m
        if self.capture_z_floor_on_sync and self.z_floor_m is None:
            self.z_floor_m = float(pose[2])
        if self.verbose:
            print(f"[{self.LOG_TAG}] synced pos={np.round(self._target_pos, 4).tolist()} "
                  f"quat_xyzw={np.round(self._target_quat, 4).tolist()}")
        return pose

    def _ensure_synced(self) -> None:
        if self._target_pos is None or self._target_quat is None:
            self.sync_from_robot()

    @property
    def target_pose(self) -> np.ndarray:
        self._ensure_synced()
        return np.concatenate([self._target_pos, self._target_quat])

    def heading_yaw_rad(self) -> float | None:
        self._ensure_synced()
        tool = Rotation.from_quat(self._target_quat).apply(np.asarray(self.TOOL_AXIS, float))
        if np.hypot(tool[0], tool[1]) < 0.1:
            return None
        return float(np.arctan2(tool[1], tool[0]))

    def step(
        self,
        token: str,
        target_in_wrist: bool | None = None,
        continuous: bool = False,
        motion_frame: str | None = None,
        step_override_m: float | None = None,
    ) -> AtomicStepResult:
        """Execute one fixed movement; translations never change orientation.

        Visibility, streaming, and legacy yaw/variable-step plugins do not modify
        explicit commands. MuJoCo itself interpolates the actuator motion.
        """
        token = token.strip().upper()
        self._ensure_synced()
        if motion_frame is not None and str(motion_frame).strip().lower() != "base":
            raise ValueError("MuJoCo Cartesian actions always use fixed base/world axes")
        if token in GRIPPER_ATOMS or token == DONE_ATOM:
            return super().step(token)

        pre_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        action = self.cartesian_actions.decode(token)
        delta = np.zeros(3)
        rotvec = np.zeros(3)
        step_kind, step_m = "", 0.0
        if action is not None:
            kind, step_kind = action.kind, action.size
            if kind == "move":
                step_m = self.cartesian_actions.translation_steps_m[action.size]
                delta = self.move_vectors[action.base_token] * step_m
            else:
                rotvec[action.axis] = np.deg2rad(
                    self.cartesian_actions.rotation_steps_deg[action.size] * action.sign,
                )
        elif token in MOVE_ATOMS:
            kind = "move"
            step_m = self.step_m if step_override_m is None else float(step_override_m)
            if not np.isfinite(step_m) or step_m <= 0:
                raise ValueError("Translation step must be finite and positive")
            delta = self.move_vectors[token] * step_m
        elif token in ROTATE_ATOMS:
            kind = "rotate"
            rotvec[2] = self.yaw_signs[token] * self.yaw_step_rad
        else:
            kind = "stop" if token == STOP_ATOM else "unknown"

        start_pos, start_quat = self._target_pos.copy(), self._target_quat.copy()
        note = "unknown token -> hold" if kind == "unknown" else ""
        self._target_pos = start_pos + delta
        if self.z_floor_m is not None and self._target_pos[2] < self.z_floor_m:
            blocked = float(self.z_floor_m - self._target_pos[2])
            self._target_pos[2] = self.z_floor_m
            note = f"z-floor: blocked {blocked:.4f}m of descent (floor={self.z_floor_m:.4f}m)"
        if np.any(rotvec):
            self._target_quat = (
                Rotation.from_rotvec(rotvec) * Rotation.from_quat(start_quat)
            ).as_quat()
        try:
            self._command_setpoint()
        except Exception:
            # MuJoCo rejects unreachable IK before moving the live model. Do not
            # leave that rejected target integrated into the next command.
            self._target_pos, self._target_quat = start_pos, start_quat
            raise
        post_pose = np.asarray(self.robot.get_ee_pose(), dtype=float)
        rotation_deg = float(np.rad2deg(np.linalg.norm(rotvec)))
        if self.verbose:
            print(f"[{self.LOG_TAG}] {token} d_pos(m)={np.round(delta, 4).tolist()} "
                  f"d_rot(deg)={np.round(np.rad2deg(rotvec), 2).tolist()}"
                  + (f" [{note}]" if note else ""))
        return CartesianStepResult(
            token=token, kind=kind, intended_delta_m=delta,
            intended_yaw_rad=float(rotvec[2]), intended_rotation_rad=rotvec,
            rotation_deg=rotation_deg, pre_pose=pre_pose, target_pose=self.target_pose,
            post_pose=post_pose, gripper_closed=self.gripper_closed, note=note,
            step_kind=step_kind, step_m=step_m,
        )
