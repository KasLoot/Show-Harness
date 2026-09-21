"""The insertion vocabulary executes exact, independent Cartesian single shots."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import MOVE_ATOMS
from core.cartesian_actions import CartesianActions
from interpreters.mujoco_atomic_controller import MujocoAtomicController


ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = {
    "MV_FWD": [1, 0, 0], "MV_BACK": [-1, 0, 0],
    "MV_RIGHT": [0, 1, 0], "MV_LEFT": [0, -1, 0],
    "MV_UP": [0, 0, 1], "MV_DOWN": [0, 0, -1],
    "ROTATE_CW": 1, "ROTATE_CCW": -1,
}
AVAILABLE = importlib.util.find_spec("mujoco") is not None and (
    ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
).is_file()


class FakeRobot:
    def __init__(self):
        self.pose = np.r_[[0.4, 0.0, 0.25], Rotation.from_euler("xyz", [20, -30, 40], degrees=True).as_quat()]
        self.width = 0.08
        self.fail_next = False

    def get_ee_pose(self):
        return self.pose.copy()

    def get_gripper_position(self):
        return np.array([self.width])

    def update_desired_ee_pose(self, pose):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Panda IK could not reach target")
        self.pose = pose.copy()

    def control_gripper(self, close):
        self.width = 0.03 if close else 0.08


def make_fake(**kwargs):
    robot = FakeRobot()
    controller = MujocoAtomicController(
        robot, PRIMITIVES, step_m=0.02, yaw_step_rad=0.15,
        settle_steps=1, settle_dt_s=0, gripper_settle_s=0, verbose=False,
        **kwargs,
    )
    controller.sync_from_robot()
    return robot, controller


class CartesianActionTests(unittest.TestCase):
    def test_vocabulary_sizes_and_prompt_are_driven_by_config(self):
        actions = CartesianActions({"cartesian_motion": {
            "translation_steps_m": {"small": 0.001, "medium": 0.008, "large": 0.08},
            "rotation_steps_deg": {"small": 1, "medium": 8, "large": 45},
        }})
        tokens = actions.action_tokens()
        self.assertEqual(len(tokens), 37)
        self.assertEqual(len(set(tokens)), len(tokens))
        self.assertIn("MV_FWD_SMALL", tokens)
        self.assertIn("ROT_Z_NEG_LARGE", tokens)
        self.assertEqual(tokens[-1], "STOP")
        text = actions.render_prompt()
        for expected in ("SMALL=1 mm", "MEDIUM=8 mm", "LARGE=80 mm", "LARGE=45 degrees",
                         "FWD=+X", "RIGHT=+Y", "UP=+Z", "right-hand rule"):
            self.assertIn(expected, text)
        robot, controller = make_fake(cartesian_actions=actions)
        before = robot.pose.copy()
        result = controller.step("MV_FWD_LARGE")
        self.assertAlmostEqual(robot.pose[0] - before[0], 0.08)
        self.assertEqual(result.step_m, 0.08)
        result = controller.step("ROT_X_POS_LARGE")
        self.assertAlmostEqual(result.rotation_deg, 45)

    def test_invalid_sizes_are_rejected(self):
        for field in ("translation_steps_m", "rotation_steps_deg"):
            for value in (None, [], {}, {"small": 1},
                          {"small": 0, "medium": 1, "large": 2},
                          {"small": 1, "medium": 1, "large": 2},
                          {"small": 1, "medium": 3, "large": 2},
                          {"small": 1, "medium": 2, "large": float("nan")},
                          {"small": 1, "medium": 2, "large": float("inf")},
                          {"small": True, "medium": 2, "large": 3},
                          {"small": "bad", "medium": 2, "large": 3}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    CartesianActions({field: value})
        for large in (180, 270, 360):
            with self.subTest(large=large), self.assertRaisesRegex(ValueError, "below 180"):
                CartesianActions({"rotation_steps_deg": {"small": 2, "medium": 10, "large": large}})

    def test_prompt_tracks_rotated_wrist_directions_in_world_axes(self):
        actions = CartesianActions()
        text = actions.render_prompt({"eef_quat": [0, 0, 0, 1]})
        self.assertIn("hand/tool +Z=[+0.000, +0.000, +1.000]", text)
        self.assertIn("wrist image-right=[+0.000, -1.000, +0.000]", text)
        self.assertIn("wrist image-down=[+1.000, +0.000, +0.000]", text)
        text = actions.render_prompt({"eef_quat": Rotation.from_euler("y", 90, degrees=True).as_quat()})
        self.assertIn("hand/tool +Z=[+1.000, +0.000, +0.000]", text)
        self.assertIn("wrist image-down=[+0.000, +0.000, -1.000]", text)

    def test_all_translations_preserve_arbitrary_orientation_and_grasp(self):
        for token in MOVE_ATOMS:
            for size, distance in (("SMALL", 0.002), ("MEDIUM", 0.01), ("LARGE", 0.05)):
                with self.subTest(token=token, size=size):
                    robot, controller = make_fake()
                    controller.step("ROT_Y_POS_LARGE")
                    controller.step("GRASP")
                    before = robot.pose.copy()
                    result = controller.step(f"{token}_{size}", target_in_wrist=False)
                    np.testing.assert_allclose(robot.pose[:3] - before[:3], np.array(PRIMITIVES[token]) * distance)
                    np.testing.assert_array_equal(robot.pose[3:], before[3:])
                    self.assertTrue(result.gripper_closed)
                    self.assertEqual((result.step_kind, result.step_m), (size.lower(), distance))

    def test_each_axis_sign_size_composes_in_world_frame_at_fixed_tcp(self):
        for axis_idx, axis in enumerate("XYZ"):
            for sign_name, sign in (("POS", 1), ("NEG", -1)):
                for size, degrees in (("SMALL", 2), ("MEDIUM", 10), ("LARGE", 30)):
                    token = f"ROT_{axis}_{sign_name}_{size}"
                    with self.subTest(token=token):
                        robot, controller = make_fake()
                        before = robot.pose.copy()
                        result = controller.step(token)
                        actual = Rotation.from_quat(robot.pose[3:]) * Rotation.from_quat(before[3:]).inv()
                        expected = np.eye(3)[axis_idx] * np.deg2rad(degrees) * sign
                        np.testing.assert_allclose(actual.as_rotvec(), expected, atol=1e-14)
                        np.testing.assert_allclose(result.intended_rotation_rad, expected)
                        np.testing.assert_array_equal(robot.pose[:3], before[:3])
                        self.assertAlmostEqual(result.rotation_deg, degrees)
                        self.assertEqual(result.step_kind, size.lower())
                        self.assertEqual(result.step_m, 0)

    def test_rotations_do_not_commute_and_inverse_restores_pose(self):
        robot, controller = make_fake()
        before = robot.pose.copy()
        for token in ("ROT_X_POS_LARGE", "ROT_Y_POS_LARGE"):
            controller.step(token)
        xy = Rotation.from_quat(robot.pose[3:])
        for token in ("ROT_Y_NEG_LARGE", "ROT_X_NEG_LARGE"):
            controller.step(token)
        np.testing.assert_allclose(robot.pose, before, atol=1e-14)
        for token in ("ROT_Y_POS_LARGE", "ROT_X_POS_LARGE"):
            controller.step(token)
        yx = Rotation.from_quat(robot.pose[3:])
        self.assertGreater((xy * yx.inv()).magnitude(), 0.2)

    def test_gimbal_lock_orientation_is_preserved_by_a_lift(self):
        robot, controller = make_fake()
        robot.pose[3:] = Rotation.from_euler("xyz", [17, 90, 23], degrees=True).as_quat()
        controller.sync_from_robot()
        before = robot.pose[3:].copy()
        controller.step("GRASP")
        result = controller.step("MV_UP")
        np.testing.assert_allclose(robot.pose[3:], before, atol=1e-14)
        self.assertEqual(result.kind, "move")
        self.assertEqual(result.step_m, 0.02)

    def test_floor_and_unknown_tokens_hold_without_hidden_rotations(self):
        robot, controller = make_fake(z_floor_m=0.24)
        result = controller.step("MV_DOWN_LARGE")
        self.assertAlmostEqual(robot.pose[2], 0.24)
        self.assertIn("z-floor", result.note)
        before = robot.pose.copy()
        for token, kind in (("STOP", "stop"), ("ROT_Q_POS_SMALL", "unknown")):
            result = controller.step(token)
            self.assertEqual(result.kind, kind)
            np.testing.assert_array_equal(robot.pose, before)
        self.assertTrue(controller.step("DONE").done)

    def test_ik_rejection_rolls_back_translation_and_orientation_setpoints(self):
        for token in ("MV_FWD_LARGE", "ROT_X_POS_LARGE"):
            with self.subTest(token=token):
                robot, controller = make_fake()
                before = controller.target_pose.copy()
                robot.fail_next = True
                with self.assertRaisesRegex(RuntimeError, "IK"):
                    controller.step(token)
                np.testing.assert_array_equal(controller.target_pose, before)
                controller.step("MV_UP_SMALL")
                np.testing.assert_allclose(robot.pose[:3], before[:3] + [0, 0, 0.002])
                np.testing.assert_array_equal(robot.pose[3:], before[3:])


@unittest.skipUnless(AVAILABLE, "Run bash scripts/setup.sh mujoco for the physics check")
class PhysicalCartesianTests(unittest.TestCase):
    def test_explicit_sizes_and_world_rotations_reach_measured_tcp(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg["cartesian_motion"] = {}
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        controller = make_controller(session, cfg)
        controller.verbose = False
        for size, distance in (("SMALL", 0.002), ("MEDIUM", 0.01), ("LARGE", 0.05)):
            before = session.get_ee_pose()
            controller.step(f"MV_FWD_{size}")
            np.testing.assert_allclose(session.get_ee_pose()[:3] - before[:3], [distance, 0, 0], atol=0.001)
            controller.step(f"MV_BACK_{size}")
        for axis_idx, axis in enumerate("XYZ"):
            before = session.get_ee_pose()
            controller.step(f"ROT_{axis}_POS_LARGE")
            after = session.get_ee_pose()
            actual = Rotation.from_quat(after[3:]) * Rotation.from_quat(before[3:]).inv()
            expected = np.eye(3)[axis_idx] * np.deg2rad(30)
            np.testing.assert_allclose(actual.as_rotvec(), expected, atol=0.015)
            np.testing.assert_allclose(after[:3], before[:3], atol=0.002)
            controller.step(f"ROT_{axis}_NEG_LARGE")


if __name__ == "__main__":
    unittest.main()
