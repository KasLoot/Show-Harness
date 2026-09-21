"""Real contact-physics checks for the round plug scene; no VLM or renderer needed."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("mujoco") is not None and (
    ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
).is_file()


class PlugSuccessGoalTests(unittest.TestCase):
    def test_prompt_and_checker_share_default_and_configured_clearance(self):
        from core.sim.mujoco_success import (
            PLUG_RETREAT_CLEARANCE_M, PLUG_RETREAT_MARGIN_M,
            plug_retreat_clearance_m, plug_success_prompt,
        )

        self.assertEqual(PLUG_RETREAT_CLEARANCE_M, .05)
        self.assertEqual(PLUG_RETREAT_MARGIN_M, .005)
        for config, required_mm, target_mm in (({}, 50, 55),
                                             ({"plug_success": {"retreat_clearance_m": .07}}, 70, 75)):
            with self.subTest(config=config):
                self.assertEqual(plug_retreat_clearance_m(config) * 1000, required_mm)
                prompt = plug_success_prompt(config)
                self.assertIn(f"strictly more than {required_mm} mm", prompt)
                self.assertIn(f"at least {target_mm} mm", prompt)
                self.assertIn("world-Z gap", prompt)
                self.assertIn("before the final DONE", prompt)

    def test_invalid_retreat_configuration_is_rejected(self):
        from core.sim.mujoco_success import plug_retreat_clearance_m

        for value in (0, -.01, True, False, None, "invalid", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "retreat_clearance_m"):
                plug_retreat_clearance_m({"plug_success": {"retreat_clearance_m": value}})
        with self.assertRaisesRegex(ValueError, "plug_success"):
            plug_retreat_clearance_m({"plug_success": []})


@unittest.skipUnless(AVAILABLE, "Run bash scripts/setup.sh mujoco for the physics check")
class MujocoPlugSceneTests(unittest.TestCase):
    def make_session(self, scene="plug_insert", **overrides):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg["scene"] = scene
        cfg.update(overrides)
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        return session

    def put_plug(self, session, tip, rotation):
        """Place test fixtures only; runtime actions never set object poses."""
        import mujoco

        joint = session.data.joint("plug_joint")
        joint.qpos[:3] = np.asarray(tip) - rotation.apply(session.model.site("plug_tip").pos)
        joint.qpos[3:] = np.roll(rotation.as_quat(), 1)
        joint.qvel[:] = 0
        mujoco.mj_forward(session.model, session.data)

    def test_scene_selection_preserves_pick_place_and_rejects_unknown_scene(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.pop("scene", None)
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        self.assertEqual(session.scene, "pick_place")
        self.assertIsNotNone(session.model.body("cube"))
        self.assertFalse(session.check_success())
        cfg["scene"] = "../pick_place"
        with self.assertRaisesRegex(ValueError, "scene"):
            MujocoSession(cfg)

    def test_stand_is_stable_and_round_socket_accepts_any_yaw(self):
        session = self.make_session()
        initial = session.data.body("plug").xpos.copy()
        session._advance(1.0)
        np.testing.assert_allclose(session.data.body("plug").xpos, initial, atol=1e-4)
        self.assertFalse(session.check_success())
        for camera in ("side", "front", "wrist"):
            self.assertIsNotNone(session.model.camera(camera))

        # A round shaft falls into the actual bore and rests on the floor at every
        # yaw. The handle's orientation marker does not impose an artificial key.
        for yaw in (0, 37, 90, 180, 270):
            with self.subTest(yaw=yaw):
                self.put_plug(session, [.48, .16, .09], Rotation.from_euler("z", yaw, degrees=True))
                session._advance(1.0)
                self.assertTrue(session.check_success())
                self.assertAlmostEqual(session.data.site("plug_tip").xpos[2], .015, delta=.001)
                for _ in range(3):
                    session._advance(.1)
                    self.assertTrue(session.check_success(), "The pin must remain at rest at any yaw")

    def test_round_bore_is_open_with_black_wall_and_bottom_and_real_side_contacts(self):
        import mujoco

        session = self.make_session()
        hit = np.array([-1], dtype=np.int32)
        # Exclude noncolliding visual meshes (group 2). The first actual geometry
        # below the opening center must be the bottom, never a solid mesh-hull cap.
        distance = mujoco.mj_ray(session.model, session.data,
                                 np.array([.48, .16, .1]), np.array([0., 0., -1.]),
                                 np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8), 1, -1, hit)
        self.assertEqual(hit[0], session.model.geom("socket_floor").id)
        self.assertAlmostEqual(distance, .085, places=6)
        for name in ("socket_interior_visual", "socket_floor"):
            self.assertTrue(np.all(session.model.geom(name).rgba[:3] < .01))
        for name in ("socket_interior_visual", "socket_exterior_visual"):
            self.assertEqual(session.model.geom(name).contype, 0)
            self.assertEqual(session.model.geom(name).conaffinity, 0)

        peg = session.model.geom("plug_peg").id
        self.assertEqual(session.model.geom("plug_peg").type, mujoco.mjtGeom.mjGEOM_CYLINDER)
        wall_ids = {session.model.geom(i).id for i in range(session.model.ngeom)
                    if session.model.geom(i).name.startswith("socket_wall_")}
        self.assertGreaterEqual(len(wall_ids), 96)

        def shaft_wall_contacts():
            return [contact for contact in session.data.contact
                    if peg in contact.geom and any(g in wall_ids for g in contact.geom)]

        self.put_plug(session, [.48, .16, .03], Rotation.identity())
        self.assertFalse(shaft_wall_contacts(), "A centered shaft must fit inside the bore")
        self.put_plug(session, [.484, .16, .03], Rotation.identity())
        self.assertTrue(shaft_wall_contacts(), "An off-center shaft must collide with the bore wall")

    def test_success_requires_depth_alignment_orientation_release_rest_and_retreat(self):
        session = self.make_session()
        aligned = Rotation.from_euler("z", 90, degrees=True)
        tip = np.array([.48, .16, .015])
        self.put_plug(session, tip, aligned)
        self.assertTrue(session.check_success())

        cases = (
            ("hover", tip + [0, 0, .06], aligned),
            ("partial insertion", tip + [0, 0, .015], aligned),
            ("wedged just above the seat", tip + [0, 0, .003], aligned),
            ("below floor", tip - [0, 0, .01], aligned),
            ("lateral misalignment", tip + [.006, 0, 0], aligned),
            ("shaft intersects wall despite near-centered tip", tip + [.0036, 0, 0], aligned),
            ("shaft tilt intersects rim", tip, aligned * Rotation.from_euler("y", 5, degrees=True)),
            ("tilted", tip, aligned * Rotation.from_euler("x", 15, degrees=True)),
            ("upside down", tip, aligned * Rotation.from_euler("x", 180, degrees=True)),
        )
        for label, location, rotation in cases:
            with self.subTest(case=label):
                self.put_plug(session, location, rotation)
                self.assertFalse(session.check_success())

        self.put_plug(session, tip, aligned)
        with patch.object(session, "get_gripper_position", return_value=np.array([.032])):
            self.assertFalse(session.check_success(), "A plug still being held is not complete")
        pose = session.get_ee_pose()
        pose[:3] = [.48, .16, .11]
        with patch.object(session, "get_ee_pose", return_value=pose):
            self.assertFalse(session.check_success(), "The fingers must withdraw from the socket")
        for index, speed in ((0, .03), (3, .2)):
            session.data.joint("plug_joint").qvel[:] = 0
            session.data.joint("plug_joint").qvel[index] = speed
            self.assertFalse(session.check_success(), "The plug must come to rest")

    def test_retreat_boundary_diagnostics_do_not_loosen_completion_or_move_the_robot(self):
        session = self.make_session()
        self.put_plug(session, [.48, .16, .015], Rotation.identity())
        plug_top = float(session.data.site("plug_top").xpos[2])
        time_before = float(session.data.time)
        qpos_before = session.data.qpos.copy()
        ctrl_before = session.data.ctrl.copy()
        for gap, succeeds in ((.0426, False), (.05, False), (.0526, True)):
            pose = session.get_ee_pose()
            pose[:3] = [.48, .16, plug_top + gap]
            with self.subTest(gap=gap), patch.object(session, "get_ee_pose", return_value=pose):
                diagnostics = session.success_diagnostics()
                self.assertEqual(diagnostics["success"], succeeds)
                self.assertEqual(session.check_success(), succeeds)
                self.assertEqual(diagnostics["failed_criteria"], [] if succeeds else ["retreat_clearance"])
                self.assertEqual(len(diagnostics["criteria"]), 8)
                retreat = diagnostics["criteria"]["retreat_clearance"]
                self.assertEqual((retreat["operator"], retreat["threshold"], retreat["unit"]),
                                 (">", .05, "m"))
                self.assertAlmostEqual(retreat["value"], gap)
                self.assertAlmostEqual(retreat["required_tcp_height_m"], plug_top + .05)
                self.assertEqual(retreat["passed"], succeeds)
                # A logs-only diagnostic must contain plain JSON values.
                self.assertEqual(json.loads(json.dumps(diagnostics, allow_nan=False)), diagnostics)
        self.assertEqual(session.data.time, time_before)
        np.testing.assert_array_equal(session.data.qpos, qpos_before)
        np.testing.assert_array_equal(session.data.ctrl, ctrl_before)

    def test_custom_clearance_is_used_by_the_physical_checker(self):
        session = self.make_session(plug_success={"retreat_clearance_m": .07})
        self.put_plug(session, [.48, .16, .015], Rotation.identity())
        pose = session.get_ee_pose()
        pose[2] = session.data.site("plug_top").xpos[2] + .06
        with patch.object(session, "get_ee_pose", return_value=pose):
            diagnostics = session.success_diagnostics()
            self.assertEqual(diagnostics["failed_criteria"], ["retreat_clearance"])
            self.assertEqual(diagnostics["criteria"]["retreat_clearance"]["threshold"], .07)
            self.assertFalse(session.check_success())

    def test_success_diagnostics_are_not_injected_into_visual_observations(self):
        session = self.make_session()
        with patch.object(session, "success_diagnostics", side_effect=AssertionError("logs only")), \
                patch.object(session, "render", return_value=np.zeros((16, 16, 3), np.uint8)), \
                patch.object(session, "render_depth", return_value=np.full((16, 16), .1, np.float32)):
            observation = session.get_observation()
        for name in ("success_diagnostics", "criteria", "plug_top_height_m", "retreat_clearance"):
            self.assertNotIn(name, observation)
            self.assertNotIn(name, observation["wrist_depth_calibration"])

    def test_physical_grasp_lift_turn_insert_release_and_retreat(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco_plug.yaml")
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        controller = make_controller(session, cfg)
        controller.verbose = False
        sizes = controller.cartesian_actions.translation_steps_m
        used_sizes = set()

        def move_to(position, maximum="large"):
            for _ in range(150):
                error = np.asarray(position) - controller.target_pose[:3]
                if np.max(np.abs(error)) <= sizes["small"] / 2:
                    return
                axis = int(np.argmax(np.abs(error)))
                size = next(name for name in ("large", "medium", "small")
                            if sizes[name] <= sizes[maximum]
                            and sizes[name] <= abs(error[axis]) + sizes["small"] / 2)
                base = (("MV_BACK", "MV_FWD"), ("MV_LEFT", "MV_RIGHT"),
                        ("MV_DOWN", "MV_UP"))[axis][int(error[axis] > 0)]
                token = f"{base}_{size.upper()}"
                result = controller.step(token)
                self.assertEqual(result.step_kind, size)
                used_sizes.add(size)
            self.fail("Atomic movements did not reach target")

        # Oracle positions are exclusive to this physics test. The deployed
        # planner/controller continue to act from their ordinary RGB observations.
        move_to([.48, -.10, .18])
        move_to([.48, -.10, .106], maximum="medium")
        controller.step("GRASP")
        self.assertTrue(controller.gripper_closed)
        self.assertGreater(session.get_gripper_position()[0], .02)
        self.assertLess(session.get_gripper_position()[0], .04)
        move_to([.48, -.10, .20], maximum="medium")
        self.assertGreater(session.data.body("plug").xpos[2], .18)
        controller.step("STOP")
        held_offset = session.data.body("plug").xpos - session.get_ee_pose()[:3]
        for _ in range(10):
            controller.step("STOP")
        np.testing.assert_allclose(
            session.data.body("plug").xpos - session.get_ee_pose()[:3], held_offset, atol=.001,
            err_msg="Contact grasp must hold without creeping during a long action sequence",
        )
        for _ in range(3):
            controller.step("ROT_Z_POS_LARGE")
        move_to([.48, .16, .20])
        self.assertFalse(session.check_success(), "Hovering above the hole does not count")
        move_to([.48, .16, .14], maximum="medium")
        move_to([.48, .16, .081], maximum="small")
        self.assertFalse(session.check_success(), "Insertion still needs release and retreat")
        controller.step("RELEASE")
        move_to([.48, .16, .20])
        self.assertEqual(used_sizes, {"small", "medium", "large"})
        self.assertTrue(session.check_success())
        for _ in range(3):
            controller.step("STOP")
            self.assertTrue(session.check_success(), "The seated pin must remain stably at rest")


if __name__ == "__main__":
    unittest.main()
