"""Physical pick/place check. Requires `bash scripts/setup.sh mujoco`; no cloud calls."""
import importlib.util
import base64
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("mujoco") is not None and (
    ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
).is_file()


@unittest.skipUnless(AVAILABLE, "Run bash scripts/setup.sh mujoco for the physics check")
class MujocoTests(unittest.TestCase):
    def test_wrist_front_side_reach_planner_controller_and_saved_observations(self):
        for config in ("robot_mujoco.yaml", "robot_mujoco_plug.yaml"):
            with self.subTest(config=config):
                self._check_view_inputs(config)

    def _check_view_inputs(self, config):
        from PIL import Image
        from core.config import load_yaml
        from core.sim.mujoco_depth import depth_to_grayscale
        from core.vlm.vlm_client import VLMResponse, _message_content
        from scripts.run_mujoco import main

        cfg = load_yaml(ROOT / "configs" / config)
        cfg.update(max_steps=1, camera_resolution=128)
        plug = cfg.get("scene") == "plug_insert"
        if plug:
            # Exercise the shared setting through actual planner/controller prompts
            # and persisted evaluator diagnostics, beyond the default 50 mm.
            cfg["plug_success"] = {"retreat_clearance_m": 0.07}
        order = ("wrist", "front", "side", "wrist_insert") if plug else ("wrist", "front", "side")
        frames = {name: np.full((128, 128, 3), value, np.uint8)
                  for name, value in (("side", 40), ("wrist", 100), ("front", 200),
                                      ("wrist_insert", 240), ("wrist_depth", 150))}
        depth_m = np.full((128, 128), 0.12, np.float32)
        frames["wrist_depth"] = depth_to_grayscale(depth_m)
        plan = {"subgoals": [{"id": "lift", "motion": "LIFT", "target": "cube",
                              "affordance": "cube", "description": "Lift the cube.",
                              "completion": "Cube above table."}]}
        action = {"decision": "MV_UP", "reasoning": "Lift above the table."}
        client = Mock()
        client.complete_json.side_effect = [VLMResponse("", json.dumps(v), {"json": v}) for v in (plan, action)]
        with TemporaryDirectory() as directory, \
                patch("core.sim.mujoco_session.MujocoSession.render", side_effect=frames.__getitem__), \
                patch("core.sim.mujoco_session.MujocoSession.render_depth", return_value=depth_m), \
                patch("scripts.run_mujoco.make_vlm_client", return_value=client) as factory, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            self.assertEqual(main(["--robot-config", str(config_path), "--log-dir", directory,
                                   "--model", "test-openai-model"]), 1)
            self.assertEqual(factory.call_args.args[1]["vlm"]["model"], "test-openai-model")
            self.assertEqual(factory.call_args.args[1]["vlm_backend"], "openai")
            self.assertEqual(factory.call_args.args[1]["vlm"]["provider"], "openai")
            self.assertEqual(factory.call_args.args[1]["vlm"]["api_key"], "test-key")
            self.assertEqual(client.complete_json.call_count, 2)
            for call in client.complete_json.call_args_list:
                np.testing.assert_array_equal(call.args[1], frames["wrist"])
                self.assertEqual(len(call.kwargs["wrist_image"]), len(order) - 1)
                for sent, name in zip(call.kwargs["wrist_image"], order[1:]):
                    np.testing.assert_array_equal(sent, frames[name])
                content = _message_content(call.args[0], call.args[1], call.kwargs["wrist_image"])
                images = [part for part in content if part["type"] == "image_url"]
                self.assertEqual(len(images), len(order))
                for part, name in zip(images, order):
                    encoded = part["image_url"]["url"].split(",", 1)[1]
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as sent:
                        np.testing.assert_array_equal(np.asarray(sent), frames[name])
                self.assertNotIn("AgentView", call.args[0])
                self.assertNotIn("Front (C)", call.args[0])
                self.assertIn("Four simultaneous RGB images" if plug else "Three simultaneous images", call.args[0])
                if plug:
                    self.assertIn("WRIST_DEPTH_MM", call.args[0])
                    self.assertIn("TCP plane is 48.0 mm", call.args[0])
                    self.assertIn("strictly more than 70 mm", call.args[0])
                    self.assertIn("at least 75 mm", call.args[0])
                    self.assertNotIn("{retreat_", call.args[0])
                    self.assertNotIn("success_diagnostics", call.args[0])
                    self.assertNotIn("plug_top_height_m", call.args[0])
                else:
                    self.assertNotIn("Wrist Depth", call.args[0])
                self.assertIn("closer", call.args[0])
                self.assertIn("farther", call.args[0])
                self.assertIn("orthographic", call.args[0])
                for obsolete in ("down-right", "down-left", "up-right", "up-left",
                                 "A and C mix height", "Front (B) for coarse"):
                    self.assertNotIn(obsolete, call.args[0])
            controller_prompt = client.complete_json.call_args_list[-1].args[0]
            self.assertIn("24.0 cm above the table", controller_prompt)
            labels = "Wrist (A), Front (B), Right Side (C)" + (", Angled Wrist (D)" if plug else "")
            self.assertIn(f"Images are ordered {labels}.", controller_prompt)
            self.assertNotIn("MV_DOWN first", controller_prompt)
            run = next(Path(directory).rglob("steps.jsonl")).parent
            step = json.loads((run / "steps.jsonl").read_text().splitlines()[0])
            summary = json.loads((run / "summary.json").read_text())
            if plug:
                for record in (step, summary):
                    diagnostics = record["success_diagnostics"]
                    self.assertFalse(diagnostics["success"])
                    self.assertEqual(diagnostics["criteria"]["retreat_clearance"]["threshold"], 0.07)
            else:
                for record in (step, summary):
                    self.assertEqual(record["success_diagnostics"]["scene"], "pick_place")
                    self.assertEqual(record["success_diagnostics"]["criteria"], {})
            saved_order = (*order, "wrist_depth") if plug else order
            self.assertEqual({p.name for p in (run / "images").iterdir()}, set(saved_order))
            for name in saved_order:
                np.testing.assert_array_equal(np.asarray(Image.open(run / "images" / name / "0000.png")), frames[name])
                np.testing.assert_array_equal(np.asarray(Image.open(run / f"planner_{name}.png")), frames[name])
            roles = json.loads((run / "planner_diagnostics.json").read_text())["attempts"][0]["image_roles"]
            self.assertEqual(len(roles), len(order))
            role_names = ("Wrist", "Front", "Side", "Angled Wrist") if plug else ("Wrist", "Front", "Side")
            for role, name in zip(roles, role_names):
                self.assertIn(name, role)
            if plug:
                np.testing.assert_array_equal(np.load(run / "planner_wrist_depth_m.npy"), depth_m)
                np.testing.assert_array_equal(np.load(run / "depth/wrist/0000.npy"), depth_m)
                initial_depth = (run / "planner_wrist_depth.txt").read_text()
                self.assertIn(initial_depth, client.complete_json.call_args_list[0].args[0])
                current_depth = (run / "depth/wrist/0000.txt").read_text()
                self.assertIn(current_depth, controller_prompt)
                self.assertNotIn("Wrist Depth (E)", controller_prompt)
            self.assertFalse(list(run.rglob("*.mp4")))

    def test_insertion_camera_mount_tracks_hand_and_reports_measured_world_axes(self):
        import mujoco
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco_plug.yaml")
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        camera = session.model.camera("wrist_insert")
        local_right = np.array([0.0, -1.0, 0.0])
        local_up = np.array([-0.7488700551657237, 0.0, -0.6627168629785166])
        local_rotation = np.column_stack((local_right, local_up, np.cross(local_right, local_up)))
        self.assertEqual(camera.bodyid[0], session.model.body("hand").id)
        np.testing.assert_allclose(camera.pos, [0.1, 0.0, 0.045], atol=1e-12)
        self.assertAlmostEqual(camera.fovy[0], 65)
        self.assertEqual(session.model.cam_projection[camera.id], mujoco.mjtProjection.mjPROJ_PERSPECTIVE)
        self.assertEqual(set(session.cameras), {"side", "wrist", "front", "wrist_insert", "wrist_depth"})
        controller = make_controller(session, cfg)
        observed_axes = []
        with patch.object(session, "render", return_value=np.zeros((16, 16, 3), np.uint8)), \
                patch.object(session, "render_depth", return_value=np.full((16, 16), 0.12, np.float32)):
            for token in (None, "ROT_X_POS_MEDIUM", "ROT_Z_NEG_MEDIUM"):
                if token:
                    controller.step(token)
                hand = session.data.body("hand")
                hand_rotation = hand.xmat.reshape(3, 3)
                view = session.data.camera("wrist_insert")
                expected_rotation = hand_rotation @ local_rotation
                np.testing.assert_allclose(view.xpos, hand.xpos + hand_rotation @ camera.pos, atol=1e-10)
                np.testing.assert_allclose(view.xmat.reshape(3, 3), expected_rotation, atol=1e-10)
                obs = session.get_observation()
                self.assertEqual(list(obs["extra_views"]), ["side", "wrist_insert", "wrist_depth"])
                calibration = obs["wrist_depth_calibration"]
                self.assertEqual((calibration["near_m"], calibration["far_m"]), (0.0, 0.30))
                self.assertAlmostEqual(calibration["tcp_depth_m"], 0.048, places=8)
                axes = obs["insertion_camera_axes"]
                for name, expected in (("image_right", expected_rotation[:, 0]),
                                       ("image_down", -expected_rotation[:, 1]),
                                       ("sightline", -expected_rotation[:, 2])):
                    np.testing.assert_allclose(axes[name], expected, atol=1e-10)
                observed_axes.append(np.asarray(axes["image_down"]))
                if token is None:
                    np.testing.assert_allclose(axes["image_right"], [0, 1, 0], atol=0.001)
                    np.testing.assert_allclose(axes["image_down"], [0.7488700551657237, 0, -0.6627168629785166], atol=0.001)
                    np.testing.assert_allclose(axes["sightline"], [-0.6627168629785166, 0, -0.7488700551657237], atol=0.001)
        self.assertTrue(all(np.linalg.norm(after - before) > 0.1
                            for before, after in zip(observed_axes, observed_axes[1:])))

        cube = MujocoSession(load_yaml(ROOT / "configs/robot_mujoco.yaml"))
        self.addCleanup(cube.close)
        self.assertEqual(set(cube.cameras), {"side", "wrist", "front"})
        self.assertEqual(mujoco.mj_name2id(cube.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_insert"), -1)
        with patch.object(cube, "render", return_value=np.zeros((16, 16, 3), np.uint8)):
            cube_obs = cube.get_observation()
            for key in ("insertion_camera_axes", "wrist_depth", "wrist_depth_calibration"):
                self.assertNotIn(key, cube_obs)

    def test_recording_global_view_does_not_enter_observations_or_vlm_inputs(self):
        from core.config import load_yaml
        from core.record.images import vlm_camera_views
        from core.record.mujoco_recorder import MujocoRecorder
        from core.sim.mujoco_session import MujocoSession

        cfg = load_yaml(ROOT / "configs/robot_mujoco_plug.yaml")
        cfg["camera_resolution"] = 128
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        cameras = session.cameras
        before = session.data.time
        rgb = np.zeros((128, 128, 3), np.uint8)
        with TemporaryDirectory() as directory, \
                patch.object(session, "render", return_value=rgb), \
                patch.object(session, "render_depth", return_value=np.full((128, 128), .12, np.float32)), \
                patch("core.record.mujoco_recorder.MujocoRecorder._render_global", return_value=rgb) as render_global:
            recorder = MujocoRecorder(session, directory, fps=5, decision_hold_s=0)
            session.recorder = recorder
            try:
                observation = session.get_observation()
                views = vlm_camera_views(observation, observation["agentview"], observation["wrist"])
                self.assertEqual([name for name, _ in views], ["Wrist", "Front", "Side", "Angled Wrist"])
                self.assertNotIn("lab_overview", observation)
                self.assertNotIn("lab_overview", observation["extra_views"])
                self.assertEqual(session.cameras, cameras)
                self.assertEqual(session.data.time, before)
                self.assertEqual(recorder.global_camera, "lab_overview")
                render_global.assert_called_once_with(448)
            finally:
                recorder.close()

    def test_orthographic_front_side_basis_and_depth_match_physical_moves(self):
        import mujoco
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cosine = np.sqrt(3) / 2
        cameras = {
            "front": ([1.4992304845413265, 0.03, 0.75],
                      [[0, 1, 0], [-0.5, 0, cosine], [cosine, 0, 0.5]]),
            "side": ([0.46, 1.0692304845413265, 0.75],
                     [[-1, 0, 0], [0, -0.5, cosine], [0, cosine, 0.5]]),
        }
        # Each vector is (image-right, image-down, toward-camera), in metres.
        # Orthographic projection must not divide image coordinates by depth.
        expected_directions = {
            "MV_FWD": {"front": [0, 0.5, cosine], "side": [-1, 0, 0]},
            "MV_RIGHT": {"front": [1, 0, 0], "side": [0, 0.5, cosine]},
            "MV_UP": {"front": [0, -cosine, 0.5], "side": [0, -cosine, 0.5]},
        }
        for config in ("robot_mujoco.yaml", "robot_mujoco_plug.yaml"):
            with self.subTest(config=config):
                cfg = load_yaml(ROOT / "configs" / config)
                session = MujocoSession(cfg)
                try:
                    controller = make_controller(session, cfg)
                    self.assertEqual(session.table_height_m, 0.0)
                    self.assertEqual(controller.z_floor_m, 0.025)
                    for name, (position, basis) in cameras.items():
                        camera_id = session.model.camera(name).id
                        self.assertEqual(session.model.cam_projection[camera_id],
                                         mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
                        self.assertAlmostEqual(session.model.cam_fovy[camera_id], 0.6)
                        np.testing.assert_allclose(session.data.camera(name).xpos, position, atol=1e-8)
                        np.testing.assert_allclose(session.data.camera(name).xmat.reshape(3, 3).T,
                                                   basis, atol=1e-8)
                    self.assertEqual(session.model.cam_projection[session.model.camera("wrist").id],
                                     mujoco.mjtProjection.mjPROJ_PERSPECTIVE)

                    def image_and_depth(name, point):
                        camera = session.data.camera(name)
                        local = camera.xmat.reshape(3, 3).T @ (point - camera.xpos)
                        return local * [1, -1, 1]

                    for forward, reverse in (("MV_FWD", "MV_BACK"),
                                             ("MV_RIGHT", "MV_LEFT"), ("MV_UP", "MV_DOWN")):
                        for token, sign in ((forward, 1), (reverse, -1)):
                            before = {name: image_and_depth(name, session.get_ee_pose()[:3])
                                      for name in cameras}
                            result = controller.step(token)
                            for name in cameras:
                                delta = image_and_depth(name, session.get_ee_pose()[:3]) - before[name]
                                expected = np.asarray(expected_directions[forward][name]) * sign * result.step_m
                                # Match the simulator's existing 3 mm tracking tolerance;
                                # exact axis isolation is established by the camera basis above.
                                np.testing.assert_allclose(delta, expected, atol=0.003,
                                                           err_msg=f"{token} in {name} ({config})")
                                for measured, component in zip(delta, expected):
                                    if component:
                                        self.assertGreater(measured * component, 0, f"{token} in {name}")
                finally:
                    session.close()

    def test_all_three_step_sizes_physically_move_the_requested_distance(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        controller = make_controller(session, cfg, variable_step=True)
        for token, visible, distance, kind in (
            ("MV_FWD", False, 0.10, "large"), ("MV_BACK", False, 0.10, "large"),
            ("MV_DOWN", True, 0.10, "large"),
            ("MV_FWD", False, 0.05, "coarse"), ("MV_BACK", False, 0.05, "coarse"),
            ("MV_DOWN", True, 0.05, "coarse"),
            ("MV_FWD", True, 0.02, "fine"), ("MV_BACK", None, 0.02, "fine"),
        ):
            before = session.get_ee_pose()[:3]
            result = controller.step(token, target_in_wrist=visible)
            self.assertEqual((result.step_m, result.step_kind), (distance, kind))
            np.testing.assert_allclose(session.get_ee_pose()[:3] - before,
                                       controller.move_vectors[token] * distance, atol=0.003)

    def test_physical_success_stops_before_another_model_request(self):
        import mujoco
        from core.config import load_yaml
        from core.launch import make_runner
        from core.prompting.prompt_loader import load_prompt_dir
        from core.record.episode_logger import EpisodeLogger
        from core.sim.mujoco_session import MujocoSession
        from core.vlm.vlm_client import VLMResponse
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.update(max_steps=3, camera_resolution=128)
        cfg["plugins"]["subgoal"] = False
        client = Mock()
        answer = {"decision": "MV_UP", "reasoning": "Retreat above the placed cube."}
        client.complete_json.return_value = VLMResponse("", json.dumps(answer), {"json": answer})
        with TemporaryDirectory() as directory, patch("mujoco.Renderer") as renderer:
            renderer.return_value.render.return_value = np.zeros((128, 128, 3), np.uint8)
            session = MujocoSession(cfg)
            try:
                # Physical fixture: cube already placed; the arm still needs to retreat.
                session.data.joint("cube_joint").qpos[:3] = [0.48, 0.16, 0.03]
                mujoco.mj_forward(session.model, session.data)
                pose = session.get_ee_pose()
                pose[2] = 0.075
                session.update_desired_ee_pose(pose)
                self.assertFalse(session.check_success())
                logger = EpisodeLogger(directory, task_id=0, record_video=False)
                runner = make_runner(cfg, load_prompt_dir(ROOT / "prompts"), client, session,
                                     make_controller(session, cfg), logger, False)
                result = runner.run()
                self.assertTrue(session.check_success())
                self.assertTrue(result.success)
                self.assertEqual(result.end_reason, "task_success")
                self.assertEqual(result.steps, 1)
                self.assertEqual(client.complete_json.call_count, 1)
            finally:
                session.close()

    def test_gripper_noops_are_fed_back_without_claiming_success(self):
        from core.config import load_yaml
        from core.launch import make_runner
        from core.prompting.prompt_loader import load_prompt_dir
        from core.record.episode_logger import EpisodeLogger
        from core.sim.mujoco_session import MujocoSession
        from core.vlm.vlm_client import VLMResponse
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.update(max_steps=3, camera_resolution=128)
        cfg["plugins"]["subgoal"] = False
        client = Mock()
        client.complete_json.side_effect = [
            VLMResponse("", json.dumps(answer), {"json": answer})
            for answer in ({"decision": "RELEASE", "reasoning": "Open the fingers."},
                           {"decision": "RELEASE", "reasoning": "Open the fingers."},
                           {"decision": "DONE", "reasoning": "The fingers are already open."})
        ]
        with TemporaryDirectory() as directory, patch("mujoco.Renderer") as renderer:
            renderer.return_value.render.return_value = np.zeros((128, 128, 3), np.uint8)
            session = MujocoSession(cfg)
            try:
                logger = EpisodeLogger(directory, task_id=0, record_video=False)
                runner = make_runner(cfg, load_prompt_dir(ROOT / "prompts"), client, session,
                                     make_controller(session, cfg), logger, False)
                result = runner.run()
                prompt = client.complete_json.call_args_list[1].args[0]
                self.assertIn("RELEASE(no-op)", prompt)
                self.assertIn("gripper width is 8.0 cm", prompt)
                steps = json.loads((logger.run_dir / "steps.json").read_text())
                self.assertTrue(steps[0]["noop"] and steps[1]["noop"])
                self.assertEqual(client.complete_json.call_count, 3)
                self.assertFalse(result.success, "Open fingers alone must not count as task success")
            finally:
                session.close()

    def test_variable_step_flag_wires_prompt_and_physical_distance(self):
        from core.config import load_yaml
        from core.vlm.vlm_client import VLMResponse
        from scripts.run_mujoco import main, make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.update(max_steps=1, camera_resolution=128, recording={"fps": 5, "decision_hold_s": 0})
        cfg["plugins"].update(subgoal=False, variable_step=True)
        client = Mock()
        initial = []

        def build_controller(session, config, **kwargs):
            controller = make_controller(session, config, **kwargs)
            initial.append(controller.target_pose[:3].copy())
            return controller

        cases = (
            (False, False, 0.1, "MV_FWD", 0.02),  # No flag: fixed even if YAML enables the plugin.
            (True, True, 1.0, "MV_FWD", 0.02),
            (True, False, 1.0, "MV_FWD", 0.10),
            (True, True, 0.1, "MV_FWD", 0.02),    # Height must not coarsen visible alignment.
            (True, True, 1.0, "MV_UP", 0.10),
            (True, None, 1.0, "MV_FWD", 0.02),
            (True, None, 0.1, "MV_FWD", 0.02),    # Missing metadata at height: the reported loop.
            (True, True, 0.1, "MV_DOWN", 0.10),
            (True, "false", 0.1, "MV_FWD", 0.02), # A string is not a boolean visibility signal.
        )
        with TemporaryDirectory() as directory, patch("mujoco.Renderer") as renderer, \
                patch("scripts.run_mujoco.make_vlm_client", return_value=client), \
                patch("scripts.run_mujoco.make_controller", side_effect=build_controller), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
            renderer.return_value.render.return_value = np.zeros((128, 128, 3), np.uint8)
            config_path = Path(directory) / "config.yaml"
            for i, (enabled, visible, height, token, distance) in enumerate(cases):
                with self.subTest(enabled=enabled, visible=visible, height=height, token=token):
                    cfg["high_above_table_m"] = height
                    config_path.write_text(yaml.safe_dump(cfg))
                    answer = {"decision": token, "reasoning": "Move toward the target."}
                    if visible is not None:
                        answer["target_in_wrist"] = visible
                    client.reset_mock()
                    client.complete_json.return_value = VLMResponse("", json.dumps(answer), {"json": answer})
                    output = Path(directory) / f"case_{i}"
                    args = ["--robot-config", str(config_path), "--log-dir", str(output),
                            "--vlm-backend", "ollama"]
                    if enabled:
                        args.append("--variable-step")
                    if i == 2:
                        args.append("--record")
                    self.assertEqual(main(args), 1)
                    self.assertEqual(client.complete_json.call_count, 1)
                    prompt = client.complete_json.call_args.args[0]
                    self.assertEqual("WRIST CHECK:" in prompt, enabled)
                    self.assertEqual("large steps ~10 cm" in prompt, enabled)
                    schema = client.complete_json.call_args.kwargs["schema"]
                    self.assertEqual("target_in_wrist" in schema["required"], enabled)
                    step = json.loads(next(output.rglob("steps.jsonl")).read_text().strip())
                    delta = np.array([distance, 0, 0]) if token == "MV_FWD" else np.array([
                        0, 0, distance if token == "MV_UP" else -distance,
                    ])
                    np.testing.assert_allclose(np.asarray(step["eef"]) - initial[-1], delta, atol=0.003)
                    if enabled:
                        self.assertEqual(step["target_in_wrist"], visible if isinstance(visible, bool) else None)
                        self.assertEqual(step["step_kind"], {0.02: "fine", 0.05: "coarse", 0.10: "large"}[distance])
                        self.assertEqual(step["step_cm"], distance * 100)
                    else:
                        self.assertNotIn("step_kind", step)
                    if i == 2:
                        events = [json.loads(line) for line in next(output.rglob("annotations.jsonl")).read_text().splitlines()]
                        result = next(e["result"] for e in events if e["event"] == "step_complete")
                        self.assertEqual((result["step_kind"], result["step_cm"]), ("large", 10))

            # The offline CLI checks every large axis, including immediate reversals.
            cfg["high_above_table_m"] = 0.1
            config_path.write_text(yaml.safe_dump(cfg))
            output = Path(directory) / "adaptive_smoke"
            self.assertEqual(main(["--robot-config", str(config_path), "--log-dir", str(output),
                                   "--smoke-test", "--variable-step", "--record"]), 0)
            events = [json.loads(line) for line in next(output.rglob("annotations.jsonl")).read_text().splitlines()]
            moves = [e for e in events if e["event"] == "step_complete"][:6]
            self.assertTrue(all(e["result"]["step_cm"] == 10 for e in moves))
            self.assertGreater(moves[0]["sim_time_s"], 1.8, "Large moves must get a longer smooth ramp")

            cfg["coarse_step_m"] = 0
            config_path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "coarse_step_m"):
                main(["--robot-config", str(config_path), "--variable-step", "--smoke-test"])

    def test_cli_video_recording_is_opt_in_for_runs_and_smoke_tests(self):
        from core.config import load_yaml
        from core.vlm.vlm_client import VLMResponse
        from scripts.run_mujoco import main

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.update(max_steps=1, camera_resolution=128, recording={"fps": 5, "decision_hold_s": 0})
        cfg["plugins"]["subgoal"] = False
        client = Mock()
        answer = {"decision": "MV_UP", "reasoning": "Lift above the table."}
        client.complete_json.return_value = VLMResponse("", json.dumps(answer), {"json": answer})
        with TemporaryDirectory() as directory, patch("mujoco.Renderer") as renderer, \
                patch("scripts.run_mujoco.make_vlm_client", return_value=client), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
            renderer.return_value.render.return_value = np.zeros((128, 128, 3), np.uint8)
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            for smoke in (False, True):
                for record in (False, True):
                    with self.subTest(smoke=smoke, record=record):
                        output = Path(directory) / f"smoke_{smoke}_record_{record}"
                        args = ["--robot-config", str(config_path), "--log-dir", str(output)]
                        if smoke:
                            args.append("--smoke-test")
                        if record:
                            args.append("--record")
                        self.assertEqual(main(args), 0 if smoke else 1)
                        videos = {path.name for path in output.rglob("*.mp4")}
                        if record:
                            self.assertEqual({"side.mp4", "wrist.mp4", "front.mp4", "combined.mp4"}, videos)
                            self.assertTrue(list(output.rglob("annotations.jsonl")))
                        else:
                            self.assertFalse(videos)
                            self.assertFalse(list(output.rglob("annotations.jsonl")))
                        if not smoke:
                            summary = json.loads(next(output.rglob("summary.json")).read_text())
                            self.assertEqual(bool(summary["video_path"]), record)
                            self.assertTrue(list(output.rglob("steps.jsonl")))

    def test_smooth_arm_and_gripper_motion_is_visible_and_paced(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        controller = make_controller(session, cfg)
        frames = []
        session._viewer = Mock()
        session._viewer.is_running.return_value = True
        session._viewer.sync.side_effect = lambda: frames.append(
            (session.get_ee_pose().copy(), session.data.ctrl.copy())
        )
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        start_pose = session.get_ee_pose()
        start_ctrl = session.data.ctrl.copy()
        with patch("core.sim.mujoco_session.time.monotonic", side_effect=lambda: clock[0]), \
                patch("core.sim.mujoco_session.time.sleep", side_effect=sleep):
            controller.step("MV_FWD")
            self.assertGreater(len(frames), 20, "GUI must show intermediate physics states")
            targets = np.array([frame[1][session._actuators] for frame in frames])
            self.assertGreater(len(np.unique(targets, axis=0)), 20, "Targets must follow a trajectory")
            first_delta = np.linalg.norm(targets[0] - start_ctrl[session._actuators])
            total_delta = np.linalg.norm(targets[-1] - start_ctrl[session._actuators])
            self.assertLess(first_delta, total_delta / 10, "Do not jump to the final setpoint")
            travel = np.array([frame[0][0] - start_pose[0] for frame in frames])
            self.assertTrue(np.any((travel > 0.003) & (travel < 0.017)))
            self.assertAlmostEqual(travel[-1], 0.02, delta=0.003)
            self.assertAlmostEqual(clock[0], cfg["motion_s"] + cfg["settle_s"], places=6)
            frames.clear()
            session.control_gripper(True)
            widths = [frame[1][session._gripper] for frame in frames]
            self.assertTrue(any(1 < width < 254 for width in widths), "Gripper must ramp too")
            self.assertEqual(widths[-1], 0)
            self.assertLess(session.get_gripper_position()[0], cfg["empty_width_m"])
            self.assertAlmostEqual(clock[0], 2 * (cfg["motion_s"] + cfg["settle_s"]), places=6)
            session._viewer.is_running.return_value = False
            before_close = session.data.time
            with self.assertRaises(KeyboardInterrupt):
                session.control_gripper(False)
            self.assertEqual(session.data.time, before_close, "Closing the viewer must stop immediately")

    def test_runner_records_the_decision_before_motion(self):
        from core.config import load_yaml
        from core.launch import make_runner
        from core.prompting.prompt_loader import load_prompt_dir
        from core.record.episode_logger import EpisodeLogger
        from core.record.mujoco_recorder import MujocoRecorder
        from core.sim.mujoco_session import MujocoSession
        from core.vlm.vlm_client import VLMResponse
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.update(max_steps=1, camera_resolution=128)
        cfg["plugins"]["subgoal"] = False
        client = Mock()
        answer = {"decision": "MV_UP", "reasoning": "Lift above the table."}
        client.complete_json.return_value = VLMResponse("", json.dumps(answer), {"json": answer})
        # Use real physics and runner wiring; replace only the OpenGL rasterizer.
        with TemporaryDirectory() as directory, patch("mujoco.Renderer") as renderer:
            renderer.return_value.render.return_value = np.zeros((128, 128, 3), np.uint8)
            session = MujocoSession(cfg)
            try:
                controller = make_controller(session, cfg)
                before = session.get_ee_pose()[2]
                logger = EpisodeLogger(directory, task_id=0)
                session.recorder = MujocoRecorder(session, logger.run_dir / "videos", fps=10, decision_hold_s=0)
                runner = make_runner(cfg, load_prompt_dir(ROOT / "prompts"), client,
                                     session, controller, logger, False)
                result = runner.run()
                self.assertGreater(session.get_ee_pose()[2], before + 0.015)
                events = [json.loads(line) for line in session.recorder.paths["annotations"].read_text().splitlines()]
                decision = next(e for e in events if e["event"] == "decision")
                completed = next(e for e in events if e["event"] == "step_complete")
                self.assertEqual(decision["sim_time_s"], 0)
                self.assertEqual(decision["decision"], "MV_UP")
                self.assertEqual(json.loads(decision["vlm_output"]), answer)
                self.assertAlmostEqual(completed["sim_time_s"], cfg["motion_s"] + cfg["settle_s"])
                self.assertEqual(result.video_path, str(session.recorder.paths["combined"]))
                summary = json.loads((logger.run_dir / "summary.json").read_text())
                self.assertEqual(summary["recordings"]["annotations"], str(session.recorder.paths["annotations"]))
            finally:
                session.close()

    def test_physical_pick_place_and_floor(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        controller = make_controller(session, cfg)
        self.assertFalse(session.check_success())

        def move_to(position):
            for _ in range(80):
                error = np.asarray(position) - controller.target_pose[:3]
                if np.max(np.abs(error)) < 1e-5:
                    return
                axis = int(np.argmax(np.abs(error)))
                token = (("MV_BACK", "MV_FWD"), ("MV_LEFT", "MV_RIGHT"),
                         ("MV_DOWN", "MV_UP"))[axis][int(error[axis] > 0)]
                controller.step(token, step_override_m=min(abs(error[axis]), controller.step_m))
            self.fail("Scripted atomic actions did not reach target")

        # Oracle positions are used only in this test; the deployed VLM sees RGB.
        cube = session.data.body("cube").xpos.copy()
        move_to([cube[0], cube[1], 0.16])
        move_to([cube[0], cube[1], cfg["z_floor_m"]])
        controller.step("MV_DOWN")
        self.assertGreaterEqual(controller.target_pose[2], cfg["z_floor_m"])
        controller.step("GRASP")
        self.assertTrue(controller.gripper_closed)
        move_to([cube[0], cube[1], 0.16])
        self.assertGreater(session.data.body("cube").xpos[2], 0.12)
        target = session.data.body("target").xpos.copy()
        move_to([target[0], target[1], 0.16])
        self.assertFalse(session.check_success(), "hovering over the pad is not success")
        move_to([target[0], target[1], 0.035])
        controller.step("RELEASE")
        move_to([target[0], target[1], 0.16])
        self.assertTrue(session.check_success())


if __name__ == "__main__":
    unittest.main()
