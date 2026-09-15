"""Physical pick/place check. Requires `bash scripts/setup.sh mujoco`; no cloud calls."""
import importlib.util
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
    def test_side_wrist_front_reach_planner_controller_and_saved_observations(self):
        from PIL import Image
        from core.config import load_yaml
        from core.vlm.vlm_client import VLMResponse
        from scripts.run_mujoco import main

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        cfg.update(max_steps=1, camera_resolution=128)
        frames = {name: np.full((128, 128, 3), value, np.uint8)
                  for name, value in (("side", 40), ("wrist", 100), ("front", 200))}
        plan = {"subgoals": [{"id": "lift", "motion": "LIFT", "target": "cube",
                              "affordance": "cube", "description": "Lift the cube.",
                              "completion": "Cube above table."}]}
        action = {"decision": "MV_UP", "reasoning": "Lift above the table."}
        client = Mock()
        client.complete_json.side_effect = [VLMResponse("", json.dumps(v), {"json": v}) for v in (plan, action)]
        with TemporaryDirectory() as directory, \
                patch("core.sim.mujoco_session.MujocoSession.render", side_effect=frames.__getitem__), \
                patch("scripts.run_mujoco.make_vlm_client", return_value=client) as factory, \
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            self.assertEqual(main(["--robot-config", str(config_path), "--log-dir", directory,
                                   "--model", "test-model:cloud"]), 1)
            self.assertEqual(factory.call_args.args[1]["vlm"]["model"], "test-model:cloud")
            self.assertEqual(factory.call_args.args[1]["vlm_backend"], "gemini")
            self.assertEqual(factory.call_args.args[1]["vlm"]["provider"], "gemini")
            self.assertEqual(factory.call_args.args[1]["vlm"]["api_key"], "test-key")
            self.assertEqual(client.complete_json.call_count, 2)
            for call in client.complete_json.call_args_list:
                np.testing.assert_array_equal(call.args[1], frames["side"])
                self.assertEqual(len(call.kwargs["wrist_image"]), 2)
                for sent, name in zip(call.kwargs["wrist_image"], ("wrist", "front")):
                    np.testing.assert_array_equal(sent, frames[name])
                self.assertNotIn("AgentView", call.args[0])
                self.assertIn("A and C mix height and table-plane depth", call.args[0])
            controller_prompt = client.complete_json.call_args_list[-1].args[0]
            self.assertIn("24.0 cm above the table", controller_prompt)
            self.assertIn("down-right in A, down-left in C", controller_prompt)
            self.assertNotIn("MV_DOWN first", controller_prompt)
            run = next(Path(directory).rglob("steps.jsonl")).parent
            self.assertEqual({p.name for p in (run / "images").iterdir()}, set(frames))
            for name, frame in frames.items():
                np.testing.assert_array_equal(np.asarray(Image.open(run / "images" / name / "0000.png")), frame)
            np.testing.assert_array_equal(np.asarray(Image.open(run / "planner_front.png")), frames["front"])
            self.assertFalse(list(run.rglob("*.mp4")))

    def test_angled_camera_directions_match_physical_moves(self):
        from core.config import load_yaml
        from core.sim.mujoco_session import MujocoSession
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco.yaml")
        session = MujocoSession(cfg)
        self.addCleanup(session.close)
        controller = make_controller(session, cfg)

        def project(name, point):
            camera = session.data.camera(name)
            local = camera.xmat.reshape(3, 3).T @ (point - camera.xpos)
            return np.array([local[0], -local[1]]) / -local[2]

        for name in ("side", "front"):
            self.assertEqual(session.model.cam_projection[session.model.camera(name).id],
                             session.model.cam_projection[session.model.camera("wrist").id])
        self.assertEqual(session.table_height_m, 0.0)
        self.assertEqual(controller.z_floor_m, 0.025)
        for token, inverse, signs in (
            ("MV_FWD", "MV_BACK", {"side": (1, 1), "front": (-1, 1)}),
            ("MV_RIGHT", "MV_LEFT", {"side": (1, -1), "front": (1, 1)}),
            ("MV_UP", "MV_DOWN", {"side": (0, -1), "front": (0, -1)}),
        ):
            before = {name: project(name, session.get_ee_pose()[:3]) for name in signs}
            controller.step(token)
            for name, expected in signs.items():
                delta = project(name, session.get_ee_pose()[:3]) - before[name]
                for value, sign in zip(delta, expected):
                    if sign:
                        self.assertGreater(value * sign, 0, f"{token} in {name}")
            controller.step(inverse)

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
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
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
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
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
