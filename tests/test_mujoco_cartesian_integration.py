"""Offline coverage of VLM -> one Cartesian action -> shared rollout logging."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from core.config import load_yaml
from core.sim.mujoco_depth import depth_to_text
from core.vlm.roles import CONTROLLER_TOKENS, ControllerAgent
from core.vlm.vlm_client import VLMParseError, VLMResponse, recover_allowed_token


ROOT = Path(__file__).resolve().parents[1]


class _PoseSession:
    """Instant pose feedback; physics is covered separately by MuJoCo tests."""

    control_mode = "mujoco"
    table_height_m = 0.0
    recorder = None

    def __init__(self, cfg, **kwargs):
        self.robot = self
        self.scene = cfg.get("scene", "pick_place")
        self.cameras = ("side", "wrist", "front", "wrist_insert", "wrist_depth") if self.scene == "plug_insert" else ("side", "wrist", "front")
        self.pose = np.array([0.4, 0.0, 0.24, 0.0, 0.0, 0.0, 1.0])
        self.width = 0.08
        self.block_descent = False
        self.pose_commands = []
        self.frames = {
            name: np.full((16, 16, 3), value, dtype=np.uint8)
            for name, value in (("side", 40), ("wrist", 100), ("front", 200),
                                ("wrist_insert", 240), ("wrist_depth", 150))
            if name in self.cameras
        }

    def get_ee_pose(self):
        return self.pose.copy()

    def get_gripper_position(self):
        return np.array([self.width])

    def control_gripper(self, close):
        self.width = 0.025 if close else 0.08

    def update_desired_ee_pose(self, pose):
        target = np.asarray(pose, dtype=float).copy()
        if self.block_descent:
            target[2] = max(self.pose[2], target[2])
        self.pose = target
        self.pose_commands.append(self.pose.copy())

    def get_observation(self):
        obs = {
            "agentview": self.frames["front"],
            "wrist": self.frames["wrist"],
            "extra_views": {"side": self.frames["side"]},
            "primary_camera": "Front",
            "wrist_first": True,
            "ee_pose": self.get_ee_pose(),
            "gripper_width": self.width,
        }
        if self.scene == "plug_insert":
            obs["extra_views"]["wrist_insert"] = self.frames["wrist_insert"]
            obs["extra_views"]["wrist_depth"] = self.frames["wrist_depth"]
            obs["wrist_depth"] = self.frames["wrist_depth"]
            obs["wrist_depth_calibration"] = {"near_m": 0.0, "far_m": 0.30, "tcp_depth_m": 0.048}
            obs["wrist_depth_calibration"]["representation"] = "text"
            obs["wrist_depth_m"] = np.full((16, 16), self.pose[2] - 0.12, dtype=np.float32)
            obs["wrist_depth_text"] = depth_to_text(obs["wrist_depth_m"], obs["wrist_depth_calibration"])
            rotation = Rotation.from_quat(self.pose[3:])
            obs["insertion_camera_axes"] = {
                name: rotation.apply(vector).tolist() for name, vector in (
                    ("image_right", [0, -1, 0]),
                    ("image_down", [0.7488700551657237, 0, 0.6627168629785166]),
                    ("sightline", [-0.6627168629785166, 0, 0.7488700551657237]),
                )
            }
        return obs

    def check_success(self):
        return False

    def close(self):
        pass


def _answer(token):
    answer = {"decision": token, "reasoning": "Align the held plug with the socket."}
    return VLMResponse("", json.dumps(answer), {"json": answer})


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo entrypoint requires its optional dependency")
class MujocoCartesianIntegrationTests(unittest.TestCase):
    def _run(self, tokens, *, config="robot_mujoco_plug.yaml", block_descent=False, extra_args=()):
        from scripts.run_mujoco import main, make_controller

        cfg = load_yaml(ROOT / "configs" / config)
        cfg.update(max_steps=len(tokens), camera_resolution=16)
        cfg["v0"]["controller_prompt_log_every"] = 1
        cfg["plugins"].update(subgoal=False, recovery=False)
        client = Mock()
        client.complete_json.side_effect = [_answer(token) for token in tokens]
        session = _PoseSession(cfg)
        session.block_descent = block_descent
        controllers = []

        def build_controller(*args, **kwargs):
            controller = make_controller(*args, **kwargs)
            controller.step = Mock(wraps=controller.step)
            controllers.append(controller)
            return controller

        with TemporaryDirectory() as directory, \
                patch("scripts.run_mujoco.MujocoSession", return_value=session), \
                patch("scripts.run_mujoco.make_controller", side_effect=build_controller), \
                patch("scripts.run_mujoco.make_vlm_client", return_value=client), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GEMINI_API_KEY": "test-key", "OLLAMA_API_KEY": "test-key"}):
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(cfg))
            result = main(["--robot-config", str(path), "--log-dir", directory, *extra_args])
            run = next(Path(directory).rglob("steps.jsonl")).parent
            records = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
            summary = json.loads((run / "summary.json").read_text())
            prompt_files = {p.stem: p.read_text() for p in (run / "controller_prompts").glob("*.txt")}
            depth_files = {p.stem: p.read_text() for p in (run / "depth" / "wrist").glob("*.txt")}
            raw_depth_files = {p.stem: np.load(p, allow_pickle=False) for p in (run / "depth" / "wrist").glob("*.npy")}

        self.assertEqual(result, 1, "A model decision alone cannot certify physical success")
        self.assertEqual(client.complete_json.call_count, len(tokens))
        plug = session.scene == "plug_insert"
        order = ("front", "side", "wrist_insert") if plug else ("front", "side")
        for index, call in enumerate(client.complete_json.call_args_list):
            np.testing.assert_array_equal(call.args[1], session.frames["wrist"])
            old_sets = min(index, cfg.get("visual_history_steps", 0))
            expected_order = list(order) + list(("wrist", *order)) * old_sets
            self.assertEqual(len(call.kwargs["wrist_image"]), len(expected_order))
            for sent, name in zip(call.kwargs["wrist_image"], expected_order):
                np.testing.assert_array_equal(sent, session.frames[name])
            if old_sets:
                saved_prompt = prompt_files[f"{index:04d}"]
                self.assertIn(f"{(1 + len(order)) * (1 + old_sets)} media part(s)", saved_prompt)
                self.assertEqual(saved_prompt.count("[verified]"), (1 + len(order)) * old_sets)
                self.assertNotIn("MISMATCH", saved_prompt)
                self.assertNotIn("unknown camera", saved_prompt)
            labels = "Wrist (A), Front (B), Right Side (C)" + (", Angled Wrist (D)" if plug else "")
            self.assertIn(f"Images are ordered {labels}.", call.args[0])
            if plug:
                self.assertIn("TCP plane is 48.0 mm", call.args[0])
                self.assertIn("WRIST_DEPTH_MM", call.args[0])
                self.assertIn(depth_files[f"{index:04d}"], call.args[0])
                if index:
                    self.assertIn(depth_files[f"{index - 1:04d}"], call.args[0])
                self.assertEqual(raw_depth_files[f"{index:04d}"].shape, (16, 16))
                self.assertNotIn("Wrist Depth (E)", call.args[0])
            else:
                self.assertNotIn("Wrist Depth", call.args[0])
        client.complete_token.assert_not_called()
        self.assertEqual([call.args[0] for call in controllers[0].step.call_args_list],
                         ["RELEASE", *tokens])
        self.assertEqual([record["act"] for record in records], list(tokens))
        self.assertEqual(summary["steps"], len(tokens))
        self.assertFalse(summary["success"])
        return client, session, controllers[0], records

    def test_sized_translations_reach_executor_and_log_with_one_vlm_call(self):
        for size, distance in (("SMALL", 0.002), ("MEDIUM", 0.01), ("LARGE", 0.05)):
            with self.subTest(size=size):
                token = f"MV_FWD_{size}"
                client, session, controller, records = self._run([token])
                np.testing.assert_allclose(session.pose[:3], [0.4 + distance, 0.0, 0.24])
                self.assertEqual(len(session.pose_commands), 1)
                self.assertEqual(records[0]["step_kind"], size.lower())
                self.assertAlmostEqual(records[0]["step_cm"], distance * 100)
                schema = client.complete_json.call_args.kwargs["schema"]
                self.assertIn(token, schema["properties"]["decision"]["enum"])
                self.assertIn("ROT_Y_NEG_MEDIUM", schema["properties"]["decision"]["enum"])
                self.assertNotIn("target_in_wrist", schema["required"])

    def test_post_grasp_large_lift_executes_50mm_with_paired_depth_history(self):
        client, session, controller, records = self._run(["GRASP", "MV_UP_LARGE", "STOP"])
        self.assertAlmostEqual(session.pose[2], 0.29)
        self.assertAlmostEqual(records[1]["post_pose"][2] - records[1]["pre_pose"][2], 0.05)
        # STOP sends a hold setpoint, not another lift.
        self.assertEqual(len(session.pose_commands), 2)
        np.testing.assert_array_equal(session.pose_commands[0], session.pose_commands[1])
        prompt = client.complete_json.call_args_list[2].args[0]
        self.assertIn("BEFORE action MV_UP_LARGE", prompt)
        self.assertIn("measured TCP translation world [X,Y,Z] mm=[+0.00, +0.00, +50.00]", prompt)

    def test_all_rotation_axes_preserve_position_and_single_action_cadence(self):
        for axis, sign, size, degrees in (
            ("X", "POS", "SMALL", 2), ("Y", "NEG", "MEDIUM", -10),
            ("Z", "POS", "LARGE", 30),
        ):
            with self.subTest(axis=axis, size=size):
                token = f"ROT_{axis}_{sign}_{size}"
                client, session, controller, records = self._run([token])
                np.testing.assert_allclose(session.pose[:3], [0.4, 0.0, 0.24])
                expected = Rotation.from_euler(axis.lower(), degrees, degrees=True)
                np.testing.assert_allclose(Rotation.from_quat(session.pose[3:]).as_matrix(),
                                           expected.as_matrix(), atol=1e-8)
                self.assertEqual(len(session.pose_commands), 1)
                self.assertEqual(records[0]["step_kind"], size.lower())
                self.assertAlmostEqual(records[0]["rotation_deg"], abs(degrees))
                expected_vector = np.zeros(3)
                expected_vector["XYZ".index(axis)] = np.deg2rad(degrees)
                np.testing.assert_allclose(records[0]["rotation_vector_rad"], expected_vector,
                                           atol=1e-6)
                self.assertNotIn("step_cm", records[0])
                prompt = client.complete_json.call_args.args[0]
                self.assertIn(token, prompt)
                self.assertIn("their image axes are NOT fixed", prompt)
                self.assertIn("Current measured orientation", prompt)
                self.assertNotIn("TARGET below the grasp point -> MV_FWD", prompt)
                self.assertNotIn("below center means MV_FWD", prompt)

    def test_sized_moves_and_rotations_remain_in_the_next_decision_history(self):
        client, session, controller, records = self._run(
            ["MV_DOWN_SMALL", "ROT_Y_POS_MEDIUM", "MV_UP_SMALL"]
        )
        prompts = [call.args[0] for call in client.complete_json.call_args_list]
        self.assertIn("Recent moves, newest first: MV_DOWN_SMALL", prompts[1])
        self.assertIn("Recent moves, newest first: ROT_Y_POS_MEDIUM, MV_DOWN_SMALL", prompts[2])
        np.testing.assert_allclose(session.pose[:3], [0.4, 0.0, 0.24])
        self.assertEqual(len(session.pose_commands), 3)

    def test_angled_wrist_calibration_reaches_next_prompt_after_rotation(self):
        client, session, controller, records = self._run(["ROT_X_POS_LARGE", "STOP"])
        prompts = [call.args[0] for call in client.complete_json.call_args_list]
        initial = _PoseSession(load_yaml(ROOT / "configs/robot_mujoco_plug.yaml"))
        for prompt, snapshot in zip(prompts, (initial, session)):
            axes = snapshot.get_observation()["insertion_camera_axes"]
            for name, values in axes.items():
                vector = ", ".join(f"{value:+.3f}" for value in values)
                label = name.replace("_", "-")
                self.assertIn(f"Angled Wrist {label}=[{vector}]", prompt)
        self.assertNotEqual(initial.get_observation()["insertion_camera_axes"],
                            session.get_observation()["insertion_camera_axes"])
        self.assertIn("image center is NOT the grasp point", prompts[0])
        self.assertIn("no keyed yaw alignment is needed", prompts[0])

    def test_stalled_sized_descent_is_reported_and_rotation_clears_the_measurement(self):
        client, session, controller, records = self._run(
            ["MV_DOWN_SMALL", "ROT_Y_POS_SMALL", "STOP"], block_descent=True,
        )
        prompts = [call.args[0] for call in client.complete_json.call_args_list]
        self.assertIn("lowered 0.0 of 0.2 cm", prompts[1])
        self.assertNotIn("lowered 0.0 of 0.2 cm", prompts[2])
        self.assertAlmostEqual(records[0]["post_pose"][2], 0.24)
        self.assertAlmostEqual(records[0]["target_pose"][2], 0.238)

    def test_original_pick_place_config_keeps_fixed_translation_vocabulary(self):
        from interpreters.real_atomic_controller import RealAtomicController

        client, session, controller, records = self._run(["MV_FWD"], config="robot_mujoco.yaml")
        self.assertIs(type(controller), RealAtomicController)
        np.testing.assert_allclose(session.pose[:3], [0.42, 0.0, 0.24])
        schema = client.complete_json.call_args.kwargs["schema"]
        self.assertEqual(schema["properties"]["decision"]["enum"], list(CONTROLLER_TOKENS))

    def test_cartesian_flag_preserves_cube_task_and_requires_explicit_sizes(self):
        from scripts.run_mujoco import main

        client, session, controller, records = self._run(
            ["MV_FWD_SMALL"], config="robot_mujoco.yaml", extra_args=("--cartesian-motion",),
        )
        prompt = client.complete_json.call_args.args[0]
        self.assertIn("red cube", prompt)
        self.assertIn("blue pad", prompt)
        self.assertNotIn("shaft", prompt)
        self.assertNotIn("socket", prompt)
        self.assertNotIn("Angled Wrist", prompt)
        self.assertNotIn("insertion_camera_axes", session.get_observation())
        self.assertIn("Bare MV_* moves 20 mm", prompt)
        self.assertEqual(prompt.count("EXPLICIT CARTESIAN ACTIONS"), 1)
        self.assertEqual(records[0]["step_cm"], 0.2)
        with patch("scripts.run_mujoco.MujocoSession") as session_factory:
            with self.assertRaises(SystemExit) as error:
                main(["--cartesian-motion", "--variable-step", "--smoke-test"])
            self.assertEqual(error.exception.code, 2)
            session_factory.assert_not_called()

    def test_malformed_response_recovers_full_sized_token_without_prefix_matches(self):
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco_plug.yaml")
        actions = make_controller(_PoseSession(cfg), cfg).cartesian_actions
        client = Mock()
        agent = ControllerAgent(client, "{output_contract}", "",
                                cartesian_actions=actions)
        for token in ("MV_FWD_SMALL", "ROT_Z_NEG_MEDIUM"):
            with self.subTest(token=token):
                client.reset_mock()
                client.complete_json.side_effect = VLMParseError("truncated JSON", f'{{"decision":"{token}"')
                response = agent.decide("insert plug", {}, "none", "NONE", "OPEN", None)
                self.assertEqual(response.token, token)
                self.assertTrue(response.payload["recovered_from_malformed_json"])
                client.complete_json.assert_called_once()
                client.complete_token.assert_not_called()
        allowed = tuple(CONTROLLER_TOKENS) + tuple(actions.action_tokens())
        for malformed in ("MV_FWD_SMALL_EXTRA", "ROT_Y_NEG_HUGE", "ROT_Z_POS", "MV_DOWN_LARGER"):
            self.assertEqual(recover_allowed_token(malformed, allowed), "")
        client.reset_mock()
        client.complete_json.side_effect = None
        client.complete_json.return_value = _answer("ROT_Y_NEG_HUGE")
        client.complete_token.return_value = VLMResponse("STOP", "STOP", {})
        response = agent.decide("insert plug", {}, "none", "NONE", "OPEN", None)
        self.assertEqual(response.token, "STOP")
        client.complete_json.assert_called_once()
        client.complete_token.assert_called_once()


if __name__ == "__main__":
    unittest.main()
