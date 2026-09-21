"""MuJoCo runner recording flag wiring, without rendering or API calls."""
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from core.v0_types import EpisodeResult


@unittest.skipUnless(importlib.util.find_spec("mujoco"),
                     "MuJoCo entrypoint requires its optional dependency")
class MujocoRecordingCliTests(unittest.TestCase):
    def test_normal_recording_flags_control_config_and_recorder(self):
        from scripts.run_mujoco import main

        for flags, expected_record, expected_global in (
            ([], False, False),
            (["--record"], True, False),
            (["--record-global"], True, True),
            (["--record", "--record-global"], True, True),
        ):
            with self.subTest(flags=flags), TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                    patch("scripts.run_mujoco.load_secrets_env"), \
                    patch("scripts.run_mujoco.MujocoSession") as session_factory, \
                    patch("scripts.run_mujoco.make_controller",
                          return_value=Mock(cartesian_actions=None)), \
                    patch("scripts.run_mujoco.make_vlm_client") as client_factory, \
                    patch("scripts.run_mujoco.EpisodeLogger") as logger_factory, \
                    patch("scripts.run_mujoco.make_runner") as runner_factory, \
                    patch("scripts.run_mujoco.MujocoRecorder") as recorder_factory, \
                    patch("scripts.run_mujoco.load_prompt_dir",
                          return_value={"common_context": "", "controller_mujoco_prompt": ""}), \
                    patch("builtins.print"):
                session_factory.return_value.recorder = None
                logger_factory.return_value.run_dir = Path(directory)
                runner_factory.return_value.run.return_value = EpisodeResult(
                    False, 0, "test_finished", "", directory,
                )

                self.assertEqual(main(["--vlm-backend", "openai", "--max-steps", "1",
                                       "--log-dir", directory, *flags]), 1)

                cfg = client_factory.call_args.args[1]
                logger_factory.return_value.write_metadata.assert_called_once()
                metadata = logger_factory.return_value.write_metadata.call_args.args[0]
                self.assertEqual(metadata["recording_enabled"], expected_record)
                self.assertEqual(cfg.get("recording", {}).get("global_video", False),
                                 expected_global)
                if expected_record:
                    recorder_factory.assert_called_once()
                    if expected_global:
                        self.assertTrue(recorder_factory.call_args.kwargs["global_video"])
                    else:
                        self.assertNotIn("global_video", recorder_factory.call_args.kwargs)
                else:
                    recorder_factory.assert_not_called()

    def test_smoke_recording_flags_pass_recording_config_to_recorder(self):
        from scripts.run_mujoco import main

        class SmokeController:
            cartesian_actions = None
            move_vectors = ()
            gripper_closed = False

            def step(self, token, target_in_wrist=None):
                return SimpleNamespace(kind="gripper", step_kind="fine", step_m=0,
                                       grasp_empty=True, rotation_deg=None)

        class SmokeSession:
            cameras = ("front",)
            recorder = None

            def get_observation(self):
                return {"front": np.zeros((2, 2, 3), dtype=np.uint8)}

            def get_ee_pose(self):
                return np.zeros(7)

            def close(self):
                pass

        for flags, expected_record, expected_global in (
            ([], False, False),
            (["--record"], True, False),
            (["--record-global"], True, True),
            (["--record", "--record-global"], True, True),
        ):
            with self.subTest(flags=flags), TemporaryDirectory() as directory, \
                    patch("scripts.run_mujoco.load_secrets_env"), \
                    patch("scripts.run_mujoco.MujocoSession", return_value=SmokeSession()), \
                    patch("scripts.run_mujoco.make_controller", return_value=SmokeController()), \
                    patch("scripts.run_mujoco.MujocoRecorder") as recorder_factory, \
                    patch("scripts.run_mujoco.save_png"), \
                    patch("builtins.print"):
                result = main(["--smoke-test", "--log-dir", directory, *flags])
                self.assertEqual(result, 0)
                if expected_record:
                    recorder_factory.assert_called_once()
                    if expected_global:
                        self.assertTrue(recorder_factory.call_args.kwargs["global_video"])
                    else:
                        self.assertNotIn("global_video", recorder_factory.call_args.kwargs)
                else:
                    recorder_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
