"""Offline temporal observation/action wiring through the real single-action loop."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from core.config import load_yaml
from core.record.episode_logger import EpisodeLogger
from core.runners.motion_history import motion_history_effect
from core.runners.real import RealEpisodeRunner
from core.v0_types import Subgoal, V0Config
from core.vlm.vlm_client import VLMResponse
from interpreters.mujoco_atomic_controller import CartesianStepResult
from plugins.recovery.plugin import RecoveryDecision
from tests.test_mujoco_cartesian_integration import _PoseSession
from tests.test_real_runner_planning import _Planner


ROOT = Path(__file__).resolve().parents[1]


class _ReusedFrames(_PoseSession):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.observation_count = 0

    def get_observation(self):
        self.observation_count += 1
        # A camera may reuse its backing array for the next frame.
        for frame in self.frames.values():
            frame.fill(self.observation_count)
        return super().get_observation()


class _Decisions:
    last_prompt = ""

    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.calls = []

    def decide(self, **kwargs):
        self.calls.append({**kwargs, "pixel": int(kwargs["ctx"].wrist[0, 0, 0]),
                           "depth_text": kwargs["ctx"].obs.get("wrist_depth_text", "")})
        return VLMResponse(next(self.tokens), "", {})


def _stage(name):
    return Subgoal(name, "plug", "pin", name, "Align the pin with the bore.", "Aligned.")


class VisualHistoryRunnerTests(unittest.TestCase):
    def _runner(self, root, tokens, *, history=1, stages=None, steps=None):
        from scripts.run_mujoco import make_controller

        cfg = load_yaml(ROOT / "configs/robot_mujoco_plug.yaml")
        session = _ReusedFrames(cfg)
        controller = make_controller(session, cfg)
        controller.sync_from_robot()
        decider = _Decisions(tokens)
        runner = RealEpisodeRunner(
            session=session, controller=controller,
            planner=_Planner(stages) if stages else None,
            controls=SimpleNamespace(controller=decider),
            logger=EpisodeLogger(root, 0, record_video=False, primary_camera="front"),
            config=V0Config(70, 0, 1, controller_prompt_log_every=0),
            task="Insert the plug", gripper_color="black", max_steps=len(tokens) if steps is None else steps,
            loop_period_s=0, use_wrist_image=True, debug=False,
            visual_history_steps=history,
        )
        return runner, decider, session

    def _run(self, runner):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run()

    def test_snapshots_are_copied_bounded_and_paired_with_executed_motion(self):
        with TemporaryDirectory() as root:
            runner, decider, session = self._runner(
                root, ["MV_DOWN_SMALL", "ROT_Y_POS_MEDIUM", "MV_RIGHT_SMALL", "STOP"], history=2,
            )
            self._run(runner)
        self.assertEqual([len(call["visual_history"]) for call in decider.calls], [0, 1, 2, 2])
        history = decider.calls[-1]["visual_history"]
        self.assertEqual([entry.step_idx for entry in history], [1, 2])
        self.assertEqual([entry.action for entry in history], ["ROT_Y_POS_MEDIUM", "MV_RIGHT_SMALL"])
        for entry in history:
            self.assertEqual(len(entry.views), 4)
            self.assertEqual(entry.depth_text, decider.calls[entry.step_idx]["depth_text"])
            for _, frame in entry.views:
                self.assertTrue(np.all(frame == decider.calls[entry.step_idx]["pixel"]))
                self.assertFalse(np.shares_memory(frame, session.frames["wrist"]))
        self.assertIn("measured world rotation-vector deg=[+0.00, +10.00, +0.00]", history[0].effect)
        self.assertIn("measured TCP translation world [X,Y,Z] mm=[+0.00, +2.00, +0.00]", history[1].effect)

    def test_depth_history_keeps_source_array_across_movement_and_stage_boundary(self):
        with TemporaryDirectory() as root:
            runner, decider, _ = self._runner(
                root, ["MV_UP_LARGE", "DONE", "STOP"], history=2,
                stages=[_stage("LIFT"), _stage("TRANSPORT")],
            )
            self._run(runner)
            self.assertNotEqual(decider.calls[0]["depth_text"], decider.calls[1]["depth_text"])
            for index, entry in enumerate(decider.calls[2]["visual_history"]):
                self.assertEqual(entry.depth_text, decider.calls[index]["depth_text"])
                saved = (runner.logger.run_dir / "depth" / "wrist" / f"{index:04d}.txt").read_text()
                self.assertEqual(saved, entry.depth_text)
            self.assertEqual(decider.calls[2]["ctx"].subgoal.motion, "TRANSPORT")

    def test_default_one_and_disabled_memory(self):
        for count in (0, 1):
            with self.subTest(count=count), TemporaryDirectory() as root:
                runner, decider, _ = self._runner(root, ["MV_DOWN_SMALL", "STOP", "STOP"], history=count)
                self._run(runner)
                if count:
                    self.assertEqual([e.step_idx for e in decider.calls[-1]["visual_history"]], [1])
                else:
                    self.assertTrue(all("visual_history" not in call for call in decider.calls))

    def test_history_crosses_alignment_done_into_insertion(self):
        with TemporaryDirectory() as root:
            runner, decider, _ = self._runner(
                root, ["MV_RIGHT_SMALL", "DONE", "MV_DOWN_SMALL"], history=2,
                stages=[_stage("ALIGN"), _stage("INSERT")],
            )
            self._run(runner)
        call = decider.calls[-1]
        self.assertEqual(call["ctx"].subgoal.motion, "INSERT")
        self.assertEqual(call["recent_moves"], "none")
        self.assertEqual([e.action for e in call["visual_history"]], ["MV_RIGHT_SMALL", "DONE"])
        self.assertTrue(all(e.stage == "ALIGN" for e in call["visual_history"]))

    def test_recovery_override_is_the_historical_action(self):
        with TemporaryDirectory() as root:
            runner, decider, _ = self._runner(root, ["STOP"], steps=2)
            recovery = RecoveryDecision("test", "test", token="MV_UP_SMALL")
            with patch.object(runner, "_recovery_before_decision", side_effect=[recovery, None]):
                self._run(runner)
        self.assertEqual(len(decider.calls), 1)
        entry = decider.calls[0]["visual_history"][0]
        self.assertEqual(entry.action, "MV_UP_SMALL")
        self.assertIn("mm=[+0.00, +0.00, +2.00]", entry.effect)

    def test_recovery_release_and_rollback_remain_in_history(self):
        with TemporaryDirectory() as root:
            runner, decider, session = self._runner(
                root, ["GRASP", "STOP"], stages=[_stage("GRASP"), _stage("INSERT")],
            )
            recovery = RecoveryDecision(
                "empty_grasp", "empty", release=True, rollback_index=0,
                reset_history=True, grasp_empty=True, prompt_note="Empty grasp; recheck alignment.",
            )
            with patch.object(runner, "_recovery_after_step", side_effect=[recovery, None]):
                self._run(runner)
        entry = decider.calls[1]["visual_history"][0]
        self.assertEqual(entry.action, "GRASP then recovery RELEASE")
        self.assertIn("recovery RELEASE:", entry.effect)
        self.assertIn("controller gripper state=OPEN", entry.effect)
        self.assertIn("Empty grasp", entry.effect)
        self.assertIn("GRASP(empty)", decider.calls[1]["recent_moves"])
        self.assertEqual(session.width, 0.08)

    def test_blocked_descent_reports_requested_and_actual_separately(self):
        with TemporaryDirectory() as root:
            runner, decider, session = self._runner(root, ["MV_DOWN_SMALL", "STOP"])
            session.block_descent = True
            self._run(runner)
        effect = decider.calls[-1]["visual_history"][0].effect
        self.assertIn("Requested TCP translation world [X,Y,Z] mm=[+0.00, +0.00, -2.00]", effect)
        self.assertIn("measured TCP translation world [X,Y,Z] mm=[+0.00, +0.00, +0.00]", effect)
        self.assertIn("does not establish object motion", effect)

    def test_new_episode_and_replan_clear_history(self):
        with TemporaryDirectory() as root:
            stages = [_stage("ALIGN"), _stage("REASON"), _stage("INSERT")]
            runner, decider, _ = self._runner(root, ["DONE", "STOP"], steps=3, stages=stages)
            runner.deepplan_plugin = SimpleNamespace(
                is_pivot=lambda sg: sg.motion == "REASON",
                resolve=lambda **kw: SimpleNamespace(resolved_subgoals=[_stage("INSERT")]),
            )
            with patch.object(runner, "_deepplan_plan_record", return_value={}):
                self._run(runner)
            self.assertEqual(decider.calls[-1]["visual_history"], ())
            runner.planner = None
            runner.deepplan_plugin = None
            runner.logger = EpisodeLogger(Path(root) / "second", 0, record_video=False)
            runner.controls.controller = new_decider = _Decisions(["STOP"])
            runner.max_steps = 1
            self._run(runner)
            self.assertEqual(new_decider.calls[0]["visual_history"], ())

    def test_rotation_effect_is_world_frame_and_missing_pose_is_not_zero(self):
        initial = Rotation.from_euler("z", 90, degrees=True)
        final = Rotation.from_euler("x", 2, degrees=True) * initial
        result = CartesianStepResult(
            "ROT_X_POS_SMALL", "rotate", intended_rotation_rad=np.deg2rad([2, 0, 0]),
            pre_pose=np.r_[np.zeros(3), initial.as_quat()],
            post_pose=np.r_[np.zeros(3), final.as_quat()],
        )
        self.assertIn("measured world rotation-vector deg=[+2.00, +0.00, +0.00]", motion_history_effect(result))
        result.post_pose = None
        self.assertIn("measured TCP motion unavailable", motion_history_effect(result))


if __name__ == "__main__":
    unittest.main()
