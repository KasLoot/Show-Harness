"""Offline replay validation, command ordering, pose checks, and CLI safety."""
import builtins
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.sim.mujoco_replay import (
    execute_replay, load_replay_plan, pose_error, validate_tolerances,
)
from scripts.mujoco.replay import hold_final_view, main


POSE = [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
CONFIG = {
    "scene": "plug_insert", "reset_qpos": [0.0] * 7, "fine_step_m": 0.002,
    "empty_width_m": 0.005, "open_width_m": 0.065, "z_floor_m": 0.025,
    "plugins": {"variable_step": False}, "cartesian_motion": {},
    "vlm": {"api_key": "secret-must-not-be-printed"},
}


def record(index=0, token="DONE", *, pre=None, post=None, **fields):
    return {"i": index, "act": token, "stage": "INSERT",
            "pre_pose": POSE.copy() if pre is None else pre,
            "post_pose": POSE.copy() if post is None else post, **fields}


class Session:
    def __init__(self):
        self.pose = np.asarray(POSE, float).copy()
        self.recorder = Mock()

    def get_ee_pose(self):
        return self.pose.copy()

    def check_success(self):
        return False


class Controller:
    def __init__(self, session):
        self.session = session
        self.calls = []
        self.release_count = 0

    def step(self, token, **kwargs):
        self.calls.append((token, kwargs))
        if token == "MV_FWD_SMALL":
            self.session.pose[0] += 0.002
        elif token == "RELEASE":
            self.release_count += 1
            if self.release_count > 1:
                # Recovery may change the scene/pose after the main action's
                # logged post_pose. Verification must precede this operation.
                self.session.pose[1] += 0.005
        return SimpleNamespace(token=token, post_pose=self.session.get_ee_pose())


class MujocoReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name) / "original"
        self.run_dir.mkdir()

    def write_run(self, records=None, *, config=None, simulator="mujoco"):
        metadata = {"simulator": simulator, "config": deepcopy(CONFIG if config is None else config)}
        (self.run_dir / "metadata.json").write_text(json.dumps(metadata))
        records = [record()] if records is None else records
        (self.run_dir / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
        return load_replay_plan(self.run_dir)

    def test_replays_every_record_including_done_with_visibility_and_recovery(self):
        recovered_pose = POSE.copy()
        recovered_pose[1] += 0.005
        plan = self.write_run([
            record(token="GRASP", recovery={"release": True, "event": "lost_grasp"}, recover=True),
            record(1, "DONE", pre=recovered_pose, post=recovered_pose),
            record(2, "STOP", pre=recovered_pose, post=recovered_pose, target_in_wrist=False),
        ])
        session = Session()
        controller = Controller(session)
        result = execute_replay(plan, session, controller, verify=True)
        self.assertEqual([token for token, _ in controller.calls], ["RELEASE", "GRASP", "RELEASE", "DONE", "STOP"])
        self.assertEqual(controller.calls[-1][1], {"target_in_wrist": False, "continuous": False})
        self.assertEqual(result["actions_replayed"], 3)
        self.assertEqual(result["comparisons"], 6)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["task_success"], "Replay completion is not task success")
        self.assertEqual(result["max_position_error_m"], 0)
        self.assertTrue(result["steps"][0]["recovery_release"])
        self.assertEqual(session.recorder.begin_step.call_count, 3)
        self.assertEqual(session.recorder.end_step.call_count, 3)

    def test_final_report_includes_physical_success_diagnostics(self):
        plan = self.write_run()
        session = Session()
        session.success_diagnostics = Mock(return_value={
            "success": False, "failed_criteria": ["retreat_clearance"],
        })
        result = execute_replay(plan, session, Controller(session))
        self.assertEqual(result["success_diagnostics"]["failed_criteria"], ["retreat_clearance"])
        session.success_diagnostics.assert_called_once()

    def test_before_decision_recovery_token_is_not_executed_twice(self):
        plan = self.write_run([record(token="RELEASE", recovery={"token": "RELEASE", "event": "empty_grasp"})])
        session = Session()
        controller = Controller(session)
        execute_replay(plan, session, controller)
        self.assertEqual([token for token, _ in controller.calls], ["RELEASE", "RELEASE"])

    def test_post_pose_mismatch_aborts_before_recovery_release(self):
        wrong_pose = POSE.copy()
        wrong_pose[0] += 0.01
        plan = self.write_run([record(token="MV_FWD_SMALL", post=wrong_pose, recovery={"release": True})])
        session, report = Session(), {}
        controller = Controller(session)
        with self.assertRaisesRegex(RuntimeError, "step 0.*post"):
            execute_replay(plan, session, controller, verify=True, report=report)
        self.assertEqual([token for token, _ in controller.calls], ["RELEASE", "MV_FWD_SMALL"])
        self.assertEqual(report["status"], "pose_mismatch")
        self.assertEqual(report["actions_replayed"], 1)
        self.assertAlmostEqual(report["max_position_error_m"], 0.008)
        session.recorder.end_step.assert_not_called()

    def test_pre_pose_mismatch_prevents_the_next_action(self):
        pre = POSE.copy()
        pre[2] -= 0.01
        plan = self.write_run([record(token="MV_FWD_SMALL", pre=pre)])
        session, report = Session(), {}
        controller = Controller(session)
        with self.assertRaisesRegex(RuntimeError, "step 0.*pre"):
            execute_replay(plan, session, controller, verify=True, report=report)
        self.assertEqual([token for token, _ in controller.calls], ["RELEASE"])
        self.assertEqual(report["actions_replayed"], 0)

    def test_pose_comparison_is_quaternion_sign_invariant_and_optional(self):
        opposite = POSE.copy()
        opposite[3:] = [0, 0, 0, -1]
        self.assertEqual(pose_error(POSE, opposite), {"position_m": 0.0, "rotation_deg": 0.0})
        rotated = POSE.copy()
        rotated[3:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        self.assertAlmostEqual(pose_error(POSE, rotated)["rotation_deg"], 90)
        plan = self.write_run([record(token="MV_FWD_SMALL")])
        session = Session()
        result = execute_replay(plan, session, Controller(session), verify=False)
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["max_position_error_m"], 0.002)

    def test_legacy_pose_less_records_require_explicitly_unverified_replay(self):
        self.write_run([{"i": 0, "act": "DONE"}])
        with self.assertRaisesRegex(ValueError, "--verify"):
            load_replay_plan(self.run_dir, require_poses=True)
        plan = load_replay_plan(self.run_dir)
        session = Session()
        report = execute_replay(plan, session, Controller(session))
        self.assertEqual(report["comparisons"], 0)

    def test_rejects_ambiguous_or_malformed_records_before_execution(self):
        bad_rows = [
            {"i": 0, "act": "STILL"}, {"i": 1, "act": "DONE"},
            record(token="NOT_AN_ACTION"), record(src="human"), record(target_in_wrist="false"),
            record(recovery={"release": "true"}), record(recovery={"token": "RELEASE"}),
            record(recover=True), record(pre=[0] * 7), record(post=[float("nan")] * 7),
            record(step_cm=float("inf")), record(i=True),
        ]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.write_run([row])
        for rows in ([record(0), record(0)], [record(0), record(2)], [record(0), record(2), record(1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.write_run(rows)
        self.write_run()
        (self.run_dir / "steps.jsonl").write_text('{"i":0,"act":')
        with self.assertRaisesRegex(ValueError, "truncated"):
            load_replay_plan(self.run_dir)

    def test_rejects_hardware_missing_initial_config_and_chunked_runs(self):
        with self.assertRaisesRegex(ValueError, "hardware"):
            self.write_run(simulator="franka")
        for setting in ("action_chunk", "dagger", "rotation", "smooth"):
            cfg = deepcopy(CONFIG)
            cfg["plugins"][setting] = True
            with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, setting):
                self.write_run(config=cfg)
        cfg = deepcopy(CONFIG)
        del cfg["reset_qpos"]
        with self.assertRaisesRegex(ValueError, "reset_qpos"):
            self.write_run(config=cfg)

    def test_effective_legacy_plugin_flags_cannot_hide_implicit_motion(self):
        cfg = deepcopy(CONFIG)
        cfg["tools"] = {"action_chunk": True}
        with self.assertRaisesRegex(ValueError, "action_chunk"):
            self.write_run(config=cfg)
        cfg["plugins"]["action_chunk"] = {"enabled": False}
        cfg["plugins"]["variable_step"] = {"enabled": False}
        plan = self.write_run(config=cfg)
        self.assertFalse(plan.variable_step)
        cfg["motion_frame"] = "wrist"
        with self.assertRaisesRegex(ValueError, "motion frame"):
            self.write_run(config=cfg)

    def test_dry_run_does_not_load_simulator_or_create_outputs(self):
        self.write_run()
        original = {path.name: path.read_bytes() for path in self.run_dir.iterdir()}
        import_function = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name, ("core.sim.mujoco_session", "scripts.run_mujoco"))
            return import_function(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import), patch("builtins.print") as printed:
            self.assertEqual(main([str(self.run_dir), "--dry-run", "--verify"]), 0)
        self.assertEqual({path.name: path.read_bytes() for path in self.run_dir.iterdir()}, original)
        self.assertEqual(list(self.run_dir.parent.iterdir()), [self.run_dir])
        self.assertNotIn(CONFIG["vlm"]["api_key"], str(printed.call_args_list))

    def test_refuses_original_or_existing_output_directory(self):
        self.write_run()
        for output in (self.run_dir, self.run_dir / "replay", self.run_dir.parent):
            with self.subTest(output=output), self.assertRaises(SystemExit) as error:
                main([str(self.run_dir), "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)

    def test_hold_final_syncs_without_advancing_physics(self):
        viewer = Mock()
        viewer.is_running.side_effect = [True, True, False]
        session = SimpleNamespace(_viewer=viewer, data=SimpleNamespace(time=1.25))
        with patch("scripts.mujoco.replay.time.sleep"):
            hold_final_view(session)
        self.assertEqual(viewer.sync.call_count, 2)
        self.assertEqual(session.data.time, 1.25)

    def test_invalid_tolerances_are_rejected(self):
        for position, rotation in ((-1, 0), (0, -1), (np.nan, 0), (0, np.inf)):
            with self.subTest(position=position, rotation=rotation), self.assertRaises(ValueError):
                validate_tolerances(position, rotation)


if __name__ == "__main__":
    unittest.main()
