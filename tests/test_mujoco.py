"""Original-scene geometry, physics, and policy-boundary regression checks.

Install requirements-mujoco.txt and run the two scripts/mujoco asset commands.
These tests use no Google API calls. Optional native dependencies/assets are
skipped in the repo's lightweight CI environment.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models/mujoco/rubiks_cube_bowl"


@pytest.fixture(scope="module")
def cfg():
    os.environ.setdefault("MUJOCO_GL", "egl")
    pytest.importorskip("mujoco")
    if not (ASSETS / "provenance.json").exists():
        pytest.skip("Original RoboLab MuJoCo assets have not been prepared")
    from scripts.run_mujoco import parse_args, resolve_config
    return resolve_config(parse_args(["--no-vlm"]))


@pytest.fixture
def task(cfg):
    from core.sim.mujoco_task import MujocoTask
    t = MujocoTask(copy.deepcopy(cfg))
    yield t
    t.close()


def test_original_task_and_zero_shot_harness(cfg):
    from core.config import load_yaml
    original = load_yaml(ROOT / "configs/primitives_franka.yaml")
    assert original["step_m"] == cfg["fine_step_m"] == 0.02
    assert cfg["task"] == "Put the cube in the bowl"
    assert cfg["vlm"]["model"] == "gemini-3.8-flash"
    for name in ("subgoal", "proprioception", "recovery", "variable_step", "action_chunk", "mem_text"):
        assert cfg["plugins"][name]
    for name in ("coords", "affordance", "video_ref"):
        assert not cfg["plugins"][name]


def test_source_objects_and_home_pose(task):
    assert task.provenance["source_revision"] == "ad45d4f974725d020f82c2b0d77d78533aeba2b3"
    np.testing.assert_allclose(task.data.body("bowl").xpos, [0.442576, 0.126594, 0.03034], atol=0.001)
    np.testing.assert_allclose(task.data.body("rubiks_cube").xpos, [0.430675, -0.097464, 0.034096], atol=0.001)
    np.testing.assert_allclose(task.ee_pose[:3], [0.3065, 0, 0.4367], atol=0.001)
    assert float(task.model.body("bowl").mass[0]) == pytest.approx(0.5)
    assert float(task.model.body("rubiks_cube").mass[0]) == pytest.approx(0.2)
    assert task.provenance["bowl_collision_parts"] > 1
    assert task.model.geom("left_short_finger_collision").id >= 0
    assert not task.provenance["ground_visible"]
    assert task.model.geom("ground").rgba[3] == 0  # authored collision-only ground


def test_original_controller_and_prompt_are_reused(task, cfg):
    from core.launch import make_runner
    from core.prompting.prompt_loader import load_prompt_dir
    from core.runners.real import RealEpisodeRunner
    from core.vlm.roles import ControllerAgent
    from interpreters.franka_atomic_controller import FrankaAtomicController
    from scripts.run_mujoco import make_controller

    session, controller = make_controller(task, cfg)
    runner = make_runner(cfg, load_prompt_dir(ROOT / "prompts"), None, session,
                         controller, None, False)
    assert type(runner) is RealEpisodeRunner
    assert type(controller) is FrankaAtomicController
    agent = runner.controls.controller.agent
    assert type(agent) is ControllerAgent
    assert agent.prompt_template == (ROOT / "prompts/controller.txt").read_text().strip()
    assert runner.planner is not None and runner.recovery_plugin.enabled
    assert not runner.affordance_plugin.enabled


def test_robot_observation_contains_no_task_truth_or_annotations(task, monkeypatch):
    from core.sim.mujoco_task import MujocoSession
    from core.record.images import prepare_view
    os.environ.setdefault("MUJOCO_GL", "egl")
    task.render()  # allocate the two OpenGL renderers
    captured = {}
    for prefix, renderer in (("agentview", task.renderer), ("wrist", task.wrist_renderer)):
        def capture(render=renderer.render, name=prefix):
            frame = render()
            captured[name] = frame.copy()
            return frame
        monkeypatch.setattr(renderer, "render", capture)
    obs = MujocoSession(task).get_observation()
    assert set(obs) == {"agentview", "wrist", "ee_pose", "gripper_width"}
    assert obs["agentview"].shape == obs["wrist"].shape == (256, 256, 3)
    # Compare the exact camera capture used by the policy, avoiding driver-level
    # one-byte rasterization differences between separate render calls.
    for prefix in ("agentview", "wrist"):
        raw = prepare_view(captured[prefix], **{
            key: task.cfg.get(f"{prefix}_{key}")
            for key in ("rotation_degrees", "flip", "crop_aspect", "square_size")})
        np.testing.assert_array_equal(obs[prefix], raw)


def test_atomic_displacements_and_unreachable_pose(task, cfg):
    from core.action_units import MOVE_ATOMS
    from scripts.run_mujoco import make_controller
    _, controller = make_controller(task, cfg)
    for token in MOVE_ATOMS:
        r = controller.step(token, step_override_m=0.02)
        np.testing.assert_allclose(r.post_pose[:3] - r.pre_pose[:3],
                                   0.02 * controller.move_vectors[token], atol=0.0005)
    before = task.data.qpos.copy()
    with pytest.raises(RuntimeError, match="cannot reach"):
        task.solve_ik(np.array([10, 10, 10, 1, 0, 0, 0], dtype=float))
    np.testing.assert_array_equal(before, task.data.qpos)


def test_original_fingers_hold_cube_and_bowl_collision_is_hollow(task, cfg):
    from core.sim.mujoco_task import evaluate_task
    from scripts.run_mujoco import make_controller
    _, controller = make_controller(task, cfg)
    controller.verbose = False
    assert not evaluate_task(task)["success"]
    # Deterministic physics fixture only; never used by the VLM runner.
    for token, count in [("MV_FWD", 6), ("MV_LEFT", 3), ("MV_DOWN", 14), ("GRASP", 1),
                         ("MV_UP", 6), ("MV_RIGHT", 10), ("MV_FWD", 1), ("MV_DOWN", 4)]:
        for _ in range(count):
            controller.step(token, step_override_m=0.02)
        if token == "GRASP":
            assert 0.05 < task.gripper_width < 0.065
        if token == "MV_UP":
            assert evaluate_task(task)["cube_center_world"][2] > 0.12
        assert not evaluate_task(task)["success"]
    controller.step("RELEASE")
    controller.step("MV_UP", step_override_m=0.04)
    task.advance(2)
    result = evaluate_task(task)
    assert result["success"] and result["gripper_detached"] and result["contact_with_bowl"]
    assert result["cube_center_world"][2] < 0.06  # inside, not perched on a filled hull


def test_missing_key_is_actionable(cfg, monkeypatch):
    from scripts.run_mujoco import parse_args, resolve_config
    monkeypatch.setattr("scripts.run_mujoco.load_secrets_env", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        resolve_config(parse_args([]))


def test_ground_truth_success_is_not_implied_by_model_done(task):
    from core.sim.mujoco_task import evaluate_task
    # A text/model completion cannot make the untouched scene succeed.
    result = evaluate_task(task)
    assert result["gripper_detached"]
    assert not result["object_in_container"] and not result["success"]
    assert json.dumps(result)  # independent evaluator is serializable for audit logs


def test_interrupt_cancels_remaining_episodes(cfg, monkeypatch):
    from scripts import run_mujoco
    calls = []
    monkeypatch.setattr(run_mujoco, "resolve_config", lambda args: cfg)

    def interrupted(args, config, index):
        calls.append(index)
        return {"success": False, "model_result": {"end_reason": "interrupted"}}

    monkeypatch.setattr(run_mujoco, "run_episode", interrupted)
    assert run_mujoco.main(["--episodes", "3"]) == 130
    assert calls == [0]


def test_runtime_failure_is_scored_and_secrets_are_redacted(task, cfg, tmp_path):
    from types import SimpleNamespace
    from scripts.run_mujoco import record_runtime_failure
    config = copy.deepcopy(cfg)
    config["vlm"]["api_key"] = "synthetic-test-secret"
    logger = SimpleNamespace(run_dir=tmp_path,
        write_summary=lambda data: (tmp_path / "summary.json").write_text(json.dumps(data)))
    result = record_runtime_failure(task, logger, config, RuntimeError("synthetic-test-secret failed"))
    assert not result["success"]
    assert result["episode_status"] == "runtime_error"
    assert result["error"] == "[REDACTED] failed"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["control_mode"] == "mujoco" and summary["end_reason"] == "runtime_error"
    assert "synthetic-test-secret" not in (tmp_path / "evaluation.json").read_text()
