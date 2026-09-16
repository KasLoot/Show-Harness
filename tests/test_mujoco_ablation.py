"""Isolation tests for the camera-only intervention and common request audit."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.record.images import image_to_data_url
from core.sim.mujoco_ablation import (
    AblationInstrumentation, SIDE_CAMERA_DESCRIPTION, add_camera_description,
    SIDE_GRASP_RULE_TEMPLATE, add_side_grasp_check, add_side_image, side_camera_spec,
)

ORIGINAL_GRASP_RULE = (
    "- GRASP when BOTH AgentView and Wrist view confirm the main body "
    "is clearly between the center of two grippers"
)


def payload():
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    parts = [{"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
             {"type": "image_url", "image_url": {"url": image_to_data_url(image[::-1])}},
             {"type": "text", "text": "TASK: Put the cube in the bowl\nSTAGE: GRASP\n" + ORIGINAL_GRASP_RULE}]
    return {"model": "test-model", "messages": [{"role": "user", "content": parts}],
            "temperature": 0.0, "max_tokens": 2048, "reasoning_effort": "medium"}


def test_only_difference_is_one_image_part():
    original = payload()
    saved = copy.deepcopy(original)
    side = np.zeros((4, 4, 3), dtype=np.uint8)
    assert add_side_image(original, side, False) is original
    variant = add_side_image(original, side, True)
    assert len(variant["messages"][0]["content"]) == 4
    added = variant["messages"][0]["content"].pop(2)
    assert added["type"] == "image_url"
    assert variant == original == saved  # text, old images and inference params identical
    with pytest.raises(RuntimeError, match="not captured"):
        add_side_image(original, None, True)


def test_camera_description_only_adds_one_text_part():
    original = add_side_image(payload(), np.zeros((4, 4, 3), dtype=np.uint8), True)
    saved = copy.deepcopy(original)
    assert add_camera_description(original, False) is original
    variant = add_camera_description(original, True)
    assert variant["messages"][0]["content"].pop() == {
        "type": "text", "text": SIDE_CAMERA_DESCRIPTION}
    assert variant == original == saved
    with pytest.raises(ValueError, match="three-camera"):
        add_camera_description(payload(), True)
    with pytest.raises(ValueError, match="requires the side image"):
        AblationInstrumentation(False, describe_side=True)


def test_side_grasp_check_changes_one_condition_and_no_other_payload_fields():
    original = add_camera_description(add_side_image(payload(), np.zeros((4, 4, 3), dtype=np.uint8), True), True)
    saved = copy.deepcopy(original)
    variant = add_side_grasp_check(original, True)
    old_text = original["messages"][0]["content"][3]["text"]
    changed = variant["messages"][0]["content"][3]["text"]
    assert changed.splitlines()[:-1] == old_text.splitlines()[:-1]
    assert "fixed side view (image 3)" in changed.splitlines()[-1]
    assert "vertically overlap its middle" in changed.splitlines()[-1]
    variant["messages"][0]["content"][3]["text"] = old_text
    assert variant == original == saved  # all three images, description, params and other text
    planner = payload()
    planner["messages"][0]["content"][-1]["text"] = "ROLE: SubgoalPlanner"
    assert add_side_grasp_check(planner, True) is planner
    with pytest.raises(ValueError, match="requires"):
        AblationInstrumentation(True, check_side_grasp=True)


@pytest.mark.parametrize("send_side,describe_side,check_grasp", [
    (False, False, False), (True, False, False), (True, True, False), (True, True, True)])
def test_every_http_attempt_is_saved_without_policy_state_leak(send_side, describe_side, check_grasp, tmp_path, monkeypatch):
    image = np.full((4, 4, 3), 71, dtype=np.uint8)
    data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3), ctrl=np.zeros(1), time=0.0)
    task = SimpleNamespace(cfg={"observer_cameras": [{"name": "ablation_side"}]},
                           data=data, render_observer=lambda name: image.copy())
    monkeypatch.setattr("core.sim.mujoco_task.build_model_xml", lambda cfg: ("<mujoco/>", {}))
    observation = {"agentview": image, "wrist": image, "ee_pose": np.zeros(7), "gripper_width": .08}
    session = SimpleNamespace(robot=object(), config=object(), get_observation=lambda: observation)
    step_result = SimpleNamespace(grasp_empty=False, note="")
    controller = SimpleNamespace(step=lambda token, **kw: step_result)
    calls = []

    def post(url, **kw):
        calls.append(kw["json"])
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [], "secret": "TEST-KEY"})

    client = SimpleNamespace(_api_key="TEST-KEY", session=SimpleNamespace(post=post))
    audit = AblationInstrumentation(send_side, describe_side=describe_side, check_side_grasp=check_grasp)
    wrapped = audit.install(task, session, controller, SimpleNamespace(run_dir=tmp_path), client)
    assert wrapped.get_observation() is observation
    p = payload()
    client.session.post("https://example.invalid/chat", json=p)
    client.session.post("https://example.invalid/chat", json=p)  # same-payload transport retry
    assert controller.step("MV_UP") is step_result
    audit.close()
    assert len(calls) == 2
    assert calls[0] == calls[1]
    for index in (1, 2):
        directory = tmp_path / "requests" / f"{index:04d}"
        record = json.loads((directory / "request.json").read_text())
        assert len(record["images"]) == (3 if send_side else 2)
        prompt = p["messages"][0]["content"][-1]["text"]
        if check_grasp:
            prompt = prompt.replace(ORIGINAL_GRASP_RULE, SIDE_GRASP_RULE_TEMPLATE.format(affordance="main body"))
        if describe_side:
            prompt += "\n\n" + SIDE_CAMERA_DESCRIPTION
        assert (directory / "prompt.txt").read_text() == prompt
        assert "TEST-KEY" not in (directory / "response.txt").read_text()
    assert set(observation) == {"agentview", "wrist", "ee_pose", "gripper_width"}
    assert "qpos" not in json.dumps(calls)
    assert len((tmp_path / "physics_states.jsonl").read_text().splitlines()) == 1


def test_description_cli_requires_side_camera():
    from scripts.run_mujoco import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--describe-side-camera"])
    args = parse_args(["--extra-view", "side", "--describe-side-camera"])
    assert args.describe_side_camera and args.extra_view == "side"
    with pytest.raises(SystemExit):
        parse_args(["--extra-view", "side", "--side-grasp-check"])
    args = parse_args(["--extra-view", "side", "--describe-side-camera", "--side-grasp-check"])
    assert args.side_grasp_check


@pytest.mark.parametrize("factor,names,treatment,key", [
    ("side-description", ["side", "side_described"], "append_fixed_camera_description", "describe_side"),
    ("side-grasp-check", ["side_described", "side_grasp_checked"], "replace_visual_grasp_condition", "check_side_grasp"),
])
def test_description_protocol_changes_only_one_factor(tmp_path, monkeypatch, factor, names, treatment, key):
    from scripts.mujoco import ablate
    monkeypatch.setattr(ablate, "ROOT", tmp_path)
    monkeypatch.setattr(ablate, "runtime_manifest", lambda: {})
    monkeypatch.setattr(ablate, "run_args", lambda args: None)
    monkeypatch.setattr(ablate, "resolve_config", lambda args: {
        "task": "Put the cube in the bowl", "vlm": {"model": "test", "api_key": "TEST-KEY"},
        "max_steps": 150})
    protocol = ablate.prepare(tmp_path / "study", 3, factor)
    a, b = names
    assert [r["condition"] for r in protocol["order"]] == [a, b, b, a, a, b]
    assert protocol["sole_treatment"] == treatment
    before, after = [protocol["condition_parameters"][name] for name in names]
    assert [k for k in before if before[k] != after[k]] == [key]
    assert protocol["camera_description"] == SIDE_CAMERA_DESCRIPTION
    assert protocol["max_steps"] == 150


def test_passive_camera_does_not_change_physics_or_existing_camera_parameters():
    os.environ.setdefault("MUJOCO_GL", "egl")
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    if not (root / "models/mujoco/rubiks_cube_bowl/panda.xml").exists():
        pytest.skip("MuJoCo assets not prepared")
    from core.sim.mujoco_task import MujocoTask
    from scripts.run_mujoco import parse_args, resolve_config, make_controller
    cfg = resolve_config(parse_args(["--no-vlm"]))
    extra = copy.deepcopy(cfg)
    extra["observer_cameras"] = [side_camera_spec()]
    a, b = MujocoTask(cfg), MujocoTask(extra)
    try:
        assert b.model.ncam == a.model.ncam + 1
        for name in ("body_mass", "body_inertia", "body_pos", "geom_size", "geom_type",
                     "actuator_gainprm", "actuator_biasprm", "jnt_range"):
            np.testing.assert_array_equal(getattr(a.model, name), getattr(b.model, name))
        for name in ("cam_pos", "cam_quat", "cam_intrinsic", "cam_fovy"):
            for camera in ("front_cam", "wrist_cam"):
                np.testing.assert_array_equal(getattr(a.model, name)[a.model.camera(camera).id],
                                              getattr(b.model, name)[b.model.camera(camera).id])
        _, ca = make_controller(a, cfg)
        _, cb = make_controller(b, extra)
        for token in ("MV_FWD", "MV_LEFT", "MV_DOWN"):
            ca.step(token, step_override_m=.02)
            cb.step(token, step_override_m=.02)
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        before = b.data.qpos.copy()
        side = b.render_observer("ablation_side")
        assert side.shape == (256, 256, 3) and side.dtype == np.uint8
        np.testing.assert_array_equal(b.data.qpos, before)
    finally:
        a.close()
        b.close()
