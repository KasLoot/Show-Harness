"""Motion completion, synchronized observations, and offline trajectory replay."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cfg():
    os.environ.setdefault("MUJOCO_GL", "egl")
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    if not (root / "models/mujoco/rubiks_cube_bowl/provenance.json").exists():
        pytest.skip("Prepared MuJoCo scene required")
    from scripts.run_mujoco import parse_args, resolve_config
    config = resolve_config(parse_args(["--no-vlm"]))
    config["recording"]["fps"] = 10
    return config


def test_scaling_changes_visual_and_collision_extent_and_preserves_mass(cfg):
    from core.sim.mujoco_task import MujocoTask
    original = copy.deepcopy(cfg)
    original["cube_size_m"] = None
    a, b = MujocoTask(original), MujocoTask(cfg)
    try:
        def bounds(task):
            cube = task.data.body("rubiks_cube")
            extents = []
            for g in np.flatnonzero(task.model.geom_bodyid == cube.id):
                mesh = task.model.geom_dataid[g]
                start, count = task.model.mesh_vertadr[mesh], task.model.mesh_vertnum[mesh]
                points = task.model.mesh_vert[start:start+count] @ task.data.geom_xmat[g].reshape(3, 3).T
                points += task.data.geom_xpos[g]
                local = (points - cube.xpos) @ cube.xmat.reshape(3, 3)
                extents.append(np.ptp(local, axis=0))
            return np.array(extents)
        scale = b.provenance["runtime_overrides"]["cube"]["uniform_scale"]
        np.testing.assert_allclose(bounds(b), bounds(a) * scale, atol=1e-7)
        assert np.max(bounds(b)) == pytest.approx(.04, abs=1e-6)
        assert b.model.body("rubiks_cube").mass[0] == pytest.approx(a.model.body("rubiks_cube").mass[0])
    finally:
        a.close()
        b.close()


def test_motion_has_intermediate_states_and_finishes_at_target(cfg, monkeypatch):
    from core.sim.mujoco_task import MujocoTask
    task = MujocoTask(cfg)
    try:
        original_step = task.mj.mj_step
        previous = task.data.qpos.copy()
        positions = []

        def step(model, data):
            nonlocal previous
            # No live pose assignment is permitted between physics steps.
            np.testing.assert_allclose(data.qpos, previous, atol=1e-12, rtol=0)
            original_step(model, data)
            previous = data.qpos.copy()

        monkeypatch.setattr(task.mj, "mj_step", step)
        task.step_callbacks.append(lambda control: positions.append(task.ee_pose[:3].copy()))
        target = task.ee_pose.copy()
        target[0] += .04
        task.command_pose(target)
        assert len(positions) > 100
        assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < .001
        assert task.pose_error(target)[0] <= cfg["motion_control"]["position_tolerance_m"]
        assert max(abs(task.data.qvel[task.arm_dofs])) <= cfg["motion_control"]["velocity_tolerance_rad_s"]
        assert not task.motion_in_progress
        assert task.motion_records[-1]["status"] == "reached"
    finally:
        task.close()


def test_observation_waits_for_an_in_progress_movement(cfg, monkeypatch):
    from core.sim.mujoco_task import MujocoTask, MujocoSession
    task = MujocoTask(cfg)
    session = MujocoSession(task)
    entered, proceed, observed = threading.Event(), threading.Event(), threading.Event()
    errors = []
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    monkeypatch.setattr(task, "render", lambda: (image.copy(), image.copy()))

    def pause_once(controls):
        if not entered.is_set():
            entered.set()
            assert proceed.wait(5)

    def move():
        try:
            target = task.ee_pose.copy()
            target[0] += .02
            task.command_pose(target)
        except BaseException as exc:
            errors.append(exc)

    def observe():
        try:
            session.get_observation()
            observed.set()
        except BaseException as exc:
            errors.append(exc)

    task.step_callbacks.append(pause_once)
    motion_thread, observation_thread = threading.Thread(target=move), threading.Thread(target=observe)
    try:
        motion_thread.start()
        assert entered.wait(5)
        observation_thread.start()
        time.sleep(.03)
        assert not observed.is_set()
        proceed.set()
        motion_thread.join(10)
        observation_thread.join(10)
        assert not errors and observed.is_set()
        assert task.last_observation_time == task.motion_records[-1]["end_time_s"]
    finally:
        proceed.set()
        motion_thread.join(10)
        if observation_thread.ident is not None:
            observation_thread.join(10)
        task.close()


def test_workspace_clipping_resynchronizes_the_next_command_without_recording(cfg):
    from core.sim.mujoco_task import MujocoTask
    from scripts.run_mujoco import make_controller
    config = copy.deepcopy(cfg)
    config["workspace_max"][0] = .316
    task = MujocoTask(config)
    try:
        _, controller = make_controller(task, config)
        controller.verbose = False
        forward = controller.step("MV_FWD", step_override_m=.04)
        assert "workspace target clipped" in forward.note
        np.testing.assert_allclose(controller.target_pose, task.ee_pose, atol=1e-12)
        before = task.ee_pose.copy()
        controller.step("MV_BACK", step_override_m=.02)
        assert task.ee_pose[0] < before[0] - .019
    finally:
        task.close()


def test_stronger_gripper_respects_closed_stop_on_empty_grasp(cfg):
    from core.sim.mujoco_task import MujocoTask
    from scripts.run_mujoco import make_controller
    task = MujocoTask(cfg)
    positions = []
    task.step_callbacks.append(lambda controls: positions.append(task.data.qpos[task.finger_qpos].copy()))
    try:
        _, controller = make_controller(task, cfg)
        controller.verbose = False
        result = controller.step("GRASP")  # above the table: deliberately empty
        assert result.grasp_empty
        assert np.min(positions) > -0.0001
        assert task.gripper_width == pytest.approx(.08, abs=.0005)
    finally:
        task.close()


@pytest.fixture(scope="module")
def recorded_run(cfg, tmp_path_factory):
    from core.sim.mujoco_task import MujocoTask
    from core.sim.mujoco_recording import MujocoRecorder
    from scripts.run_mujoco import make_controller
    task = MujocoTask(cfg)
    run = tmp_path_factory.mktemp("motion_recording")
    recorder = MujocoRecorder(task, run, cfg)
    session, controller = make_controller(task, cfg)
    controller.verbose = False
    recorder.install_controller(controller)
    calls = []

    def post(*args, **kwargs):
        calls.append(task.data.time)
        assert not task.motion_in_progress
        assert np.max(abs(task.data.qvel[task.arm_dofs])) < .02
        time.sleep(.02)  # wall-clock inference latency must not enter the physics trace
        return SimpleNamespace(status_code=200)

    client = SimpleNamespace(_api_key="TEST-KEY", session=SimpleNamespace(post=post))
    recorder.install_client(client)
    try:
        session.get_observation()
        client.session.post("https://example.invalid")
        controller.step("MV_FWD", step_override_m=.02)
        session.get_observation()
        client.session.post("https://example.invalid")
        controller.step("MV_BACK", step_override_m=.02)
        with pytest.raises(RuntimeError, match="stale camera"):
            client.session.post("https://example.invalid")
        assert len(calls) == 2
        recorder.close(final_path=run / "motion.mp4")
    finally:
        recorder.close()
        task.close()
    return run


def test_endpoint_schema_and_simulation_clock_video(recorded_run):
    import imageio.v2 as imageio
    manifest = json.loads((recorded_run / "trajectory_manifest.json").read_text())
    actions = [json.loads(x) for x in (recorded_run / "action_endpoints.jsonl").read_text().splitlines()]
    requests = [json.loads(x) for x in (recorded_run / "request_timing.jsonl").read_text().splitlines()]
    assert manifest["complete"] and manifest["samples"] > 100
    assert len(actions) == 2 and [a["request_id"] for a in actions] == [1, 2]
    for action in actions:
        after = action["after"]
        assert len(after["eef_pose6d_xyz_rpy"]) == 6
        assert len(after["eef_pose_xyz_quat_xyzw"]) == 7
        assert len(after["arm_joint_positions_rad"]) == 7
        assert len(after["finger_joint_positions_m"]) == 2
        assert action["end_sample"] > action["start_sample"]
    assert requests[1]["observation_sim_time_s"] == actions[0]["after"]["sim_time_s"]
    assert all(r["response_sim_time_s"] == r["observation_sim_time_s"] for r in requests)
    reader = imageio.get_reader(recorded_run / "motion.mp4")
    try:
        assert reader.count_frames() == manifest["video_frames"] > len(actions) * 5
        meta = reader.get_meta_data()
        duration = manifest["end_sim_time_s"] - manifest["start_sim_time_s"]
        assert abs(meta["duration"] - duration) <= 2 / manifest["video_fps"]
        assert meta["size"] == (640, 512)
    finally:
        reader.close()


def test_replay_needs_no_network_and_controls_reproduce_state(recorded_run, monkeypatch):
    import requests
    from scripts.replay_mujoco import replay
    from core.sim.mujoco_recording import trajectory_rows

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline replay attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    manifest = json.loads((recorded_run / "trajectory_manifest.json").read_text())
    assert len(list(trajectory_rows(recorded_run, manifest))) == manifest["samples"]
    states = replay(recorded_run, mode="states", speed=4, fps=5)
    controls = replay(recorded_run, mode="controls", speed=4, fps=5, record_video=False)
    assert states["api_requests"] == controls["api_requests"] == 0
    assert controls["max_qpos_error_vs_recording"] < 1e-9
    assert controls["frames"] == 0 and controls["output"] is None
    np.testing.assert_allclose(controls["final_qpos"], states["final_qpos"], atol=1e-9)


def test_endpoint_playback_ignores_old_timing_and_controls(recorded_run, monkeypatch, tmp_path):
    import mujoco
    import requests
    from scripts import replay_mujoco
    from core.sim.mujoco_recording import trajectory_rows

    def forbidden(*args, **kwargs):
        raise AssertionError("Endpoint playback attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    manifest = json.loads((recorded_run / "trajectory_manifest.json").read_text())
    rows = trajectory_rows(recorded_run, manifest)
    initial = next(rows)
    rows.close()

    def initial_only(*args):
        yield initial
        raise AssertionError("Endpoint playback read per-step live motion instead of endpoints")

    monkeypatch.setattr(replay_mujoco, "trajectory_rows", initial_only)
    actions = replay_mujoco.load_actions(recorded_run)
    original_step = mujoco.mj_step
    previous = None

    def physical_step(model, data):
        nonlocal previous
        if previous is not None:
            np.testing.assert_array_equal(data.qpos, previous)
        original_step(model, data)
        previous = data.qpos.copy()

    monkeypatch.setattr(mujoco, "mj_step", physical_step)
    first = replay_mujoco.replay(recorded_run, fps=5, output=tmp_path / "first.mp4")
    previous = None
    changed = copy.deepcopy(actions)
    for action in changed:
        # Old timestamps, step counts, and request IDs are irrelevant to the new
        # trajectory. Only geometric endpoints and gripper ordering are used.
        action["before"]["sim_time_s"] = 10000
        action["after"]["sim_time_s"] = 100000
        action["start_sample"], action["end_sample"] = 999999, 9999999
        action["request_id"] = 99
        for motion in action["motions"]:
            motion["start_time_s"], motion["end_time_s"] = 10000, 100000
    monkeypatch.setattr(replay_mujoco, "load_actions", lambda path: changed)
    second = replay_mujoco.replay(recorded_run, record_video=False, output=tmp_path / "retimed.mp4")
    assert first["api_requests"] == second["api_requests"] == 0
    assert first["source_physics_states_read"] == second["source_physics_states_read"] == 1
    assert not first["source_timing_used"] and not first["source_controls_used"]
    assert first["simulated_duration_s"] == second["simulated_duration_s"]
    np.testing.assert_array_equal(first["final_qpos"], second["final_qpos"])
    assert abs(first["frames"]/5-first["simulated_duration_s"]) <= 2/5
    assert max(e["position_error_m"] for e in first["events"]) < .0005
    # Removing the final return endpoint changes the physical destination. This
    # proves the dense trace is not secretly supplying the robot trajectory.
    previous = None
    monkeypatch.setattr(replay_mujoco, "load_actions", lambda path: changed[:1])
    third = replay_mujoco.replay(recorded_run, record_video=False, output=tmp_path / "new_goal.mp4")
    assert np.max(abs(np.asarray(first["final_qpos"])-third["final_qpos"])) > .01


def test_endpoint_speed_multiplier_is_rejected(recorded_run):
    from scripts.replay_mujoco import replay
    with pytest.raises(ValueError, match="--linear-speed"):
        replay(recorded_run, speed=2)


def test_endpoint_cruise_joins_waypoints_and_bounds_joint_motion(cfg):
    from core.sim.mujoco_task import MujocoTask
    from core.sim.mujoco_playback import EndpointPlayer, PlaybackSettings
    from core.sim.mujoco_recording import MujocoRecorder
    task = MujocoTask(cfg)
    try:
        initial = MujocoRecorder.pose_record(SimpleNamespace(task=task))
        actions, before = [], initial
        for i in range(3):
            pose = task.ee_pose.copy()
            pose[0] += (i+1)*.04
            q = task.solve_ik(pose)
            work = task.ik_data
            task.mj.mj_forward(task.model, work)
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(work.site_xmat[task.eef_id].reshape(3, 3)).as_quat()
            after = {**initial, "arm_joint_positions_rad": q.tolist(),
                     "eef_pose_xyz_quat_xyzw": np.r_[work.site_xpos[task.eef_id], quat].tolist()}
            actions.append({"action_id": i, "token": "MV_FWD", "status": "completed",
                            "before": before, "after": after, "motions": []})
            before = after
        manifest = {"arm_joint_names": [f"joint{i}" for i in range(1, 8)],
                    "finger_joint_names": ["finger_joint1", "finger_joint2"]}
        settings = PlaybackSettings()
        player = EndpointPlayer(task.model, task.data, manifest, actions, settings, cfg["gripper_control"])
        assert len(player.events) == 1
        assert len(player.events[0]["waypoints"]) == 4
        list(player.execute())
        cruise = [r for r in player.trace if r["cruise"]]
        assert len(cruise) > 100
        np.testing.assert_allclose([r["target_tool_speed"] for r in cruise], .1, atol=.001)
        assert np.percentile([abs(r["tool_speed"]-.1) for r in cruise], 95) < .005
        assert max(np.max(abs(r["target_velocity"])) for r in player.trace) <= settings.joint_speed*1.01
        assert max(np.max(abs(r["target_acceleration"])) for r in player.trace) <= settings.joint_acceleration*1.01
        # Interior action endpoints are crossed at cruise speed without a stop.
        for action in actions[:-1]:
            xyz = np.asarray(action["after"]["eef_pose_xyz_quat_xyzw"][:3])
            nearest = min(player.trace, key=lambda r: np.linalg.norm(r["target_eef_xyz"]-xyz))
            assert nearest["target_tool_speed"] == pytest.approx(.1, abs=.001)
            assert np.linalg.norm(nearest["eef_xyz"]-xyz) < .001
    finally:
        task.close()


def test_gripper_subevents_and_direction_reversals_are_preserved():
    from core.sim.mujoco_playback import endpoint_events, PlaybackSettings

    def point(x):
        return {"arm_joint_positions_rad": [x]*7, "eef_pose_xyz_quat_xyzw": [x, 0, 0, 0, 0, 0, 1]}

    actions = []
    before = point(0)
    for i, x in enumerate((.04, .08, .04)):
        actions.append({"action_id": i, "status": "completed", "token": "MV_FWD",
                        "before": before, "after": point(x), "motions": []})
        before = point(x)
    actions.append({"action_id": 3, "status": "completed", "token": "GRASP", "before": before,
                    "after": before, "motions": [{"kind": "gripper", "close": True},
                                                  {"kind": "gripper", "close": False}]})
    events = endpoint_events(actions, PlaybackSettings())
    assert [e["kind"] for e in events] == ["move", "move", "gripper", "gripper"]
    assert events[0]["action_ids"] == [0, 1]
    assert [e["close"] for e in events if e["kind"] == "gripper"] == [True, False]
    actions[-1]["status"] = "failed"
    with pytest.raises(ValueError, match="incomplete"):
        endpoint_events(actions, PlaybackSettings())


def test_no_fail_joins_across_empty_grasps_and_keeps_successful_grips():
    from core.sim.mujoco_playback import endpoint_events, PlaybackSettings

    def point(x):
        return {"arm_joint_positions_rad": [x]*7, "eef_pose_xyz_quat_xyzw": [x, 0, 0, 0, 0, 0, 1]}

    def action(i, token, before, after, *, empty=False, grips=()):
        return {"action_id": i, "token": token, "status": "completed", "grasp_empty": empty,
                "before": point(before), "after": point(after),
                "motions": [{"kind": "gripper", "close": close} for close in grips]}

    actions = [action(0, "MV_FWD", 0, .04),
               action(1, "GRASP", .04, .0399, empty=True, grips=(True, False)),
               action(2, "DONE", .0399, .0399),
               action(3, "MV_FWD", .0399, .08),
               action(4, "GRASP", .08, .08, grips=(True,)),
               action(5, "RELEASE", .08, .08, grips=(False,))]
    full = endpoint_events(actions, PlaybackSettings())
    filtered = endpoint_events(actions, PlaybackSettings(), "no_fail")
    assert [e["close"] for e in full if e["kind"] == "gripper"] == [True, False, True, False]
    assert [e["kind"] for e in filtered] == ["move", "gripper", "gripper"]
    assert filtered[0]["action_ids"] == [0, 3]
    np.testing.assert_array_equal([p["q"][0] for p in filtered[0]["waypoints"]], [0, .04, .08])
    assert [(e["action_ids"], e["close"]) for e in filtered if e["kind"] == "gripper"] == [([4], True), ([5], False)]
    # Missing outcomes must not be guessed from a close/reopen pattern.
    del actions[1]["grasp_empty"]
    unknown = endpoint_events(actions, PlaybackSettings(), "no_fail")
    assert len([e for e in unknown if e["kind"] == "gripper"]) == 4
    actions[1].update(status="failed", grasp_empty=True)
    with pytest.raises(ValueError, match="incomplete"):
        endpoint_events(actions, PlaybackSettings(), "no_fail")


def test_no_fail_physical_playback_skips_initial_and_trailing_empty_grasps(recorded_run, monkeypatch):
    from scripts import replay_mujoco

    source = replay_mujoco.load_actions(recorded_run)
    initial, final = source[0]["before"], source[-1]["after"]

    def empty(i, point):
        return {"action_id": i, "request_id": 1, "token": "GRASP", "status": "completed",
                "grasp_empty": True, "before": point, "after": point,
                "motions": [{"kind": "gripper", "close": True}, {"kind": "gripper", "close": False}]}

    monkeypatch.setattr(replay_mujoco, "load_actions", lambda path: [empty(50, initial), *source, empty(51, final)])
    full = replay_mujoco.replay(recorded_run, playback_type="full", record_video=False)
    filtered = replay_mujoco.replay(recorded_run, playback_type="no_fail", record_video=False)
    assert full["playback_type"] == "full" and full["skipped_actions"] == []
    assert filtered["playback_type"] == "no_fail" and filtered["skipped_action_count"] == 2
    assert [a["action_id"] for a in filtered["skipped_actions"]] == [50, 51]
    assert all(e["kind"] == "move" for e in filtered["events"])
    assert full["simulated_duration_s"] > filtered["simulated_duration_s"] + 4
    assert Path(filtered["trajectory"]).name == "replay_endpoints_no_fail_trajectory.npz"
    # The result must equal the physical execution with those actions absent.
    monkeypatch.setattr(replay_mujoco, "load_actions", lambda path: source)
    baseline = replay_mujoco.replay(recorded_run, record_video=False, output=recorded_run / "baseline.mp4")
    np.testing.assert_array_equal(filtered["final_qpos"], baseline["final_qpos"])
    assert filtered["simulated_duration_s"] == baseline["simulated_duration_s"]
    # An all-failed log remains a valid stationary playback.
    monkeypatch.setattr(replay_mujoco, "load_actions", lambda path: [empty(50, initial)])
    idle = replay_mujoco.replay(recorded_run, playback_type="no_fail", record_video=False,
                                 output=recorded_run / "idle.mp4")
    assert idle["status"] == "completed" and idle["events"] == []
    assert idle["skipped_action_count"] == 1 and idle["simulated_duration_s"] == pytest.approx(.5)


def test_no_fail_cannot_filter_diagnostic_state_or_control_replay(recorded_run):
    from scripts.replay_mujoco import replay
    for mode in ("states", "controls"):
        with pytest.raises(ValueError, match="requires --mode endpoints"):
            replay(recorded_run, mode=mode, playback_type="no_fail")
    with pytest.raises(ValueError, match="full or no_fail"):
        replay(recorded_run, playback_type="unknown")


def test_smooth_cruise_has_constant_middle_and_smooth_endpoints():
    from core.sim.mujoco_playback import SmoothCruise
    for length in (.01, .2, 1.0):
        profile = SmoothCruise(length, 1, .25)
        assert profile.sample(0)[:3] == (0, 0, 0)
        np.testing.assert_allclose(profile.sample(profile.duration)[:3], [length, 0, 0], atol=1e-12)
        for boundary in (profile.ramp, profile.ramp+profile.cruise):
            a = profile.sample(boundary-1e-7)
            b = profile.sample(boundary+1e-7)
            np.testing.assert_allclose(a[:3], b[:3], atol=1e-6)
        if profile.cruise:
            assert profile.sample(profile.ramp+profile.cruise/2)[1:3] == (1, 0)


def test_pending_trajectory_is_saved_when_final_video_render_fails(cfg, tmp_path, monkeypatch):
    from core.sim.mujoco_task import MujocoTask
    from core.sim.mujoco_recording import MujocoRecorder
    task = MujocoTask(cfg)
    recorder = MujocoRecorder(task, tmp_path, cfg)
    try:
        task.advance(.01)

        def fail():
            raise RuntimeError("synthetic final-frame failure")

        monkeypatch.setattr(recorder, "_capture_video", fail)
        with pytest.raises(RuntimeError, match="final-frame"):
            recorder.close(final_path=tmp_path / "motion.mp4")
        manifest = json.loads((tmp_path / "trajectory_manifest.json").read_text())
        assert manifest["complete"] and manifest["samples"] == 6
        assert manifest["video_error"] == "synthetic final-frame failure"
        assert recorder.endpoints.closed and recorder.requests.closed and recorder.frames.closed
        assert not task.step_callbacks and not task.before_step_callbacks
    finally:
        recorder.close()
        task.close()


def test_runner_keeps_motion_video_and_failed_action_on_controller_error(cfg, tmp_path, monkeypatch):
    from core.sim.mujoco_task import MujocoTask
    from scripts.run_mujoco import parse_args, run_episode
    config = copy.deepcopy(cfg)
    config["log_dir"] = str(tmp_path)

    def fail_ik(*args, **kwargs):
        raise RuntimeError("synthetic unreachable target")

    def make_runner(config, prompts, client, session, controller, logger, debug):
        return SimpleNamespace(run=lambda: controller.step("MV_FWD", step_override_m=.02))

    client = SimpleNamespace(_api_key="TEST-KEY", session=SimpleNamespace(post=lambda *a, **kw: None, close=lambda: None))
    monkeypatch.setattr(MujocoTask, "solve_ik", fail_ik)
    monkeypatch.setattr("core.launch.make_runner", make_runner)
    monkeypatch.setattr("core.launch.make_vlm_client", lambda *args: client)
    with pytest.raises(RuntimeError, match="unreachable"):
        run_episode(parse_args([]), config, 0)
    evaluation_path = next(tmp_path.rglob("evaluation.json"))
    evaluation = json.loads(evaluation_path.read_text())
    assert not evaluation["success"] and evaluation["episode_status"] == "runtime_error"
    assert Path(evaluation["video_path"]).exists()
    manifest = json.loads(Path(evaluation["trajectory_manifest"]).read_text())
    assert manifest["complete"]
    actions = [json.loads(x) for x in Path(evaluation["action_endpoints"]).read_text().splitlines()]
    assert actions[-1]["status"] == "failed"
    assert len(actions[-1]["after"]["eef_pose6d_xyz_rpy"]) == 6
