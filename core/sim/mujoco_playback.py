"""Plan and execute new, time-independent motion from recorded robot endpoints.

Only the initial world state is restored. Subsequent robot and object motion is
integrated by MuJoCo; recorded per-step states, controls and timings are unused.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PlaybackSettings:
    linear_speed: float = 0.10
    angular_speed: float = 0.5
    joint_speed: float = 1.0
    joint_acceleration: float = 4.0
    ramp_time: float = 0.25

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Playback {name} must be positive and finite")


class SmoothCruise:
    """Constant-speed cruise with quintic velocity ramps (zero endpoint jerk)."""

    def __init__(self, distance, speed, ramp_time):
        self.distance = float(distance)
        # A short move has two ramps and a reduced peak speed, without a cruise.
        scale = min(1.0, np.sqrt(distance / (speed * ramp_time)))
        self.speed = speed * scale
        self.ramp = ramp_time * scale
        self.cruise = max(0.0, distance / self.speed - self.ramp)
        if self.cruise < 1e-12:
            self.cruise = 0.0
        self.duration = 2 * self.ramp + self.cruise

    @staticmethod
    def _ramp(u):
        return (2.5*u**4 - 3*u**5 + u**6,
                10*u**3 - 15*u**4 + 6*u**5,
                30*u**2 - 60*u**3 + 30*u**4)

    def sample(self, time):
        t = np.clip(time, 0.0, self.duration)
        if t < self.ramp:
            p, v, a = self._ramp(t / self.ramp)
            return self.speed*self.ramp*p, self.speed*v, self.speed/self.ramp*a, False
        if t <= self.ramp + self.cruise:
            return self.speed * (t - self.ramp/2), self.speed, 0.0, self.cruise > 0
        p, v, a = self._ramp((self.duration - t) / self.ramp)
        return self.distance - self.speed*self.ramp*p, self.speed*v, -self.speed/self.ramp*a, False


def _waypoint(record):
    q = np.asarray(record["arm_joint_positions_rad"], dtype=float)
    pose = np.asarray(record["eef_pose_xyz_quat_xyzw"], dtype=float)
    if q.shape != (7,) or pose.shape != (7,) or not np.isfinite(np.r_[q, pose]).all():
        raise ValueError("Each playback endpoint requires finite seven-joint and XYZ/quaternion poses")
    if abs(np.linalg.norm(pose[3:]) - 1) > 1e-3:
        raise ValueError("Playback endpoint contains an invalid quaternion")
    return {"q": q, "pose": pose}


def _direction(a, b, settings):
    angle = (Rotation.from_quat(b["pose"][3:]) * Rotation.from_quat(a["pose"][3:]).inv()).as_rotvec()
    return np.r_[(b["pose"][:3] - a["pose"][:3]) / settings.linear_speed,
                 angle / settings.angular_speed]


def _recorded_empty_grasp(action):
    """Use the controller's measured outcome, never a guess from gripper width."""
    return (action.get("status") == "completed" and action.get("token") == "GRASP"
            and action.get("grasp_empty") is True)


def endpoint_events(actions, settings, playback_type="full"):
    """Join nearly collinear moves; stop at direction changes and gripper events.

    Planner DONE records and duplicate positions do not insert a pause. Gripper
    Full playback preserves empty-close/reopen subevents. no_fail omits those
    explicitly marked by the controller, allowing neighboring moves to join.
    """
    if playback_type not in ("full", "no_fail"):
        raise ValueError("Playback type must be full or no_fail")
    if not actions:
        raise ValueError("No action endpoints were recorded")
    events, pending, ids = [], [_waypoint(actions[0]["before"])], []

    def flush():
        nonlocal pending, ids
        if len(pending) > 1:
            events.append({"kind": "move", "waypoints": pending, "action_ids": ids})
        pending, ids = [pending[-1]], []

    def append(point, action_id):
        if np.max(abs(point["q"] - pending[-1]["q"])) < 1e-6:
            return
        if len(pending) > 1:
            a = _direction(pending[-2], pending[-1], settings)
            b = _direction(pending[-1], point, settings)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            # Null-space moves and reversals also need a smooth stop.
            if norm < 1e-10 or np.dot(a, b) / norm < np.cos(np.deg2rad(5)):
                flush()
        pending.append(point)
        ids.append(action_id)

    for index, action in enumerate(actions):
        if action.get("status") != "completed":
            raise ValueError(f"Action {index} is incomplete; refusing to execute a partial endpoint")
        if playback_type == "no_fail" and _recorded_empty_grasp(action):
            # Drop the entire attempt, including contact/settling jitter in its
            # post-action arm pose. Do not flush: adjacent moves can join.
            continue
        after = _waypoint(action["after"])
        grips = [m for m in action.get("motions", []) if m["kind"] == "gripper"]
        if (playback_type == "no_fail" and not grips and
                np.max(abs(after["q"] - _waypoint(action["before"])["q"])) < 1e-6):
            # A following DONE/noop can copy a skipped attempt's displaced
            # endpoint. It must not reintroduce that discarded pose or a stop.
            continue
        if grips:
            append(_waypoint(action["before"]), action["action_id"])
            flush()
            for grip in grips:
                events.append({"kind": "gripper", "close": bool(grip["close"]),
                               "action_ids": [action["action_id"]], "waypoint": pending[-1]})
            # Retain the measured post-grasp joint pose, including any tiny
            # arm displacement due to contact, before proceeding to the lift.
            append(after, action["action_id"])
            flush()
        else:
            append(after, action["action_id"])
    flush()
    return events


class JointPath:
    """Interpolate joint waypoints and parameterize by tool/joint travel speed."""

    def __init__(self, player, points, settings):
        import mujoco

        self.start, self.goal = points[0], points[-1]
        q = np.array([p["q"] for p in points])
        distances = [max(np.linalg.norm(_direction(a, b, settings)),
                         np.max(abs(b["q"] - a["q"])) / settings.joint_speed)
                     for a, b in zip(points, points[1:])]
        knots = np.r_[0.0, np.cumsum(distances)]
        shape = PchipInterpolator(knots, q, axis=0)
        # Dense kinematics is planning only, on a separate data instance.
        grid = np.unique(np.concatenate([np.linspace(a, b, max(32, int(np.ceil((b-a)*200))))
                                         for a, b in zip(knots, knots[1:])]))
        coordinates, derivatives = shape(grid), shape(grid, 1)
        metrics, xyz = [], []
        jacp, jacr = np.zeros((3, player.model.nv)), np.zeros((3, player.model.nv))
        for positions, derivative in zip(coordinates, derivatives):
            work = player.kinematics
            work.qpos[player.arm_qpos] = positions
            mujoco.mj_kinematics(player.model, work)
            mujoco.mj_comPos(player.model, work)
            mujoco.mj_jacSite(player.model, work, jacp, jacr, player.eef_id)
            metrics.append(max(np.linalg.norm(jacp[:, player.arm_dofs] @ derivative) / settings.linear_speed,
                               np.linalg.norm(jacr[:, player.arm_dofs] @ derivative) / settings.angular_speed,
                               np.max(abs(derivative)) / settings.joint_speed, 1e-8))
            xyz.append(work.site_xpos[player.eef_id].copy())
        metrics = np.asarray(metrics)
        times = np.r_[0.0, np.cumsum(np.diff(grid) * (metrics[1:] + metrics[:-1]) / 2)]
        self.curve = PchipInterpolator(times, coordinates, axis=0)
        check = np.unique(np.r_[times, (times[1:] + times[:-1])/2])
        peak_derivative = np.max(abs(self.curve(check, 1)))
        curvature = np.max(abs(self.curve(check, 2)))
        # Reserve half the acceleration budget for curvature, half for ramps.
        peak = min(1.0, np.sqrt(settings.joint_acceleration / max(2*curvature, 1e-9)))
        ramp = max(settings.ramp_time, 2*1.875*peak*peak_derivative / settings.joint_acceleration)
        self.profile = SmoothCruise(times[-1], peak, ramp)
        self.length_m = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        self.knot_times = np.interp(knots, grid, times)

    def sample(self, time):
        position, speed, acceleration, cruise = self.profile.sample(time)
        q = self.curve(position)
        derivative = self.curve(position, 1)
        velocity = derivative * speed
        acc = self.curve(position, 2) * speed**2 + derivative * acceleration
        return q, velocity, acc, cruise


class EndpointPlayer:
    """Physical servo execution of retimed waypoints, with no policy access."""

    def __init__(self, model, data, manifest, actions, settings, gripper=None, *, playback_type="full"):
        import mujoco

        self.model, self.data, self.settings = model, data, settings
        self.kinematics = mujoco.MjData(model)
        self.kinematics.qpos[:] = data.qpos
        self.eef_id = model.site("eef").id
        joints = [model.joint(n) for n in manifest["arm_joint_names"]]
        self.arm_qpos = np.array([j.qposadr[0] for j in joints])
        self.arm_dofs = np.array([j.dofadr[0] for j in joints])
        self.arm_actuators = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.limits = np.array([j.range for j in joints])
        fingers = [model.joint(n) for n in manifest["finger_joint_names"]]
        self.finger_qpos = np.array([j.qposadr[0] for j in fingers])
        self.finger_dofs = np.array([j.dofadr[0] for j in fingers])
        self.finger_actuators = np.array([model.actuator(f"finger_actuator{i}").id for i in (1, 2)])
        self.grip = gripper or {}
        self.closed = False
        self.grip_reference = data.qpos[self.finger_qpos].copy()
        self.bias = np.zeros(7)
        self.events = endpoint_events(actions, settings, playback_type)
        self.skipped_actions = [
            {"action_id": a["action_id"], "request_id": a.get("request_id"),
             "token": a["token"], "reason": "recorded_empty_grasp"}
            for a in actions if playback_type == "no_fail" and _recorded_empty_grasp(a)
        ]
        self.paths = {}
        self.event_reports = []
        self.trace = []
        self.start_time = float(data.time)
        self.mass_force = np.zeros(model.nv)
        self.desired_acceleration = np.zeros(model.nv)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.final_goal = _waypoint(actions[-1]["after"])
        if playback_type == "no_fail":
            # A trailing failed grasp (or a log consisting only of failed
            # grasps) should leave the arm at its last retained destination.
            self.final_goal = _waypoint(actions[0]["before"])
            if self.events:
                last = self.events[-1]
                self.final_goal = last["waypoints"][-1] if last["kind"] == "move" else last["waypoint"]
        # Fail before any execution if the saved poses and compiled robot differ.
        for action in actions:
            for name in ("before", "after"):
                p = _waypoint(action[name])
                if np.any(p["q"] < self.limits[:, 0] - 1e-6) or np.any(p["q"] > self.limits[:, 1] + 1e-6):
                    raise ValueError("Recorded arm endpoint violates a joint limit")
                self.kinematics.qpos[self.arm_qpos] = p["q"]
                mujoco.mj_kinematics(model, self.kinematics)
                actual = self.kinematics.site_xpos[self.eef_id]
                rotation = Rotation.from_matrix(self.kinematics.site_xmat[self.eef_id].reshape(3, 3))
                if (np.linalg.norm(actual-p["pose"][:3]) > .001 or
                        (rotation * Rotation.from_quat(p["pose"][3:]).inv()).magnitude() > .01):
                    raise ValueError("Recorded pose/joint endpoints do not match the compiled robot model")
        initial = _waypoint(actions[0]["before"])
        if np.max(abs(data.qpos[self.arm_qpos] - initial["q"])) > 1e-5:
            raise ValueError("Initial simulation state does not match the first robot endpoint")
        forces = actions[0]["before"].get("finger_actuator_forces_n", [0, 0])
        self.closed = float(np.mean(forces)) < -1.0
        for i, event in enumerate(self.events):
            if event["kind"] == "move":
                self.paths[i] = JointPath(self, event["waypoints"], settings)

    def _gripper(self):
        # Force limits come from the recorded model/configuration, not current
        # repository defaults. Old position-servo models remain supported.
        model, data, ids = self.model, self.data, self.finger_actuators
        if np.allclose(model.actuator_gainprm[ids, 0], 1):
            kp = float(self.grip.get("kp", 6000))
            close = float(self.grip.get("close_force_n", abs(model.actuator_forcerange[ids, 0]).min()))
            opening = float(self.grip.get("open_force_n", min(40, model.actuator_forcerange[ids, 1].min())))
            goal = -close/kp if self.closed else .04
            increment = float(self.grip.get("speed_m_s", .04)) * model.opt.timestep
            self.grip_reference += np.clip(goal-self.grip_reference, -increment, increment)
            position = data.qpos[self.finger_qpos]
            ref = self.grip_reference.copy()
            if self.closed:
                ref = np.where(position < .001, np.maximum(ref, 0), ref)
            data.ctrl[ids] = np.clip(kp*(ref-position), -close, opening)
        else:
            goal = 0.0 if self.closed else .04
            increment = float(self.grip.get("speed_m_s", .04)) * model.opt.timestep
            self.grip_reference += np.clip(goal-self.grip_reference, -increment, increment)
            data.ctrl[ids] = self.grip_reference

    def step(self, q, velocity, acceleration, *, phase, event, cruise=False):
        import mujoco

        model, data, ids = self.model, self.data, self.arm_actuators
        dt = model.opt.timestep
        # PD position servos with velocity and inertial feedforward track the
        # *new* trajectory; joint-error integration compensates an unknown load.
        self.bias = np.clip(self.bias + 5*(q-data.qpos[self.arm_qpos])*dt, -.05, .05)
        self.desired_acceleration[self.arm_dofs] = acceleration
        mujoco.mj_mulM(model, data, self.mass_force, self.desired_acceleration)
        feed = (self.mass_force + data.qfrc_bias - data.qfrc_passive)[self.arm_dofs]
        kp = model.actuator_gainprm[ids, 0]
        kd = -model.actuator_biasprm[ids, 2]
        control = q + kd/kp * velocity + feed/kp + self.bias
        data.ctrl[ids] = np.clip(control, model.actuator_ctrlrange[ids, 0], model.actuator_ctrlrange[ids, 1])
        self._gripper()
        before = data.site_xpos[self.eef_id].copy()
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Endpoint playback produced a non-finite state")
        self.trace.append({"time": float(data.time-self.start_time), "target_q": q.copy(),
                           "target_velocity": velocity.copy(), "target_acceleration": acceleration.copy(),
                           "actual_q": data.qpos[self.arm_qpos].copy(),
                           "actual_velocity": data.qvel[self.arm_dofs].copy(),
                           "eef_xyz": data.site_xpos[self.eef_id].copy(),
                           "tool_speed": float(np.linalg.norm(data.site_xpos[self.eef_id]-before)/dt),
                           "phase": phase, "event": event, "cruise": cruise})
        self.kinematics.qpos[self.arm_qpos] = q
        mujoco.mj_kinematics(model, self.kinematics)
        mujoco.mj_comPos(model, self.kinematics)
        mujoco.mj_jacSite(model, self.kinematics, self.jacp, self.jacr, self.eef_id)
        self.trace[-1].update(target_eef_xyz=self.kinematics.site_xpos[self.eef_id].copy(),
                              target_tool_speed=float(np.linalg.norm(self.jacp[:, self.arm_dofs] @ velocity)))

    def execute(self):
        """Yield after every physics step so the caller can render/present it."""
        dt = self.model.opt.timestep
        zero = np.zeros(7)
        for index, event in enumerate(self.events):
            start = float(self.data.time)
            if event["kind"] == "move":
                path = self.paths[index]
                # Uniform dt with a final exact reference; never write live qpos.
                for i in range(1, int(np.ceil(path.profile.duration/dt))+1):
                    q, velocity, acc, cruise = path.sample(min(i*dt, path.profile.duration))
                    self.step(q, velocity, acc, phase="move", event=index, cruise=cruise)
                    yield "MOVE"
                goal = path.goal
            else:
                goal = event["waypoint"]
                self.closed = event["close"]
                self.grip_reference = np.clip(self.data.qpos[self.finger_qpos].copy(), 0, .04)
                widths = deque()
                ready = False
                for _ in range(int(np.ceil(float(self.grip.get("timeout_s", 4))/dt))):
                    self.step(goal["q"], zero, zero, phase="gripper", event=index)
                    yield "GRASP" if self.closed else "RELEASE"
                    positions = self.data.qpos[self.finger_qpos]
                    width = float(positions.sum())
                    widths.append((self.data.time, width))
                    while len(widths) > 1 and widths[0][0] < self.data.time-.1:
                        widths.popleft()
                    quiet = (self.data.time-widths[0][0] >= .095 and
                             max(w for _, w in widths)-min(w for _, w in widths) < .0002)
                    if self.closed:
                        force = float(self.grip.get("close_force_n", abs(self.model.actuator_forcerange[self.finger_actuators, 0]).min()))
                        if np.allclose(self.model.actuator_gainprm[self.finger_actuators, 0], 1):
                            ready = np.all(positions < .0005) or np.all(self.data.actuator_force[self.finger_actuators] <= -.95*force)
                        else:
                            ready = np.all(self.grip_reference < 1e-6)
                    else:
                        ready = np.all(abs(positions-.04) < .0005)
                    if ready and quiet:
                        break
                else:
                    raise RuntimeError(f"Playback gripper event {index} did not settle")
            # A feedback barrier only at turns/grips, never at joined collinear
            # endpoints. It depends on present tracking error, not old timings.
            for _ in range(int(np.ceil(3/dt))):
                error = np.linalg.norm(self.data.site_xpos[self.eef_id]-goal["pose"][:3])
                actual_rotation = Rotation.from_matrix(self.data.site_xmat[self.eef_id].reshape(3, 3))
                angle = (actual_rotation * Rotation.from_quat(goal["pose"][3:]).inv()).magnitude()
                joint_error = np.max(abs(self.data.qpos[self.arm_qpos]-goal["q"]))
                if (error < .0005 and angle < .005 and joint_error < .002
                        and np.max(abs(self.data.qvel[self.arm_dofs])) < .02):
                    break
                self.step(goal["q"], zero, zero, phase="settle", event=index)
                yield "SETTLE"
            else:
                raise RuntimeError(f"Playback arm event {index} did not converge ({error:.4f} m)")
            self.event_reports.append({"kind": event["kind"], "action_ids": event["action_ids"],
                                       "start_time_s": start-self.start_time,
                                       "end_time_s": float(self.data.time-self.start_time),
                                       "position_error_m": float(error),
                                       "rotation_error_rad": float(angle), "joint_error_rad": float(joint_error),
                                       "gripper_width_m": float(self.data.qpos[self.finger_qpos].sum()),
                                       **({"path_length_m": path.length_m,
                                           "planned_duration_s": path.profile.duration,
                                           "joined_waypoints": len(event["waypoints"])-1}
                                          if event["kind"] == "move" else {"close": self.closed})})
        # Let released objects settle at the end under physics, not saved states.
        goal_q = self.final_goal["q"]
        for _ in range(int(np.ceil(.5/dt))):
            self.step(goal_q, zero, zero, phase="final_hold", event=-1)
            yield "FINAL"

    def save_trace(self, path):
        arrays = {name: np.asarray([row[name] for row in self.trace]) for name in self.trace[0]} if self.trace else {}
        np.savez_compressed(path, **arrays)


def load_actions(run_dir):
    path = Path(run_dir) / "action_endpoints.jsonl"
    if not path.exists():
        raise FileNotFoundError("Endpoint playback needs action_endpoints.jsonl; record a new run first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
