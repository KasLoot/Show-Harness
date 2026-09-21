"""Offline replay of recorded MuJoCo actions through the original controller.

Rollouts contain commands and TCP pose checks, not full MuJoCo state trajectories.
Replay therefore re-simulates actions with the current scene assets and physics.
It never interpolates saved TCP poses or bypasses contact dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import ATOMIC_ACTIONS, DONE_ATOM, GRIPPER_ATOMS
from core.cartesian_actions import CartesianActions
from plugins.config import PluginsConfig


@dataclass(frozen=True)
class MujocoReplayPlan:
    run_dir: Path
    config: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    @property
    def variable_step(self) -> bool:
        return PluginsConfig.from_config(self.config).enabled("variable_step", default=False)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path.name}") from exc


def _pose(value: Any, label: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain seven finite pose values") from exc
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) == 0:
        raise ValueError(f"{label} must be finite [x,y,z,qx,qy,qz,qw] with a nonzero quaternion")
    return pose


def load_replay_plan(run_dir: str | Path, *, require_poses: bool = False) -> MujocoReplayPlan:
    """Validate all input before any simulator/renderer or output file is opened."""
    run_dir = Path(run_dir).resolve()
    metadata = _read_json(run_dir / "metadata.json")
    if not isinstance(metadata, dict) or metadata.get("simulator") != "mujoco":
        raise ValueError("metadata.json must identify simulator='mujoco'; hardware rollouts are not supported")
    cfg = metadata.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("metadata.json is missing its recorded config")
    if cfg.get("scene", "pick_place") not in ("pick_place", "plug_insert"):
        raise ValueError("Unsupported recorded MuJoCo scene")
    if "reset_qpos" not in cfg:
        raise ValueError("Recorded config is missing reset_qpos; the initial state cannot be reconstructed")
    try:
        joints = np.asarray(cfg["reset_qpos"], dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError
        for key in ("fine_step_m", "empty_width_m", "open_width_m"):
            if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
                raise ValueError
        if not math.isfinite(float(cfg["z_floor_m"])):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Recorded config needs seven finite reset_qpos joints and finite motion/gripper settings") from exc
    plugins = cfg.get("plugins") or {}
    if not isinstance(plugins, dict):
        raise ValueError("Recorded plugins config must be a mapping")
    if cfg.get("tools") is not None and not isinstance(cfg["tools"], dict):
        raise ValueError("Recorded legacy tools config must be a mapping")
    enabled = PluginsConfig.from_config(cfg)
    for name in ("action_chunk", "dagger", "rotation", "smooth"):
        if enabled.enabled(name, default=False):
            raise ValueError(f"Replay does not support recorded plugins.{name}")
    if cfg.get("motion_frame", "base") != "base":
        raise ValueError("Replay only supports the fixed base/world motion frame")
    if cfg.get("action_ablation_mode", "off") not in (None, "", "off"):
        raise ValueError("Replay requires explicit action tokens; action ablation is unsupported")
    actions = CartesianActions(cfg["cartesian_motion"]) if cfg.get("cartesian_motion") is not None else None
    if actions is not None and enabled.enabled("variable_step", default=False):
        raise ValueError("Recorded Cartesian motion cannot also enable variable_step")
    allowed = set(ATOMIC_ACTIONS) | set(GRIPPER_ATOMS) | {DONE_ATOM}
    if actions is not None:
        allowed.update(actions.action_tokens())
    source = run_dir / "steps.jsonl"
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("The run must contain steps.jsonl") from exc
    records = []
    previous_index = -1
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"steps.jsonl line {line_number} is malformed or truncated") from exc
        if not isinstance(record, dict):
            raise ValueError(f"steps.jsonl line {line_number} must be an object")
        index = record.get("i")
        if isinstance(index, bool) or not isinstance(index, int) or index != previous_index + 1:
            raise ValueError(f"steps.jsonl line {line_number} must have contiguous step i={previous_index + 1}; "
                             "partial or missing action sequences cannot be replayed")
        previous_index = index
        token = record.get("act")
        if not isinstance(token, str) or token not in allowed:
            raise ValueError(f"Step {index} contains an unsupported action token")
        if record.get("src") == "human" or record.get("realign"):
            raise ValueError(f"Step {index} contains unsupported human or implicit realignment motion")
        if "target_in_wrist" in record and record["target_in_wrist"] is not None and not isinstance(record["target_in_wrist"], bool):
            raise ValueError(f"Step {index} target_in_wrist must be boolean or null")
        for field in ("step_cm", "rotation_deg"):
            if field in record:
                try:
                    value = float(record[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Step {index} {field} must be finite and nonnegative") from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Step {index} {field} must be finite and nonnegative")
        for field in ("pre_pose", "post_pose", "target_pose"):
            if field in record:
                _pose(record[field], f"Step {index} {field}")
            elif require_poses and field in ("pre_pose", "post_pose"):
                raise ValueError(f"Step {index} lacks {field}; --verify requires recorded measured TCP poses")
        recovery = record.get("recovery", {})
        if not isinstance(recovery, dict):
            raise ValueError(f"Step {index} recovery must be an object")
        if "release" in recovery and not isinstance(recovery["release"], bool):
            raise ValueError(f"Step {index} recovery.release must be boolean")
        # A before-decision recovery token already appears as record.act. Only
        # recovery.release represents an ADDITIONAL post-action command.
        recovery_token = recovery.get("token")
        if recovery_token is not None and recovery_token != token:
            raise ValueError(f"Step {index} has an ambiguous recovery token that differs from act")
        if record.get("recover") and not recovery:
            raise ValueError(f"Step {index} reports recovery without its command details")
        records.append(record)
    if not records:
        raise ValueError("steps.jsonl contains no actions")
    return MujocoReplayPlan(run_dir, cfg, tuple(records))


def pose_error(actual: Any, expected: Any) -> dict[str, float]:
    """Position norm and quaternion-sign-invariant orientation difference."""
    actual = _pose(actual, "Actual pose")
    expected = _pose(expected, "Recorded pose")
    rotation = Rotation.from_quat(actual[3:]) * Rotation.from_quat(expected[3:]).inv()
    return {
        "position_m": float(np.linalg.norm(actual[:3] - expected[:3])),
        "rotation_deg": float(np.rad2deg(rotation.magnitude())),
    }


def validate_tolerances(position_tolerance_m: float, rotation_tolerance_deg: float) -> None:
    if not all(math.isfinite(value) and value >= 0
               for value in (position_tolerance_m, rotation_tolerance_deg)):
        raise ValueError("Pose tolerances must be finite and nonnegative")


def execute_replay(
    plan: MujocoReplayPlan,
    session: Any,
    controller: Any,
    *,
    verify: bool = False,
    position_tolerance_m: float = 0.0001,
    rotation_tolerance_deg: float = 0.1,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute file-order tokens; check post poses BEFORE additional recovery release.

    The caller owns session/recorder lifetime. No VLM, recovery policy, settling,
    rendering, or extra physics steps are introduced between recorded commands.
    Empty-grasp auto-reopen remains inside the original executor.
    """
    validate_tolerances(position_tolerance_m, rotation_tolerance_deg)
    report = {} if report is None else report
    report.update(
        source_run=str(plan.run_dir), mode="action_replay", status="running",
        description="Re-simulates recorded actions with current MuJoCo assets; no full saved simulator-state trajectory.",
        verification_enabled=verify, position_tolerance_m=position_tolerance_m,
        rotation_tolerance_deg=rotation_tolerance_deg, actions_replayed=0,
        max_position_error_m=0.0, max_rotation_error_deg=0.0, comparisons=0, steps=[],
    )
    recorder = getattr(session, "recorder", None)

    def check(index: int, phase: str, actual: Any, expected: Any, entry: dict) -> None:
        if expected is None:
            if verify:
                raise ValueError(f"Step {index} lacks {phase}_pose required by --verify")
            return
        error = pose_error(actual, expected)
        entry[f"{phase}_pose_error"] = error
        report["comparisons"] += 1
        report["max_position_error_m"] = max(report["max_position_error_m"], error["position_m"])
        report["max_rotation_error_deg"] = max(report["max_rotation_error_deg"], error["rotation_deg"])
        if verify and (error["position_m"] > position_tolerance_m
                       or error["rotation_deg"] > rotation_tolerance_deg):
            report["status"] = "pose_mismatch"
            raise RuntimeError(
                f"Replay diverged at step {index} ({phase} action): "
                f"position error {error['position_m'] * 1000:.4f} mm, "
                f"orientation error {error['rotation_deg']:.4f} degrees"
            )

    try:
        controller.step("RELEASE")  # Same startup command as RealEpisodeRunner.run.
        for record in plan.records:
            index, token = record["i"], record["act"]
            entry = {"i": index, "action": token, "recovery_release": False}
            report["steps"].append(entry)
            check(index, "pre", session.get_ee_pose(), record.get("pre_pose"), entry)
            if recorder is not None:
                reasoning = ((record.get("vlm") or {}).get("c") or {}).get("why", "")
                recorder.begin_step(step_idx=index, stage=str(record.get("stage", "REPLAY")),
                                    token=token, annotation=str(reasoning or "Recorded action replay; no VLM call."),
                                    output=str(reasoning), source="replay")
            controller.step(token, target_in_wrist=record.get("target_in_wrist"), continuous=False)
            report["actions_replayed"] += 1
            actual = np.asarray(session.get_ee_pose(), dtype=float)
            entry["actual_post_pose"] = actual.tolist()
            check(index, "post", actual, record.get("post_pose"), entry)
            if (record.get("recovery") or {}).get("release", False):
                controller.step("RELEASE")
                entry["recovery_release"] = True
            if recorder is not None:
                recorder.end_step(record)
        report["task_success"] = bool(session.check_success())
        diagnostics = getattr(session, "success_diagnostics", None)
        if callable(diagnostics):
            report["success_diagnostics"] = diagnostics()
        report["status"] = "completed"
        return report
    except BaseException as exc:
        if report["status"] == "running":
            report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = str(exc)
        raise
