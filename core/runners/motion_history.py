"""Measured robot motion paired with historical observations, without object state."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, size):
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    return array if array.size == size and np.isfinite(array).all() else None


def _format(vector):
    return "[" + ", ".join(f"{value:+.2f}" for value in vector) + "]"


def motion_history_effect(result) -> str:
    """Describe requested vs measured TCP motion; never infer that a plug moved."""
    if result is None:
        return "No controller motion result; measured effect unavailable."
    parts = []
    requested = _vector(getattr(result, "intended_delta_m", None), 3)
    if requested is not None:
        parts.append("Requested TCP translation world [X,Y,Z] mm=" + _format(requested * 1000))
    rotation = _vector(getattr(result, "intended_rotation_rad", None), 3)
    if rotation is None:
        rotation = np.array([0.0, 0.0, getattr(result, "intended_yaw_rad", 0.0)])
    parts.append("requested world rotation-vector deg=" + _format(np.rad2deg(rotation)))
    before = _vector(getattr(result, "pre_pose", None), 7)
    after = _vector(getattr(result, "post_pose", None), 7)
    if before is not None and after is not None:
        parts.append("measured TCP translation world [X,Y,Z] mm=" + _format((after[:3] - before[:3]) * 1000))
        if np.linalg.norm(before[3:]) > 0 and np.linalg.norm(after[3:]) > 0:
            actual_rotation = Rotation.from_quat(after[3:]) * Rotation.from_quat(before[3:]).inv()
            parts.append("measured world rotation-vector deg=" + _format(actual_rotation.as_rotvec(degrees=True)))
    else:
        parts.append("measured TCP motion unavailable")
    closed = getattr(result, "gripper_closed", None)
    if closed is not None:
        parts.append("controller gripper state=" + ("CLOSED" if closed else "OPEN"))
    if getattr(result, "grasp_empty", False):
        parts.append("empty grasp detected; no verified hold")
    note = str(getattr(result, "note", "") or "").strip()
    if note:
        parts.append("executor feedback: " + note)
    parts.append("TCP motion alone does not establish object motion, alignment, or insertion")
    return "; ".join(parts) + "."
