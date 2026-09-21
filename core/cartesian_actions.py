"""Explicit single-command Cartesian sizes for the MuJoCo insertion task.

This is an opt-in vocabulary extension. The shared hardware action vocabulary and
the planner/controller decision loop remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import MOVE_ATOMS, STOP_ATOM
from core.sim.mujoco_depth import depth_prompt


SIZES = ("small", "medium", "large")
DEFAULT_TRANSLATION_STEPS_M = {"small": 0.002, "medium": 0.01, "large": 0.05}
DEFAULT_ROTATION_STEPS_DEG = {"small": 2.0, "medium": 10.0, "large": 30.0}


@dataclass(frozen=True)
class CartesianAction:
    kind: str
    size: str
    base_token: str = ""
    axis: int = -1
    sign: int = 0


class CartesianActions:
    """One source for configured sizes, accepted tokens, and VLM instructions."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = {} if config is None else config
        if not isinstance(config, Mapping):
            raise ValueError("cartesian_motion must be a mapping")
        if "cartesian_motion" in config:
            config = config["cartesian_motion"]
            if not isinstance(config, Mapping):
                raise ValueError("cartesian_motion must be a mapping")
        self.translation_steps_m = self._sizes(
            config, "translation_steps_m", DEFAULT_TRANSLATION_STEPS_M,
        )
        self.rotation_steps_deg = self._sizes(
            config, "rotation_steps_deg", DEFAULT_ROTATION_STEPS_DEG,
        )
        if self.rotation_steps_deg["large"] >= 180:
            raise ValueError("rotation_steps_deg must be below 180 degrees: pose endpoints "
                             "execute the shortest rotation")
        self._actions = {
            f"{token}_{size.upper()}": CartesianAction("move", size, base_token=token)
            for token in MOVE_ATOMS for size in SIZES
        }
        self._actions.update({
            f"ROT_{axis}_{direction}_{size.upper()}": CartesianAction(
                "rotate", size, axis=axis_idx, sign=sign,
            )
            for axis_idx, axis in enumerate("XYZ")
            for direction, sign in (("POS", 1), ("NEG", -1))
            for size in SIZES
        })

    @staticmethod
    def _sizes(config: Mapping[str, Any], name: str, defaults: dict[str, float]) -> dict[str, float]:
        raw = config.get(name, defaults)
        if not isinstance(raw, Mapping) or set(raw) != set(SIZES):
            raise ValueError(f"cartesian_motion.{name} must define small, medium, and large")
        try:
            values = {size: float(raw[size]) for size in SIZES}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"cartesian_motion.{name} sizes must be finite positive numbers") from exc
        if (any(isinstance(raw[size], bool) for size in SIZES)
                or not all(math.isfinite(v) for v in values.values())
                or not 0 < values["small"] < values["medium"] < values["large"]):
            raise ValueError(f"cartesian_motion.{name} requires finite 0 < small < medium < large")
        return values

    def action_tokens(self) -> tuple[str, ...]:
        """Additional tokens to append to the existing controller vocabulary."""
        return (*self._actions, STOP_ATOM)

    def decode(self, token: str) -> CartesianAction | None:
        """Decode an explicit move/rotation; return None for legacy or unknown tokens."""
        return self._actions.get(token.strip().upper())

    def render_prompt(self, proprio: Mapping[str, Any] | None = None) -> str:
        """Describe the action contract; fixed axes do not depend on wrist orientation."""
        translations = ", ".join(
            f"{size.upper()}={self.translation_steps_m[size] * 1000:g} mm" for size in SIZES
        )
        rotations = ", ".join(
            f"{size.upper()}={self.rotation_steps_deg[size]:g} degrees" for size in SIZES
        )
        pose_context = ""
        if proprio is not None and proprio.get("eef_quat") is not None:
            try:
                quat = np.asarray(proprio["eef_quat"], dtype=float)
                if quat.shape == (4,) and np.isfinite(quat).all():
                    rotation = Rotation.from_quat(quat)
                    directions = (
                        ("hand/tool +Z", [0, 0, 1]),
                        ("wrist image-right", [0, -1, 0]),
                        ("wrist image-down", [1, 0, 0]),
                    )
                    vectors = "; ".join(
                        f"{label}=[{', '.join(f'{v:+.3f}' for v in rotation.apply(axis))}]"
                        for label, axis in directions
                    )
                    pose_context = (
                        f"\nCurrent measured orientation, world [X,Y,Z] unit vectors: {vectors}. "
                        "Use these current wrist directions with the fixed world movement axes; "
                        "image-down may include vertical motion after rotation."
                    )
            except (TypeError, ValueError):
                # Missing/invalid orientation must not prevent forming a prompt;
                # the camera observations remain available to the controller.
                pass
        if proprio is not None and isinstance(proprio.get("insertion_camera_axes"), Mapping):
            try:
                axes = proprio["insertion_camera_axes"]
                vectors = np.asarray([axes[key] for key in ("image_right", "image_down", "sightline")],
                                     dtype=float)
                if vectors.shape == (3, 3) and np.isfinite(vectors).all():
                    calibration = "; ".join(
                        f"Angled Wrist {label}=[{', '.join(f'{v:+.3f}' for v in vector)}]"
                        for label, vector in zip(("image-right", "image-down", "sightline"), vectors)
                    )
                    pose_context += (
                        f"\nCurrent measured Angled Wrist camera axes, world [X,Y,Z]: {calibration}. "
                        "These axes rotate with the hand. Camera translation makes stationary scene "
                        "features move oppositely to its image-plane component; motion along its "
                        "sightline changes depth and perspective. The held pin travels with the hand. "
                        "The Angled Wrist image center is below the TCP, not the grasp point."
                    )
            except (KeyError, TypeError, ValueError):
                pass
        if proprio is not None and isinstance(proprio.get("wrist_depth_calibration"), Mapping):
            pose_context += "\n" + depth_prompt(proprio["wrist_depth_calibration"])
        return (
            "EXPLICIT CARTESIAN ACTIONS (one token per decision)\n"
            "Choose a translation direction and size: "
            "MV_FWD_SMALL/MEDIUM/LARGE, MV_BACK_SMALL/MEDIUM/LARGE, "
            "MV_LEFT_SMALL/MEDIUM/LARGE, MV_RIGHT_SMALL/MEDIUM/LARGE, "
            "MV_UP_SMALL/MEDIUM/LARGE, MV_DOWN_SMALL/MEDIUM/LARGE. "
            f"Translation distances: {translations}.\n"
            "All directions use fixed robot base/world axes: "
            "FWD=+X, BACK=-X, RIGHT=+Y, LEFT=-Y, UP=+Z, DOWN=-Z. "
            "They remain fixed after wrist rotation; the wrist image rotates with the hand.\n"
            "Rotate about any fixed world axis at the current TCP position using "
            "ROT_X_POS_SMALL/MEDIUM/LARGE, ROT_X_NEG_SMALL/MEDIUM/LARGE, "
            "ROT_Y_POS_SMALL/MEDIUM/LARGE, ROT_Y_NEG_SMALL/MEDIUM/LARGE, "
            "ROT_Z_POS_SMALL/MEDIUM/LARGE, ROT_Z_NEG_SMALL/MEDIUM/LARGE. "
            f"Rotation angles: {rotations}. "
            "POS follows the right-hand rule about the named positive axis; NEG reverses it. "
            "For example ROT_Y_POS_SMALL is a single small rotation about world +Y. "
            "Rotations compose in execution order; translations and lifts preserve the full "
            "current orientation, including while holding an object.\n"
            "Use LARGE only in clear space, MEDIUM for approach/alignment, and SMALL for "
            "final alignment and insertion. Inspect the next observation after every action. "
            "GRASP closes, RELEASE opens, STOP holds position, and DONE ends the current stage. "
            "Bare MV_* tokens remain available at the configured normal translation step."
            + pose_context
        )
