"""Pair earlier camera snapshots with executed actions in one multimodal request."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.record.images import frame_fingerprint


@dataclass(frozen=True)
class VisualHistoryEntry:
    """Views captured BEFORE an executed action, plus its measured outcome.

    The runner owns the bounded queue and copies the arrays when taking snapshots.
    Stage changes may keep entries: their stage and step explicitly label them as past.
    """

    step_idx: int
    stage: str
    views: Sequence[tuple[str, np.ndarray]]
    action: str
    effect: str
    # Immutable serialized sensor array/calibration from the same pre-action
    # observation. It stays text and never consumes a camera/image slot.
    depth_text: str = ""


_CAMERA_NAMES = {
    "Wrist": "wrist", "Front": "front", "Right Side": "side", "Side": "side",
    "Angled Wrist": "wrist_insert", "Wrist Depth": "wrist_depth", "AgentView": "agentview",
}
_DISPLAY_NAMES = {
    "wrist": "Wrist", "front": "Front", "side": "Right Side",
    "wrist_insert": "Angled Wrist", "wrist_depth": "Wrist Depth", "agentview": "AgentView",
}


def compose_visual_history(
    current_views: Sequence[tuple[str, np.ndarray]],
    history: Sequence[VisualHistoryEntry] = (),
    *,
    current_step_idx: int | None = None,
) -> tuple[list[np.ndarray], str, list[dict]]:
    """Keep current views first, then old snapshots oldest-first, with an image map.

    Numbering is the actual image order in both Chat Completions and native Ollama.
    Labels live in the single text part so provider conversion cannot separate them
    from their image numbers. The same prompt/images also survive strict retries.
    """
    images: list[np.ndarray] = []
    media: list[dict] = []
    current_labels: list[str] = []

    def append(name: str, image: np.ndarray, *, offset: int, temporal: str) -> str:
        camera = _CAMERA_NAMES.get(name, name)
        display = _DISPLAY_NAMES.get(camera, name)
        slot = len(images)
        images.append(image)
        media.append({
            "slot": slot, "part_type": "image_url", "placeholder": "<image>",
            "camera": camera, "t_offset": offset, "temporal": temporal,
            **frame_fingerprint(image),
        })
        return f"image {slot + 1} = {display}"

    for name, image in current_views:
        if image is not None:
            current_labels.append(append(name, image, offset=0, temporal="current"))
    if not history:
        return images, "", media
    if not images:
        raise ValueError("Visual history requires current images")
    if current_step_idx is None:
        raise ValueError("current_step_idx is required to label visual history")
    text = [
        "VISUAL ACTION HISTORY (image numbers are 1-based attachment order)",
        f"CURRENT step {current_step_idx}: " + "; ".join(current_labels) + ".",
        "The CURRENT views remain A, B, C, etc. in their usual order. "
        "The additional sets below are PAST observations, not extra current cameras.",
        "Each past set was captured BEFORE its listed executed action. Compare it with "
        "the next past set, or CURRENT for the latest action, to judge the visible effect. "
        "Use the measured outcome to distinguish requested motion from actual motion. "
        "Historical stages describe past actions; choose the next action for the CURRENT stage. "
        "Compare the same camera across time and account for wrist rotation or occlusion. "
        "The current images determine whether an action is safe now.",
    ]
    for entry in sorted(history, key=lambda item: item.step_idx):
        if entry.step_idx >= current_step_idx:
            raise ValueError("Visual history must precede the current step")
        labels = [
            append(name, image, offset=entry.step_idx - current_step_idx, temporal="before_action")
            for name, image in entry.views if image is not None
        ]
        text += [
            f"HISTORY step {entry.step_idx}, stage {entry.stage}, BEFORE action {entry.action}: "
            + "; ".join(labels) + ".",
            f"Executed after this snapshot: {entry.action}. Measured outcome: {entry.effect}",
        ]
        if entry.depth_text:
            text.append(
                f"HISTORY WRIST DEPTH (step {entry.step_idx}, BEFORE action {entry.action}):\n"
                + entry.depth_text
            )
    return images, "\n".join(text), media
