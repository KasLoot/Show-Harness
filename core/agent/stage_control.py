from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.v0_types import SkillContext
from core.record.images import vlm_camera_views


class Controller:
    """One VLM call per step that returns the executed base/grasp token."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def last_prompt(self) -> str:
        """The most recent fully-rendered controller prompt (for periodic logging)."""
        return getattr(self.agent, "last_prompt", "")

    @property
    def last_media(self):
        return getattr(self.agent, "last_prompt_media", None)

    def decide(
        self,
        ctx: SkillContext,
        recent_moves: str,
        previous_direction: str,
        gripper_state: str,
        recovery_context: str = "",
        prev_agentview: Any = None,
        visual_history=(),
    ):
        views = vlm_camera_views(ctx.obs, ctx.agentview, ctx.wrist)
        extra_images = [frame for _, frame in views[1:]]
        wrist_images = (extra_images[0] if len(extra_images) == 1 else extra_images or None)
        return self.agent.decide(
            task=ctx.task,
            subgoal=ctx.subgoal.to_prompt_dict(),
            recent_moves=recent_moves,
            previous_direction=previous_direction,
            gripper_state=gripper_state,
            agentview_image=views[0][1],
            wrist_image=wrist_images,
            # Frame captured BEFORE the previous action executed (action-ablation
            # blind review); None everywhere else, incl. the sim runner.
            prev_agentview_image=prev_agentview,
            proprio=ctx.proprio,
            recovery_context=recovery_context,
            visual_history=visual_history,
            current_step_idx=ctx.step_idx,
            current_view_names=[name for name, _ in views],
            current_depth_text=str(ctx.obs.get("wrist_depth_text") or ""),
            debug=ctx.debug,
        )


@dataclass(frozen=True)
class StageControlSuite:
    controller: Controller
