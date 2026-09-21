"""Real-robot VLM closed-loop rollout runner.

This is the real-robot counterpart of :class:`core.subgoal_runner.EpisodeRunnerV0`.
A simulator runner drives an ``env`` (normalized actions, ``env.step`` /
``env.check_success``); here we drive a physical Franka through a
:class:`~core.franka.franka_session.FrankaSession` (camera + pose observations)
and a :class:`~interpreters.franka_atomic_controller.FrankaAtomicController`
(atomic tokens -> Cartesian-impedance setpoints, with the Z safety floor).

The control contract is identical to the simulator: the planner expands the task
into ordered subgoals, and for each subgoal the controller VLM emits exactly one
atomic token per step (``MV_*`` / ``GRASP`` / ``RELEASE`` / ``DONE``). A ``DONE``
advances to the next subgoal; ``DONE`` on the final subgoal completes the task
(there is no simulator ``check_success`` on hardware -- the VLM judging every stage
done, including the final place, is the success signal).

Logging reuses :class:`~core.record.episode_logger.EpisodeLogger`, so the on-disk layout,
per-step analysis frames, and the compiled visualization video keep one shared
convention byte-for-byte:

    <log_dir>/<variant>/<MMDD>/task_<id>/<HH-MM-SS>/rollout_{success,failure}.mp4

Safety + interruption:
  * The :class:`FrankaAtomicController` enforces the Z floor on every motion, so
    ``MV_DOWN`` can never press the arm below the locked tabletop height.
  * ``logger.close()`` (which compiles the video) runs in a ``finally`` block, so a
    ``Ctrl+C`` (``KeyboardInterrupt``) mid-rollout still writes the video and the
    summary before tearing down -- exactly like ``EpisodeRunnerV0``.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import numpy as np

import core.ui.console as console
from core.action_units import MOVE_ATOMS, ROTATE_ATOMS
from interpreters.franka_atomic_controller import AtomicStepResult, FrankaAtomicController
from core.runners.preemption import InterruptibleDecider
from core.runners.motion_history import motion_history_effect
from core.record.episode_logger import EpisodeLogger, status_flags
from core.franka.franka_session import FrankaSession
from core.record.images import save_png, to_uint8_hwc, vlm_camera_views
from core.sim.mujoco_depth import depth_prompt
from core.v0_types import EpisodeResult, SkillContext, Subgoal, V0Config
from core.visual_history import VisualHistoryEntry
from plugins.subgoal import SubgoalPlanner


DONE_TOKEN = "DONE"
GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
GRIPPER_TOKENS = (GRASP_TOKEN, RELEASE_TOKEN)
# Explicit "hold this step" (a DAGGER STILL intent): no controller call, the arm
# keeps its setpoint. Never emitted by the single-arm VLM controller.
STILL_TOKEN = "STILL"
NO_DIRECTION = "NONE"
RECENT_MOVES_MAX = 3
# Move-memory label for a just-failed (empty) GRASP, so the controller's "after an empty
# GRASP, do not GRASP in place again" rule has something to key on in the recent moves.
EMPTY_GRASP_LABEL = "GRASP(empty)"


class RealEpisodeRunner:
    """Closed-loop VLM rollout on the physical Franka (no simulator)."""

    def __init__(
        self,
        *,
        session: FrankaSession,
        controller: FrankaAtomicController,
        planner: Optional[SubgoalPlanner],
        controls: Any,
        logger: EpisodeLogger,
        config: V0Config,
        task: str,
        gripper_color: str,
        max_steps: int,
        loop_period_s: float,
        use_wrist_image: bool,
        debug: bool,
        recovery_plugin: Any = None,
        action_chunk_plugin: Any = None,
        deepplan_plugin: Any = None,
        affordance_plugin: Any = None,
        dagger_plugin: Any = None,
        action_ablation_plugin: Any = None,
        recent_moves_max: int = RECENT_MOVES_MAX,
        visual_history_steps: int = 0,
        viewer: Any = None,
    ) -> None:
        self.session = session
        self.controller = controller
        self.planner = planner
        self.controls = controls
        self.logger = logger
        self.config = config
        self.task = str(task)
        self.gripper_color = str(gripper_color)
        self.max_steps = int(max_steps)
        self.loop_period_s = float(loop_period_s)
        self.use_wrist_image = bool(use_wrist_image)
        self.debug = bool(debug)
        self.recovery_plugin = recovery_plugin
        self.action_chunk_plugin = action_chunk_plugin
        # DeepPlan: when enabled, a planned <REASON> checkpoint is resolved into concrete
        # subgoals mid-episode (see _resolve_pivot). None -> no REASON is ever planned, so
        # the pivot path is unreachable (today's linear flow).
        self.deepplan_plugin = deepplan_plugin
        # Affordance dots: grounds the current stage's contact point on stage entry and
        # premarks it on the AgentView every consumer sees (controller, live view,
        # video); once the target is wrist-visible it also tracks the point on the wrist
        # frame per step. None/disabled -> the frames pass through untouched.
        self.affordance_plugin = affordance_plugin
        # Previous step's target-in-wrist marker, gating this step's wrist-dot tracking
        # (the marker for step N is only known after step N's decision).
        self._prev_target_in_wrist = False
        self.recent_moves_max = max(1, int(recent_moves_max))
        self.visual_history_steps = int(visual_history_steps)
        if self.visual_history_steps < 0:
            raise ValueError("visual_history_steps must be nonnegative")
        self.viewer = viewer
        # DAGGER: real-time human keyboard override (plugins.dagger, single mode). Keys
        # arrive on the live view's stream thread; the runner consumes them at step
        # boundaries and preempts/drops in-flight VLM decisions (see
        # _decide_interruptible). None/disabled -> the loop is byte-identical.
        self.dagger_plugin = dagger_plugin
        # Action-type ablation (plugins.action_ablation). In blind mode the runner
        # feeds the frame captured BEFORE the last direction token back into the
        # next decision (two-frame effect review) and persists the model's
        # self-written symbol->effect table as action_table.json in the rollout.
        self.action_ablation_plugin = action_ablation_plugin
        # The single in-flight preemptible decision (DAGGER only).
        self._decider: Optional[InterruptibleDecider] = None
        # Wall-clock + last VLM latency for the live window's telemetry line.
        self._t0 = time.monotonic()
        self._last_vlm_ms: Optional[int] = None
        # Last MV_DOWN's (travelled, commanded) height in meters; None when the last
        # motion was not a descent. Read by _proprio for the next prompt.
        self._descend: Optional[tuple[float, float]] = None

    # -- main loop ---------------------------------------------------------
    def run(self) -> EpisodeResult:
        recorder = getattr(self.session, "recorder", None)
        success_check = getattr(self.session, "check_success", None)
        success = False
        end_reason = "max_steps_exceeded"
        steps = 0
        current_index = 0
        subgoal_start_step = 0
        subgoals: list[Subgoal] = []
        raw_plan = ""
        recent_moves: list[str] = []
        # Pre-action camera sets paired with what actually ran after each set.
        # Keep across stage boundaries: the first insertion request needs to see
        # the preceding alignment, not start with a blank visual memory.
        visual_history: list[VisualHistoryEntry] = []
        current_direction: Optional[str] = None
        recovery_note = ""
        # Action-ablation blind review: the frame captured BEFORE the last executed
        # direction token, and that token. Valid for exactly one step; anything else
        # in between (recovery, STILL, gripper actions) clears it so a scene change
        # is never attributed to the wrong symbol.
        review_frame: Any = None
        review_token: Optional[str] = None
        # action_chunk: while the TARGET is far (not in the wrist view) the model plans its
        # next moves in one VLM call; chunk_queue holds the still-to-execute planned moves,
        # popped one per step open-loop. Empty -> re-query the VLM.
        chunk_queue: list[str] = []
        video_path: Any = self.logger.run_dir / "rollout_failure.mp4"
        self._t0 = time.monotonic()
        self._last_vlm_ms = None

        # Start from a known, open-gripper state. The controller is already synced
        # (and the Z floor locked) by the caller before the rollout begins.
        try:
            self.controller.step(RELEASE_TOKEN)
            obs = self.session.get_observation()
            agentview, wrist = self._images(obs)
            if self.planner is not None:
                # The planner names pieces/parts, so it gets the session's high-res
                # renders when available (agentview_hd/wrist_hd; glyph-level detail
                # the 256 px observation destroys). Text-only output -- no grid or
                # drawing involved, so the swap is free of coordinate concerns.
                plan_agentview = obs.get("agentview_hd")
                plan_agentview = agentview if plan_agentview is None else plan_agentview
                plan_wrist = obs.get("wrist_hd")
                # The exact frames the planner saw are saved for auditing.
                try:
                    from PIL import Image as _Image

                    _Image.fromarray(np.asarray(plan_agentview, dtype=np.uint8)).save(
                        str(self.logger.run_dir / "planner_input.png")
                    )
                except Exception:  # noqa: BLE001 - auditing must not break the run
                    pass
                live_wrist: Any = wrist if plan_wrist is None else plan_wrist
                if live_wrist is not None:
                    try:
                        from PIL import Image as _Image

                        _Image.fromarray(np.asarray(live_wrist, dtype=np.uint8)).save(
                            str(self.logger.run_dir / "planner_wrist.png")
                        )
                    except Exception:  # noqa: BLE001 - auditing must not break the run
                        pass

                views = vlm_camera_views(obs, plan_agentview, live_wrist)
                image_roles = [
                    ("LIVE Wrist view: local gripper detail only; do not infer hidden global positions."
                     if name == "Wrist" else
                     "LIVE Angled Wrist: moving hand-mounted view of the pin, black bore and rim for insertion alignment."
                     if name == "Angled Wrist" else
                     "LIVE Wrist Depth: " + depth_prompt(obs["wrist_depth_calibration"])
                     if name == "Wrist Depth" and "wrist_depth_calibration" in obs else
                     f"LIVE {name}: current external scene and object positions.")
                    for name, _ in views
                ]
                if obs.get("primary_camera") == "Front":
                    save_png(self.logger.run_dir / "planner_front.png", plan_agentview)
                extra_views = obs.get("extra_views", {})
                for name, frame in extra_views.items():
                    save_png(self.logger.run_dir / f"planner_{name}.png", frame)
                depth_text = str(obs.get("wrist_depth_text") or "")
                if depth_text:
                    (self.logger.run_dir / "planner_wrist_depth.txt").write_text(depth_text, encoding="utf-8")
                if "wrist_depth_m" in obs:
                    np.save(self.logger.run_dir / "planner_wrist_depth_m.npy", obs["wrist_depth_m"], allow_pickle=False)
                extra_images = [frame for _, frame in views[1:]]
                plan_images = (extra_images[0] if len(extra_images) == 1 else extra_images or None)

                subgoals, raw_plan = self.planner.plan(
                    self.task,
                    views[0][1],
                    wrist=plan_images,
                    debug=self.debug,
                    image_roles=image_roles,
                    **({"observation_text": "CURRENT WRIST DEPTH (initial observation):\n"
                        + depth_prompt(obs["wrist_depth_calibration"]) + "\n" + depth_text}
                       if depth_text else {}),
                )
                if not subgoals:
                    raise RuntimeError("Planner returned no subgoals for the task")

                diagnostics_fn = getattr(self.planner, "diagnostics", None)
                diagnostics = (
                    dict(diagnostics_fn() or {}) if diagnostics_fn is not None else {}
                )
                diagnostics.update(
                    {"raw_plan": raw_plan, "subgoal_count": len(subgoals)}
                )
                save_prompt = getattr(self.logger, "save_planner_prompt", None)
                prompt_fn = getattr(self.planner, "last_prompt", None)
                try:
                    if save_prompt is not None and prompt_fn is not None:
                        save_prompt(1, str(prompt_fn()))
                except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
                    print(f"[run-real] could not save planner prompt: {exc}")
                write_diagnostics = getattr(
                    self.logger, "write_planner_diagnostics", None
                )
                try:
                    if write_diagnostics is not None:
                        write_diagnostics({"attempts": [diagnostics]})
                except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
                    print(f"[run-real] could not save planner diagnostics: {exc}")
            else:
                # Subgoal tool disabled: run the whole task as one synthetic stage so the
                # controller + hardware loop still execute. Recovery can still rewind to
                # this single stage if measured width shows an empty/lost grasp.
                subgoals = [self._single_task_subgoal()]
                raw_plan = "subgoal tool disabled: single whole-task stage"
            self.logger.write_plan(
                {
                    "task": self.task,
                    "raw_plan": raw_plan,
                    "subgoals": [sg.to_prompt_dict() for sg in subgoals],
                }
            )
            self._print_plan(subgoals)

            for step_idx in range(self.max_steps):
                steps = step_idx + 1
                loop_started = time.monotonic()
                subgoal = subgoals[current_index]
                subgoal_step = step_idx - subgoal_start_step

                # Per-subgoal step cap: abandon this stage and move to the next one.
                if subgoal_step >= self.config.max_subgoal_steps:
                    current_index += 1
                    if current_index >= len(subgoals):
                        end_reason = "subgoal_step_cap_exceeded"
                        break
                    subgoal = subgoals[current_index]
                    subgoal_start_step = step_idx
                    subgoal_step = 0
                    current_direction = None
                    recent_moves.clear()
                    chunk_queue = []

                obs = self.session.get_observation()
                agentview, wrist = self._images(obs)
                agentview, wrist = self._premark_affordance(
                    agentview, wrist, subgoal, current_index, obs=obs
                )
                self._show_live(
                    step_idx,
                    agentview,
                    wrist,
                    "deciding ...",
                    steps_done=step_idx,
                    subgoal=subgoal,
                    subgoal_index=current_index,
                    n_subgoals=len(subgoals),
                )
                ctx = SkillContext(
                    task=self.task,
                    subgoal=subgoal,
                    subgoal_index=current_index,
                    step_idx=step_idx,
                    subgoal_step_idx=subgoal_step,
                    obs=obs,
                    agentview=agentview,
                    wrist=wrist,
                    proprio=self._proprio(obs),
                    debug=self.debug,
                )
                history_views = tuple(
                    (name, np.array(frame, copy=True))
                    for name, frame in vlm_camera_views(obs, agentview, wrist)
                ) if self.visual_history_steps else ()
                history_depth_text = str(obs.get("wrist_depth_text") or "")

                # DeepPlan: execution has reached a <REASON> checkpoint -- a conditional
                # branch deferred at plan time. Resolve it from the LIVE scene and splice the
                # concrete stages in before controlling. Disabled/absent -> no REASON is ever
                # planned and deepplan_plugin is None, so this never fires (today's flow). It is
                # placed above recovery + the chunk_queue pop so a checkpoint becomes real
                # subgoals before either inspects motions or executes a queued move.
                if self.deepplan_plugin is not None and self.deepplan_plugin.is_pivot(subgoal):
                    decision = self.deepplan_plugin.resolve(
                        task=self.task,
                        subgoal=subgoal,
                        subgoals=subgoals,
                        current_index=current_index,
                        agentview_image=agentview,
                        wrist_image=wrist,
                        debug=self.debug,
                    )
                    if decision.resolved_subgoals:
                        # Runner owns the splice: drop the pivot + placeholder suffix and run
                        # the resolved tail from here. current_index already points at the new
                        # (guaranteed concrete) head; reset per-stage bookkeeping as the
                        # rollback / subgoal-done blocks do, then re-enter the loop.
                        subgoals = subgoals[:current_index] + decision.resolved_subgoals
                        subgoal_start_step = step_idx + 1
                        current_direction = None
                        recent_moves.clear()
                        visual_history.clear()
                        chunk_queue = []
                        recovery_note = ""
                        self.logger.write_plan(self._deepplan_plan_record(decision, subgoals))
                        continue
                    self.logger.write_plan(self._deepplan_plan_record(decision, subgoals))
                    if decision.event == "resolve_retry":
                        # Transient empty/garbled reply (a thinking backend can return empty
                        # content): re-observe and re-resolve the SAME checkpoint next step.
                        # The tool bounds the retries. Reset the stage step counter so the
                        # per-stage cap cannot advance off the still-unresolved pivot.
                        subgoal_start_step = step_idx + 1
                        continue
                    # Retries exhausted: do NOT splice the unresolved placeholder suffix -- a
                    # blind grasp at an unrevealed target would spin the empty-grasp recovery
                    # loop. End the episode cleanly instead.
                    end_reason = "deepplan_resolve_failed"
                    break

                recovery_decision = self._recovery_before_decision(
                    current_index=current_index,
                    subgoals=subgoals,
                    obs=obs,
                )
                human_kind: Optional[str] = None
                open_loop = False
                if recovery_decision is not None and recovery_decision.token:
                    # Recovery override pre-empts (and cancels) any pending action chunk.
                    # It also outranks a pending DAGGER intent: it corrects a measured
                    # physical fact, not a preference (the intent stays for next step).
                    token = recovery_decision.token
                    response: Any = None
                    target_in_wrist = None
                    chunk_queue = []
                    if self.affordance_plugin is not None:
                        # The grasp measurably failed, so the scene was likely
                        # disturbed: re-ground the dot on the next observation.
                        self.affordance_plugin.clear()
                elif self._dagger_enabled() and self.dagger_plugin.has_intent():
                    # DAGGER: the operator already queued input -- the human owns this
                    # step, and any planned chunk is stale by definition.
                    token, human_kind = self._human_intent()
                    response = None
                    target_in_wrist = None
                    chunk_queue = []
                elif chunk_queue:
                    # Mid-chunk: execute the next planned move open-loop (no VLM call). The
                    # plan is only built while far, so the wrist judgment stays False here.
                    token = chunk_queue.pop(0)
                    response = None
                    target_in_wrist = False
                    open_loop = True
                else:
                    gripper_state = self._observed_gripper_state(obs)
                    decide_kwargs = dict(
                        ctx=ctx,
                        recent_moves=self._recent_moves_text(recent_moves),
                        previous_direction=current_direction or NO_DIRECTION,
                        gripper_state=gripper_state,
                        recovery_context=self._recovery_prompt(recovery_note),
                        # Blind-review frame: only when the last executed action is
                        # still the previous_direction being reported (any history
                        # reset in between makes the pair inconsistent -> None).
                        prev_agentview=(
                            review_frame
                            if (review_token is not None and review_token == current_direction)
                            else None
                        ),
                    )
                    if self.visual_history_steps:
                        decide_kwargs["visual_history"] = tuple(visual_history)
                    if self._dagger_enabled():
                        # Preemptible: the moment human keys arrive the call is
                        # abandoned (and its result later dropped); None means the
                        # human owns this step.
                        response = self._decide_interruptible(decide_kwargs)
                    else:
                        response = self.controls.controller.decide(**decide_kwargs)
                    chunk_queue = []
                    if response is None:
                        token, human_kind = self._human_intent()
                        target_in_wrist = None
                    else:
                        token = response.token
                        target_in_wrist = (
                            response.payload.get("target_in_wrist")
                            if isinstance(response.payload, dict)
                            else None
                        )
                        # action_chunk: when the TARGET is far, the model plans its next
                        # moves. Execute the first now and queue the rest (distinct
                        # moves, one VLM call).
                        if self.action_chunk_plugin is not None and target_in_wrist is False:
                            plan = (
                                response.payload.get("chunk_plan")
                                if isinstance(response.payload, dict)
                                else None
                            ) or []
                            if plan:
                                token = plan[0]
                                chunk_queue = list(plan[1:])

                    every = int(
                        getattr(self.config, "controller_prompt_log_every", 0) or 0
                    )
                    if every > 0 and step_idx % every == 0:
                        controller = getattr(self.controls, "controller", None)
                        prompt_text = getattr(controller, "last_prompt", "")
                        if prompt_text:
                            self.logger.save_controller_prompt(
                                step_idx,
                                prompt_text,
                                media=getattr(controller, "last_media", None),
                            )

                # Remember this step's wrist-visibility judgment so the NEXT step's
                # affordance premark knows whether to track the dot on the wrist frame.
                # A reply WITHOUT the marker (None) keeps the previous phase: one
                # marker-less sentence must not kill an active wrist track (rollout
                # Observed on hardware: a step omitted the marker, the correct wrist dot was
                # dropped, and the mis-grounded front dot steered the arm away).
                if target_in_wrist is not None:
                    self._prev_target_in_wrist = target_in_wrist is True

                # Execute the token on the real robot (Z floor enforced inside). target_in_wrist
                # (the shared wrist-visibility judgment) is forwarded so variable_step can pick a
                # coarse step when the TARGET is not yet in the wrist view.
                # Smoothness: the remaining moves of an action chunk run back-to-back with no
                # VLM call between them, so tell the controller another move follows -- it
                # ends this one at cruise speed and the arm flows through the chunk instead of
                # stopping once per token. The last move of the chunk closes the run normally.
                # STILL (a DAGGER hold) executes nothing: no re-command, the arm holds.
                if recorder is not None:
                    recorder.begin_step(
                        step_idx=step_idx,
                        stage=f"{current_index + 1}/{len(subgoals)} {subgoal.motion}: {subgoal.target}",
                        token=token, annotation=self._step_reason(response, human_kind, open_loop),
                        output=getattr(response, "raw_text", ""),
                        source=("human" if human_kind else "action_chunk" if open_loop
                                else "vlm" if response is not None else "recovery"),
                    )
                if token == STILL_TOKEN:
                    result = None
                else:
                    result = self.controller.step(
                        token,
                        target_in_wrist=target_in_wrist,
                        continuous=bool(chunk_queue),
                    )

                subgoal_done = bool(getattr(result, "done", False)) or token == DONE_TOKEN
                # How much of a commanded MV_DOWN the arm actually travelled, for the
                # NEXT prompt: the proprioception tool reports a descent that stalled
                # against something (see plugins.proprioception).
                self._descend = descend_travel(token, result, self._descend)

                post_recovery = self._recovery_after_step(
                    token=token,
                    result=result,
                    current_index=current_index,
                    subgoals=subgoals,
                    obs=obs,
                    subgoal_done=subgoal_done,
                )
                if post_recovery is not None:
                    recovery_decision = post_recovery
                recovery_release = None
                if recovery_decision is not None and recovery_decision.release:
                    recovery_release = self.controller.step(RELEASE_TOKEN)
                if recovery_decision is not None and recovery_decision.block_done:
                    subgoal_done = False
                # A close that measurably HOLDS expires any lingering recovery note
                # BEFORE a fresh this-step note (if any) is applied below. Ported from
                # the dual runner (_expire_grasp_note): the note describes a PAST
                # event, but it used to stay until the stage completed -- observed on
                # hardware (Franka): an "Empty close" note was
                # still in the prompt after step 29's verified 38 mm hold, priming the
                # model to call the real hold "empty closed" and dither MV_UP/MV_DOWN
                # instead of DONE.
                if (
                    token == GRASP_TOKEN
                    and result is not None
                    and bool(getattr(result, "gripper_closed", False))
                    and not bool(getattr(result, "grasp_empty", False))
                ):
                    recovery_note = ""
                if recovery_decision is not None and recovery_decision.prompt_note:
                    recovery_note = recovery_decision.prompt_note
                    # An empty grasp while pinned at the Z floor cannot be fixed by going
                    # lower; tell the VLM so it re-centers in XY instead of retrying depth.
                    if getattr(recovery_decision, "grasp_empty", False) and self._at_z_floor(obs):
                        recovery_note += " (at Z floor; descent exhausted)"

                if self.visual_history_steps:
                    effect = motion_history_effect(result)
                    action = str(getattr(result, "token", None) or token)
                    if action != token:
                        effect += f"; controller request {token} was mapped to executed {action}"
                    if recovery_decision is not None:
                        if recovery_decision.release:
                            action += " then recovery RELEASE"
                            effect += "; recovery RELEASE: " + motion_history_effect(recovery_release)
                        if recovery_decision.prompt_note:
                            effect += "; recovery feedback: " + str(recovery_decision.prompt_note)
                    visual_history.append(VisualHistoryEntry(
                        step_idx=step_idx, stage=subgoal.motion, views=history_views,
                        action=action, effect=effect, depth_text=history_depth_text,
                    ))
                    del visual_history[:-self.visual_history_steps]

                plan_complete = subgoal_done and (current_index + 1 >= len(subgoals))
                if recovery_decision is not None and recovery_decision.reset_history:
                    current_direction = None
                    recent_moves.clear()
                elif token in MOVE_ATOMS or getattr(result, "kind", "") == "move":
                    recent_moves.insert(0, token)
                    del recent_moves[self.recent_moves_max:]
                    current_direction = token
                elif token in GRIPPER_TOKENS:
                    # The next decision must see RELEASE too, especially when repeating
                    # an already-open/closed command had no effect on the scene.
                    noop = "(noop)" in str(getattr(result, "note", ""))
                    recent_moves.insert(0, f"{token}(no-op)" if noop else token)
                    del recent_moves[self.recent_moves_max:]
                    current_direction = None
                elif token in ROTATE_ATOMS or getattr(result, "kind", "") == "rotate":
                    recent_moves.insert(0, token)
                    del recent_moves[self.recent_moves_max:]
                    current_direction = None
                elif subgoal_done:
                    current_direction = None

                # Blind review bookkeeping: this iteration's pre-execution frame
                # brackets the token just executed. Only a cleanly executed MV_*
                # with no recovery intervention is reviewable; everything else
                # clears the pair (frames would show motion the symbol didn't cause).
                if (
                    recovery_decision is None
                    and token in MOVE_ATOMS
                    and result is not None
                ):
                    review_frame = agentview
                    review_token = token
                else:
                    review_frame = None
                    review_token = None

                # End a simulated task as soon as it succeeds, before another model
                # command can undo the placement or retreat.
                physical_success = callable(success_check) and bool(success_check())
                record = self._record(
                    step_idx=step_idx,
                    subgoal=subgoal,
                    subgoal_index=current_index,
                    subgoal_step=subgoal_step,
                    token=token,
                    result=result,
                    response=response,
                    obs=obs,
                    subgoal_done=subgoal_done,
                    success=physical_success if callable(success_check) else plan_complete,
                    recovery_decision=recovery_decision,
                    human_kind=human_kind,
                )
                diagnostics = self._success_diagnostics()
                if diagnostics is not None:
                    record["success_diagnostics"] = diagnostics
                if recorder is not None:
                    recorder.end_step(record)
                self._show_live(
                    step_idx,
                    agentview,
                    wrist,
                    "executed",
                    steps_done=step_idx + 1,
                    subgoal=subgoal,
                    subgoal_index=current_index,
                    n_subgoals=len(subgoals),
                    token=token,
                    record=record,
                    reason=self._step_reason(response, human_kind, open_loop),
                )
                self._print_step(
                    step_idx=step_idx,
                    subgoal=subgoal,
                    subgoal_index=current_index,
                    n_subgoals=len(subgoals),
                    token=token,
                    record=record,
                    response=response,
                    recovery_decision=recovery_decision,
                    human_kind=human_kind,
                    open_loop=open_loop,
                )
                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=wrist,
                    record=record,
                    **({"extra_views": obs["extra_views"]} if obs.get("extra_views") else {}),
                    **({"wrist_depth_m": obs["wrist_depth_m"], "wrist_depth_text": history_depth_text}
                       if "wrist_depth_m" in obs else {}),
                )
                self._write_action_table(step_idx)

                if plan_complete or physical_success:
                    success = True
                    end_reason = "task_success" if physical_success else "plan_complete"
                    break
                if (
                    recovery_decision is not None
                    and recovery_decision.rollback_index is not None
                ):
                    current_index = int(recovery_decision.rollback_index)
                    subgoal_start_step = step_idx + 1
                    current_direction = None
                    recent_moves.clear()
                    # History is reset for the fresh approach, but keep the just-failed empty
                    # GRASP as the single newest move so the controller sees it and avoids
                    # grasping in place again (instead of re-deciding GRASP from a blank slate).
                    if getattr(recovery_decision, "grasp_empty", False):
                        recent_moves.insert(0, EMPTY_GRASP_LABEL)
                    chunk_queue = []
                    continue
                if subgoal_done:
                    current_index += 1
                    subgoal_start_step = step_idx + 1
                    current_direction = None
                    recent_moves.clear()
                    chunk_queue = []
                    if self._is_grasp_stage(subgoal):
                        recovery_note = ""

                elapsed = time.monotonic() - loop_started
                if self.loop_period_s > elapsed:
                    time.sleep(self.loop_period_s - elapsed)
            # Simulator sessions provide a physical predicate. A model's DONE is
            # sufficient on hardware, but cannot certify success in a simulator.
            if callable(success_check):
                success = bool(success_check())
                if success:
                    end_reason = "task_success"
                elif end_reason == "plan_complete":
                    end_reason = "task_not_complete"
        except KeyboardInterrupt:
            end_reason = "interrupted"
            print(
                "\n[run-real] Ctrl+C received -- stopping rollout and compiling the "
                "visualization video..."
            )
        except Exception as exc:
            end_reason = "error"
            if recorder is not None:
                recorder.event("error", annotation=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            video_path = self.logger.close(
                success=success, fps=video_fps(self.config.video_fps)
            ) or ""
            if recorder is not None:
                recorder.finish(success, end_reason)
                video_path = recorder.paths["combined"]
            self.logger.write_summary(
                {
                    "success": success,
                    "steps": steps,
                    "max_steps": self.max_steps,
                    "end_reason": end_reason,
                    "video_path": str(video_path),
                    "run_dir": str(self.logger.run_dir),
                    "control_mode": getattr(self.session, "control_mode", "real"),
                    **({"recordings": {k: str(v) for k, v in recorder.paths.items()}}
                       if recorder is not None else {}),
                    "task": self.task,
                    "gripper_color": self.gripper_color,
                    "z_floor_m": self.controller.z_floor_m,
                    **({"success_diagnostics": diagnostics}
                       if (diagnostics := self._success_diagnostics()) is not None else {}),
                    "raw_plan": raw_plan,
                    "subgoals": [sg.to_prompt_dict() for sg in subgoals],
                    # Every affordance grounding of the episode (metadata.json, written
                    # before the run, always holds an empty copy).
                    **(
                        {"affordance": self.affordance_plugin.metadata()}
                        if getattr(self.affordance_plugin, "enabled", False)
                        else {}
                    ),
                }
            )

        return EpisodeResult(
            success=success,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(self.logger.run_dir),
        )

    # -- helpers -----------------------------------------------------------
    def _success_diagnostics(self):
        """Persist physical criteria for audits; never feed object state to the VLM."""
        diagnostics = getattr(self.session, "success_diagnostics", None)
        return diagnostics() if callable(diagnostics) else None

    def _single_task_subgoal(self) -> Subgoal:
        """The fallback "plan" when the subgoal tool is disabled: one stage covering the
        whole task, so the controller drives toward task completion and DONE ends it."""
        return Subgoal(
            id="task",
            target=self.task,
            affordance="the task-relevant object or region",
            motion="TASK",
            description=self.task,
            completion=f"the task is visibly complete: {self.task}",
        )

    def _images(self, obs: dict[str, Any]):
        agentview = to_uint8_hwc(obs["agentview"])
        wrist = None
        if self.use_wrist_image and obs.get("wrist") is not None:
            wrist = to_uint8_hwc(obs["wrist"])
        return agentview, wrist

    def _premark_affordance(
        self, agentview, wrist, subgoal: Subgoal, index: int, obs: Optional[dict] = None
    ):
        """Affordance dots: ground the front dot once on stage entry (the 'which part'
        anchor), then -- while the arm is wrist-guided (the PREVIOUS step's target-in-
        wrist marker) -- re-locate that part on this step's wrist frame and draw it
        there too. Every consumer (controller, live view, video) sees the annotated
        frames. Wrist tracking needs the wrist-visibility marker, which the controller
        only emits when a wrist-consuming tool (variable_step/action_chunk/rotation) is
        on; otherwise the tool stays front-only. Tool off/absent -> frames pass through.

        ``obs`` may carry ``agentview_hd``/``wrist_hd`` (the session's high-res renders
        of the SAME frames, same pad geometry): the pointer/verify/track VLM calls then
        see glyph-level detail while every stored/drawn point stays on the shared
        0-1000 grid. Absent (Piper, hd off) -> the small observation is used as before."""
        tool = self.affordance_plugin
        if tool is None or not getattr(tool, "enabled", False):
            return agentview, wrist
        hd = obs or {}
        tool.ensure(
            slot="arm",
            stage_key=f"{index}:{subgoal.id}:{subgoal.target}:{subgoal.affordance}",
            task=self.task,
            subgoal=subgoal.to_prompt_dict(),
            agentview=agentview,
            agentview_hd=hd.get("agentview_hd"),
            debug=self.debug,
        )
        tool.update_wrist(
            slot="arm",
            wrist_image=wrist,
            active=bool(self._prev_target_in_wrist),
            task=self.task,
            # Cross-view alignment: the current front frame rides along so every
            # wrist locate/track is anchored to the committed front dot.
            agentview=agentview,
            wrist_hd=hd.get("wrist_hd"),
            agentview_hd=hd.get("agentview_hd"),
            debug=self.debug,
        )
        return tool.annotate(agentview), tool.annotate_wrist("arm", wrist)

    def _proprio(self, obs: dict[str, Any]) -> dict[str, Any]:
        ee_pose = np.asarray(obs.get("ee_pose", []), dtype=float).reshape(-1)
        eef_pos = ee_pose[:3].tolist() if ee_pose.size >= 3 else []
        proprio: dict[str, Any] = {
            "eef_pos": eef_pos,
            "gripper_width": float(obs.get("gripper_width", 0.0)),
            "gripper_command_name": "CLOSED" if self.controller.gripper_closed else "OPEN",
        }
        if ee_pose.size >= 7:
            proprio["eef_quat"] = ee_pose[3:7].tolist()
        if "insertion_camera_axes" in obs:
            proprio["insertion_camera_axes"] = obs["insertion_camera_axes"]
        if "wrist_depth_calibration" in obs:
            proprio["wrist_depth_calibration"] = obs["wrist_depth_calibration"]
        if self._descend is not None:
            proprio["descend_moved_m"], proprio["descend_commanded_m"] = self._descend
        return proprio

    def _at_z_floor(self, obs: dict[str, Any], eps_m: float = 0.005) -> bool:
        """True when the EEF is resting at the locked Z floor (descent is exhausted)."""
        z_floor = getattr(self.controller, "z_floor_m", None)
        if z_floor is None:
            return False
        ee_pose = np.asarray(obs.get("ee_pose", []), dtype=float).reshape(-1)
        if ee_pose.size < 3:
            return False
        return float(ee_pose[2]) - float(z_floor) <= eps_m

    @staticmethod
    def _recent_moves_text(recent_moves: list[str]) -> str:
        return ", ".join(recent_moves) if recent_moves else "none"

    def _write_action_table(self, step_idx: int) -> None:
        """Persist the action-ablation blind mode's self-written symbol->effect
        table into the rollout (action_table.json, overwritten each step so the
        file always holds the latest table + the full note history)."""
        tool = self.action_ablation_plugin
        record = tool.table_record() if tool is not None else None
        if record is None:
            return
        record["updated_step"] = int(step_idx)
        (self.logger.run_dir / "action_table.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _recovery_prompt(self, note: str) -> str:
        if self.recovery_plugin is None:
            return ""
        return self.recovery_plugin.render_prompt_context(note)

    def _deepplan_plan_record(self, decision: Any, subgoals: list[Subgoal]) -> dict[str, Any]:
        """The subgoals.json record written when a <REASON> checkpoint resolves (or fails):
        the live plan plus the resolver's branch + reasoning, for offline analysis."""
        return {
            "task": self.task,
            "deepplan_event": decision.event,
            "deepplan_branch": decision.branch,
            "deepplan_reasoning": decision.reason,
            "deepplan_resolved": decision.raw_text,
            "subgoals": [sg.to_prompt_dict() for sg in subgoals],
        }

    def _recovery_before_decision(
        self,
        *,
        current_index: int,
        subgoals: list[Subgoal],
        obs: dict[str, Any],
    ):
        if self.recovery_plugin is None:
            return None
        return self.recovery_plugin.before_decision(
            current_index=current_index,
            subgoals=subgoals,
            measured_width_m=self._measured_width(obs),
            gripper_closed=bool(self.controller.gripper_closed),
        )

    def _recovery_after_step(
        self,
        *,
        token: str,
        result: AtomicStepResult,
        current_index: int,
        subgoals: list[Subgoal],
        obs: dict[str, Any],
        subgoal_done: bool,
    ):
        if self.recovery_plugin is None or result is None:
            return None
        return self.recovery_plugin.after_step(
            token=token,
            result=result,
            current_index=current_index,
            subgoals=subgoals,
            measured_width_m=self._measured_width(obs),
            subgoal_done=subgoal_done,
            gripper_closed=bool(result.gripper_closed),
        )

    @staticmethod
    def _is_grasp_stage(subgoal: Subgoal) -> bool:
        """True when the subgoal closes the gripper on the object (the only
        stage whose completion requires a verified physical hold)."""
        return str(getattr(subgoal, "motion", "")).upper() == "GRASP"

    @staticmethod
    def _measured_width(obs: dict[str, Any]) -> float:
        return float(obs.get("gripper_width", 0.0))

    def _observed_gripper_state(self, obs: dict[str, Any]) -> str:
        """Gripper OPEN/CLOSED as the CAMERA sees it (from the measured width), NOT the
        commanded state. On real hardware a commanded close leads the physical fingers --
        and the width/image -- by ~1-2 steps; reporting the commanded state would make the
        prompt say CLOSED while the image still shows the gripper open, which is exactly
        what derailed the controller into a premature DONE."""
        thr = float(getattr(self.controller, "gripper_close_threshold_m", 0.07))
        return "CLOSED" if self._measured_width(obs) < thr else "OPEN"

    def _record(
        self,
        *,
        step_idx: int,
        subgoal: Subgoal,
        subgoal_index: int,
        subgoal_step: int,
        token: str,
        result: Optional[AtomicStepResult],
        response: Any,
        obs: dict[str, Any],
        subgoal_done: bool,
        success: bool,
        recovery_decision: Any = None,
        human_kind: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a compact step record compatible with EpisodeLogger's analysis frame.

        ``result`` is None for a held step (a DAGGER STILL): pose/grip fall back to
        the observation and the controller's commanded state."""
        if result is not None and getattr(result, "post_pose", None) is not None:
            eef = np.asarray(result.post_pose, dtype=float).reshape(-1)[:3]
        else:
            eef = np.asarray(obs.get("ee_pose", []), dtype=float).reshape(-1)[:3]
        reasoning, latency_ms = _vlm_reason_latency(response)
        if latency_ms is not None:
            self._last_vlm_ms = latency_ms
        gripper_closed = (
            bool(result.gripper_closed)
            if result is not None
            else bool(self.controller.gripper_closed)
        )
        record: dict[str, Any] = {
            "i": int(step_idx),
            "sg": int(subgoal_index),
            "sg_step": int(subgoal_step),
            "stage": subgoal.motion,
            "sid": subgoal.id,
            "act": token,
            "eef": [round(float(x), 3) for x in eef],
            "w": round(float(obs.get("gripper_width", 0.0)), 5),
            "grip": "CLOSED" if gripper_closed else "OPEN",
        }
        if human_kind is not None:
            # DAGGER: this step's token came from the operator, not the model.
            record["src"] = "human"
        # Which step magnitude the MOVE used (the variable-step plugin's coarse/fine
        # choice, or the dedicated MV_UP distance); absent when the step is fixed.
        step_kind = str(getattr(result, "step_kind", "") or "")
        if step_kind:
            record["step_kind"] = step_kind
            if getattr(result, "kind", "") == "rotate":
                record["rotation_deg"] = round(float(getattr(result, "rotation_deg", 0.0)), 3)
                record["rotation_vector_rad"] = np.asarray(
                    getattr(result, "intended_rotation_rad", [0, 0, 0]), dtype=float,
                ).tolist()
            else:
                record["step_cm"] = round(float(getattr(result, "step_m", 0.0)) * 100.0, 3)
        if getattr(self.controller, "cartesian_actions", None) is not None and result is not None:
            # Preserve all six degrees of freedom in insertion rollouts.
            for field in ("pre_pose", "target_pose", "post_pose"):
                pose = getattr(result, field, None)
                if pose is not None:
                    record[field] = np.asarray(pose, dtype=float).tolist()
        response_payload = getattr(response, "payload", None) or {}
        if "target_in_wrist" in response_payload:
            record["target_in_wrist"] = response_payload["target_in_wrist"]
        if reasoning or latency_ms is not None:
            c_entry: dict[str, Any] = {}
            if latency_ms is not None:
                c_entry["ms"] = latency_ms
            if reasoning:
                c_entry["why"] = reasoning
            record["vlm"] = {"c": c_entry}
            if latency_ms is not None:
                record["vlm_ms"] = latency_ms
        if self.affordance_plugin is not None:
            dot = self.affordance_plugin.point_of("arm")
            if dot:
                record["afford_dot"] = dot
            wrist_dot = self.affordance_plugin.wrist_point_of("arm")
            if wrist_dot:
                record["afford_wrist"] = wrist_dot
        note = str(getattr(result, "note", "") or "")
        if "(noop)" in note:
            record["noop"] = True
        if note and "z-floor" in note:
            record["blocked"] = "z_floor"
        elif note and ("reach clamp" in note or "reach fallback" in note):
            record["blocked"] = "reach"
        if getattr(result, "kind", "") == "realign":
            # MV_UP un-rotated a held object back to neutral instead of lifting.
            record["realign"] = "to_neutral"
        grasp_empty = bool(
            getattr(result, "grasp_empty", False)
            or getattr(recovery_decision, "grasp_empty", False)
        )
        if grasp_empty:
            record["grasp_fail"] = getattr(result, "note", None) or True
        if bool(getattr(recovery_decision, "grasp_unverified", False)):
            record["grasp_unverified"] = True
        if recovery_decision is not None:
            record["recover"] = True
            recovery_record: dict[str, Any] = {}
            for key in ("event", "reason", "rollback_index", "token", "release"):
                value = getattr(recovery_decision, key, None)
                if value not in (None, "", False):
                    recovery_record[key] = value
            if recovery_record:
                record["recovery"] = recovery_record
        if subgoal_done:
            record["done"] = True
        if success:
            record["ok"] = True
        return record

    # -- DAGGER: preemptible VLM decisions ----------------------------------------
    def _dagger_enabled(self) -> bool:
        return bool(getattr(self.dagger_plugin, "enabled", False))

    def _human_intent(self) -> tuple[str, str]:
        """Consume the pending DAGGER intent: ``(token, kind)``.

        The gripper placeholder resolves into GRASP/RELEASE from the controller's
        ACTUAL state at execution time (teleop's flush-time contract). No pending
        intent (a race with drain) degrades to a STILL hold."""
        intents = self.dagger_plugin.drain()
        token, kind = intents.get("arm", (STILL_TOKEN, "still"))
        if kind == "gripper":
            token = RELEASE_TOKEN if self.controller.gripper_closed else GRASP_TOKEN
        return token, kind

    def _decide_interruptible(self, decide_kwargs: dict[str, Any]) -> Optional[Any]:
        """One controller ``decide``, preempted by human DAGGER keys.

        Delegates to :class:`core.runners.preemption.InterruptibleDecider`; ``None`` means
        the human owns this step (the runner executes the human intent instead)."""
        if self._decider is None:
            self._decider = InterruptibleDecider(
                self.dagger_plugin, lambda **kw: self.controls.controller.decide(**kw)
            )
        return self._decider.decide(decide_kwargs)

    # -- console + live-view status ------------------------------------------------
    def _print_plan(self, subgoals: list[Subgoal]) -> None:
        """The planner's stage list, one line per stage (mirrors the dual header)."""
        n = len(subgoals)
        print(f"\n{console.dim('  plan')}  {console.dim(f'{n} stage' + ('s' if n != 1 else ''))}")
        for i, sg in enumerate(subgoals):
            print(
                f"        STAGE {i + 1}  {sg.motion:<8} "
                f"{console.dim(console.short(sg.completion, 68))}"
            )
        print()

    def _step_reason(
        self, response: Any, human_kind: Optional[str], open_loop: bool
    ) -> str:
        if response is not None:
            reasoning, _ = _vlm_reason_latency(response)
            return reasoning
        if human_kind is not None:
            return "human override (DAGGER)"
        if open_loop:
            return "action chunk -- executing the planned move open-loop"
        return "recovery override -- no VLM call this step"

    def _print_step(
        self,
        *,
        step_idx: int,
        subgoal: Subgoal,
        subgoal_index: int,
        n_subgoals: int,
        token: str,
        record: dict[str, Any],
        response: Any,
        recovery_decision: Any,
        human_kind: Optional[str],
        open_loop: bool,
    ) -> None:
        """One readable block per step: a rule, the stage/action line, then the
        reasoning. Same story the dual runner tells, for one arm."""
        print(console.dim(f"\n─── step {step_idx:03d} " + "─" * 52))
        stage = f"STAGE {subgoal_index + 1}/{n_subgoals} {subgoal.motion}"
        act = token if token == STILL_TOKEN else console.c(console.BOLD, token)
        flags = []
        # Step precision first: it qualifies the action itself ("MV_FWD coarse 5 cm").
        if record.get("step_kind"):
            amount, unit = ((record["rotation_deg"], "deg") if "rotation_deg" in record
                            else (record["step_cm"], "cm"))
            flags.append(console.dim(f"{record['step_kind']} {amount:g} {unit}"))
        if record.get("done"):
            flags.append(console.c(console.GREEN, "stage done"))
        if record.get("grasp_fail"):
            flags.append(console.c(console.RED, "empty close -> reopened"))
        blocked = record.get("blocked")
        if blocked == "z_floor":
            flags.append(console.c(console.YELLOW, "blocked: z-floor"))
        elif blocked == "reach":
            flags.append(console.c(console.YELLOW, "blocked: reach limit"))
        event = getattr(recovery_decision, "event", None)
        if event:
            flags.append(console.c(console.YELLOW, f"recovery: {event}"))
        if human_kind is not None:
            flags.append(console.c(console.YELLOW, "human"))
        if open_loop:
            flags.append(console.dim("chunk open-loop"))
        if record.get("realign"):
            flags.append(console.dim("realign to neutral"))
        suffix = ("   " + " · ".join(flags)) if flags else ""
        tag = console.c(console.ARM_COLOR, "  ARM")
        grip = console.dim("grip " + record.get("grip", "-"))
        print(f"{tag}  {stage:<20} {act:<20} {grip}{suffix}")
        if response is None:
            print(console.dim(f"  · {self._step_reason(None, human_kind, open_loop)}"))
            return
        reasoning, _ = _vlm_reason_latency(response)
        for line in console.wrap_reason(reasoning):
            print(f"  {console.dim('·')} {line}")

    def _telemetry_text(self, steps_done: int) -> str:
        elapsed = time.monotonic() - self._t0
        parts = [f"total {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"]
        if steps_done > 0:
            parts.append(f"avg {elapsed / steps_done:.1f} s/step")
        if self._last_vlm_ms is not None:
            parts.append(f"VLM {self._last_vlm_ms} ms")
        return "  ·  ".join(parts)

    def _show_live(
        self,
        step_idx: int,
        agentview: Optional[np.ndarray],
        wrist: Optional[np.ndarray],
        phase: str,
        steps_done: int,
        subgoal: Optional[Subgoal] = None,
        subgoal_index: int = 0,
        n_subgoals: int = 0,
        token: Optional[str] = None,
        record: Optional[dict[str, Any]] = None,
        reason: str = "",
    ) -> None:
        if self.viewer is None:
            return
        if subgoal is not None:
            stage = f"STAGE {subgoal_index + 1}/{n_subgoals} · {subgoal.motion}"
        else:
            stage = "FINISHED"
        arm: dict[str, Any] = {
            "stage": stage,
            "grip": "CLOSED" if self.controller.gripper_closed else "OPEN",
        }
        if token is not None:
            arm["token"] = token
        if record is not None:
            arm["flags"] = status_flags(record)
        self.viewer.show_single(
            step=step_idx,
            agentview=agentview,
            wrist=wrist,
            arm=arm,
            task=self.task,
            phase=phase,
            telemetry=self._telemetry_text(steps_done),
            reason=reason,
        )


def descend_travel(
    token: str, result: Any, previous: Optional[tuple[float, float]]
) -> Optional[tuple[float, float]]:
    """``(travelled, commanded)`` height for a just-executed MV_DOWN, else carry/clear.

    Measured, not inferred: the commanded height is the controller's own clamped delta,
    the travelled height the measured pose before vs after. Any OTHER arm motion clears
    it (the reading must describe the step just executed); a gripper action or a hold
    leaves it in place, since the arm did not move and an earlier stall still holds.
    """
    if result is None or getattr(result, "kind", "") not in ("move", "rotate"):
        return previous
    pre, post = getattr(result, "pre_pose", None), getattr(result, "post_pose", None)
    commanded = float(np.asarray(getattr(result, "intended_delta_m", [0, 0, 0]))[2])
    if pre is None or post is None or commanded >= 0:
        return None
    travelled = float(np.asarray(pre, dtype=float)[2] - np.asarray(post, dtype=float)[2])
    return max(0.0, travelled), abs(commanded)


def _vlm_reason_latency(response: Any) -> tuple[str, Optional[int]]:
    payload = getattr(response, "payload", None) or {}
    json_obj = payload.get("json")
    reasoning = ""
    if isinstance(json_obj, dict):
        reasoning = str(json_obj.get("reasoning") or "").strip()
    latency_s = payload.get("latency_s")
    latency_ms = int(round(float(latency_s) * 1000.0)) if latency_s is not None else None
    return reasoning, latency_ms


def video_fps(video_fps: float) -> float:
    return min(30.0, max(0.5, float(video_fps)))
