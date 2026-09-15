# Simulators

Show-Harness integrates MuJoCo, ManiSkill, and RoboLab (Isaac Lab). They serve two roles:

1. **Zero-shot evaluation** — run the deployment pipelines (the subgoal planner
   stack or the fine-tuned action model, see `docs/finetuned.md`) in sim
   with no robot attached.
2. **Sim-to-real training data** — generate demonstrations on the same action
   lattice the real robot deploys with (single-axis, ~2 cm per token), so sim
   rollouts mix directly with real teleop rollouts for fine-tuning.

Each integration keeps the deployment contracts: the same nine-token action
vocabulary, the same image transforms (`core/record/images.py`), and measured
per-config step calibration so one token means ~2 cm of physical travel.

## MuJoCo + cloud models

Run the zero-shot planner/controller with a Franka Panda in a local MuJoCo
pick-and-place scene. The default model is `gemini-3.5-flash-lite`, using the existing
Gemini API provider. Ollama Cloud remains available with `--vlm-backend ollama`.
No local model server, model weights, GPU, or robot
hardware is required. The initial scene contains a red cube and a blue target pad.
The shared hardware imports may print a `pyrealsense2` warning; MuJoCo does not use it.

```bash
# Creates/uses .venv with uv; downloads only the official Panda MJCF and meshes.
bash scripts/setup.sh mujoco

# Use an existing exported key, or add this line to gitignored configs/secrets.env:
# GEMINI_API_KEY="your-key"

# Check all six 2 cm moves, empty-grasp recovery, three cameras, and recording (no API calls).
uv run --no-project python scripts/run_mujoco.py --smoke-test --record

# Run the complete model-controlled task; exits 0 only on physical success.
uv run --no-project python scripts/run_mujoco.py
# Add --record to save videos and their annotated timeline.
uv run --no-project python scripts/run_mujoco.py --record
# Use the existing Ollama profile instead (requires OLLAMA_API_KEY):
uv run --no-project python scripts/run_mujoco.py --vlm-backend ollama --model glm-5.3-flash:cloud

# Interactive MuJoCo viewport on macOS (mjpython is required for the viewer).
.venv/bin/mjpython scripts/run_mujoco.py --gui
# Linux/Windows:
# uv run --no-project python scripts/run_mujoco.py --gui
```

If macOS `mjpython` fails at `otool` with exit status 69, run
`otool -l .venv/bin/python` to see the underlying error. When the selected Xcode
installation reports an unaccepted license and `/Library/Developer/CommandLineTools`
is installed, select those tools for this run:

```bash
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  uv run --no-project mjpython scripts/run_mujoco.py --gui --variable-step --record
```

This selects the developer tools only for that process; the system-wide Xcode
selection is unchanged.

Linux without a display may need `MUJOCO_GL=egl` (EGL drivers) or
`MUJOCO_GL=osmesa` (Mesa) before launching. Offscreen rendering on macOS uses
the native graphics session; `MUJOCO_GL=egl` is not a macOS setting.

Configuration: [`configs/robot_mujoco.yaml`](../configs/robot_mujoco.yaml).
`--vlm-backend` selects a profile; `--model`, `--vlm-url`, `--max-steps`, and
`--log-dir` override it for one run. Set `vlm_backends.gemini.model` or
`GEMINI_MODEL` to change the Gemini model; the Ollama profile uses
`vlm_backends.ollama.model`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL`.
The selected backend determines the endpoint and API key, so `--model` alone
does not switch providers. The model must accept images.

Gemini uses Google's [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai)
with JSON output, `GEMINI_API_KEY`, and low reasoning effort. Gemini 3 calls use
the backend's configured temperature (1.0), including planner/controller calls
that otherwise request zero, following
[Google's temperature guidance](https://ai.google.dev/gemini-api/docs/gemini-3#temperature).

Ollama uses native `/api/chat` with bearer authentication, ordered
base64 camera images, and non-streaming replies. Ollama Cloud currently does not
support structured output constraints, so the harness supplies the JSON contract
in the prompt and validates action tokens locally. `reasoning_effort: low` keeps
GLM's thinking enabled; `max_tokens` includes its reasoning budget.
See [Ollama Cloud](https://docs.ollama.com/cloud),
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs), and
[GLM-5.3-Flash](https://ollama.com/library/glm-5.3-flash).

This reuses the existing zero-shot runner, subgoal planner, controller prompts,
Franka action vocabulary, recovery, and episode logger. The simulator session
converts Cartesian setpoints to joint targets using damped Jacobian IK; MuJoCo
actuators, contacts, and friction perform the motion and grasp. Arm and gripper
targets follow the existing minimum-jerk profile over `motion_s` (default 0.35 s),
then hold for `settle_s` (0.15 s). The live viewer refreshes throughout the action
at approximately 60 fps and is paced to simulation time. Headless runs can
compute faster; recorded video uses simulation time for smooth playback.
Physics pauses during cloud calls, making motion independent of network latency.
`fine_step_m`, `motion_s`, `settle_s`, `ik_damping`,
`reset_qpos`, and `z_floor_m` remain configurable.
`observation_context` and `prompts/controller_mujoco.txt` describe the three cameras
and their action directions to the model; update them when changing camera geometry.
The planner and controller both receive separate images in this order: Side (A),
Wrist (B), Front (C). The side/front views are perspective views centered on
the manipulation workspace, looking down approximately 34 degrees from different
sides, with a 34-degree vertical field of view. Table-plane travel and vertical
height both affect vertical image position. The controller prompt describes the
calibrated diagonal motion directions for each view. Wrist remains the primary
guide for fine horizontal alignment. "Above the target" means alignment in the table
plane with positive vertical clearance, rather than alignment in a single image.
Side/front `pos`, `xyaxes`, and `fovy` are set in `assets/mujoco/pick_place.xml`;
a smaller `fovy` zooms in. The old AgentView and separate global cameras have been removed.
Proprioception uses the scene's actual tabletop height, independently of the
2.5 cm TCP safety floor. Descent hints require rough horizontal alignment first.
The scene requires MuJoCo 3.5+ for the
[`projection` camera attribute](https://mujoco.readthedocs.io/en/3.8.0/changelog.html#version-3-5-0-february-12-2026).

Rollouts are written under `rollouts/mujoco/<backend>/` (`gemini` or `ollama`), including camera observations,
prompts, actions, metadata with credentials redacted, a summary, and videos.
Success requires the cube to rest on the pad, the fingers to be open, and
the gripper to have retreated; a model's `DONE` alone is insufficient. The physical
predicate is checked after every action, and a successful episode stops immediately
to prevent later model commands from undoing it. The
built-in physical predicate is for this pick-and-place scene. If changing the
scene/task, update `MujocoSession.check_success` along with it. This scene uses
stock Panda fingers and has not been calibrated for the released fine-tuned policies.

### Optional variable step size

Movement steps remain fixed at `fine_step_m` (2 cm) by default. Add
`--variable-step` to connect the existing adaptive-step plugin:

```bash
.venv/bin/mjpython scripts/run_mujoco.py --gui --variable-step
# Optional recording and an offline calibration check:
uv run --no-project python scripts/run_mujoco.py --smoke-test --variable-step --record
```

With the flag, the available sizes are `fine_step_m: 0.02`, `coarse_step_m: 0.05`,
and `large_step_m: 0.10` (metres). Travel uses 5 cm for `MV_UP`, for `MV_DOWN`
above `high_above_table_m` (10 cm), or for an explicitly distant target. That
travel increases to 10 cm only above `large_above_table_m` (20 cm) of clearance
from the table-contact reference. Near-table descent uses 2 cm when the target
is visible. Horizontal alignment always uses `fine_step_m`
when the target is visible or visibility is unknown, regardless of height.
This avoids large left/right corrections repeatedly overshooting a visible cube.
The VLM keeps the same movement tokens and returns a `target_in_wrist` JSON
boolean alongside its decision. Legacy `WRIST: YES/NO` markers are still accepted;
missing or invalid visibility does not trigger coarse horizontal motion.
No extra model call is needed. The same plugin instance controls the prompt and
execution. The flag is required even if a config declares `plugins.variable_step`.

Alignment is approximate at each discrete step: once roughly centered, the model
can descend and refine from a closer image. Visible targets use the top-down wrist
view for grasp, transport, and placement alignment, avoiding height/depth confusion
in the external view. Recent-action memory includes both
gripper commands and marks unchanged commands as `(no-op)`, and the prompt shows
the measured gripper width. This supplies feedback for repeated commands without
automatically choosing a descent or declaring a stage complete.
Height-based descent hints are suppressed during release and retreat.

The smooth motion and safety floor still apply. In variable-step mode, longer
translations receive proportionally longer ramps to maintain comparable speed;
the default fixed-step timing is unchanged. `steps.jsonl` records the chosen
`step_kind`, `step_cm`, and `target_in_wrist`; with `--record`, completed-step
annotations include these fields too, with `step_kind` set to `fine`, `coarse`,
or `large`. The smoke test supplies a scripted far-target signal to check all
six large directions without a VLM call. Older configs without `large_step_m`
retain the two-size behavior.

### Recording and annotated logs

Recording is disabled by default. Add `--record` (also with `--gui` or
`--smoke-test`) to enable it. Each recorded run's `videos/` directory contains:

| File | Content |
| --- | --- |
| `side.mp4` | Close angled side view of the manipulation workspace |
| `wrist.mp4` | Gripper-mounted camera |
| `front.mp4` | Close angled front view of the manipulation workspace |
| `combined.mp4` | Side / Wrist above, Front / VLM output and decision below |
| `annotations.jsonl` | Timestamped decisions, full VLM output, execution, recovery, and results |

The camera videos are 512 × 512 and the four-panel canvas is 1024 × 1088 at
the default resolution. All four videos have identical frame counts and run at
30 fps. Frames are captured during physics steps, so the recordings show the
motion between actions. Cloud waiting time is omitted; each action's result is
held for one second for readability. These video pauses do not advance physics.
Change `recording.fps` and `recording.decision_hold_s` in the config to tune
recording. MuJoCo uses this four-panel recorder instead of the generic per-decision
analysis video. Without `--record`, no videos are created; ordinary step logs and observation images are still
saved. `summary.json` and the printed `video_path` point to the combined video
when recording is enabled and use an empty string otherwise.

Each annotation includes UTC, simulation time, video time, frame index, stage,
decision source, and explanatory text. `decision` events precede the motion;
`step_complete` events include the action result and `[frame_start, frame_end)`
interval in every video (indices start at zero). `vlm_output` retains the full
answer even when the canvas shortens long text. Gripper commands, including an
automatic reopen after an empty grasp, are annotated separately. Interruptions
and errors finalize the videos and preserve the log; frames stream to disk as
playable fragmented MP4s while the run is in progress.

The same Side, Wrist, and Front cameras feed the model and recorder, independently
of the interactive viewer. Observations are saved in `images/side`, `images/wrist`,
and `images/front`. The shared runner's internal `agentview` input now carries Side.
`--smoke-test --record` also writes example recordings to
`rollouts/mujoco/smoke_test/videos/`, marked `smoke_test` rather than VLM decisions.

The setup downloads the Apache-2.0-licensed
[MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda)
and its license at a pinned revision into ignored `third_party/mujoco_menagerie/`.
The offline physical grasp-and-place regression check is:

```bash
uv run --no-project python -m unittest tests.test_mujoco tests.test_ollama tests.test_mujoco_recording
```

## ManiSkill

Fine-tuned policy only, in ManiSkill 3's translation-only `pd_ee_delta_pos`
control mode (rotation locked — the assumption the atomic-token policy makes).

- `scripts/run_maniskill_mvtoken.py` — entry point; config `configs/robot_maniskill.yaml`.
  One config covers both protocols: it defaults to the fixed layout preset, and
  `--traj-id random --layout wide` reproduces the randomized object layout the
  training data was generated with. `--env-id` switches scenes.
- `scripts/maniskill/eval_batch.sh <config> <model> <n_episodes> [max_steps] [tag]`
  — batch eval over consecutive seeds; prints the closed-loop success rate.

ManiSkill needs its own Python environment; the
runner's remaining dependencies (numpy, requests, pyyaml, PIL, imageio) are
standard. Scenes are declared as one `SceneSpec` row each in
`core/sim/maniskill_scenes.py`. The default `BlockPAP-v1` is a real2sim replica
of the real Franka rig (table, pedestal, block + coaster, calibrated front
camera) and needs an RLinf checkout (`RLINF_ROOT`); `BlockStack-v1` is the same
rig with a stacking task. Stock tasks (`PickCube-v1`, `StackCube-v1`) run too
but with a much larger domain gap.

```bash
python scripts/run_maniskill_mvtoken.py --version v3 --model <adapter> --max-steps 60 --probe-axes
```

`--probe-axes` records the measured per-token TCP delta into `calibration.json`.

Calibration facts (load-bearing; full derivations in the yaml comments):

- `step_m: 0.026` x `sim_steps_per_decision: 2` is the commanded setting that
  achieves ~20.2 mm per decision (PD lag makes achieved < commanded) — see
  `configs/robot_maniskill.yaml`.
- `wrist_flip: both` and `agentview_square_size: 256` are training contracts;
  read the yaml comments before touching either, and regenerate data after any
  camera change.

Training-data generation lives in `scripts/trajectory/real2sim/`: a
simulator-agnostic core (`atomic_tokenizer.py` — token vocabulary, closed-loop
2 cm execution, Manhattan/RDP/chase planners, teleop-format writer) plus one
backend per simulator (`backends/maniskill.py`, `backends/robolab.py`). It
produces rollouts in exactly the real-teleop format, so sim and real data mix
without special cases. See `scripts/trajectory/real2sim/README.md`.

## RoboLab (Isaac Lab)

NVIDIA's [RoboLab](https://github.com/NVLabs/RoboLab) benchmark: 120 authored
Isaac Sim manipulation tasks with automated success predicates and photoreal
rendering.

- `scripts/run_robolab_mvtoken.py` — entry point; config `configs/robot_robolab.yaml`.
- `scripts/robolab/eval_batch.sh <model> <n_episodes> [task ...]` — batch eval,
  one process per task (Isaac Sim's cold start dominates otherwise).

The RoboLab checkout location comes from the `ROBOLAB_ROOT` environment
variable (see `robolab_root()` in `core/sim/robolab_task.py`). Isaac Sim pins
Python 3.11 with its own large dependency set, so run this repo's scripts with
the RoboLab venv's interpreter — it also satisfies everything the runner needs.
First launch requires accepting Isaac Sim's EULA (`OMNI_KIT_ACCEPT_EULA=YES`),
and `libGLU.so.1` must be loadable or Isaac Sim segfaults during stage creation
with a misleading backtrace; `launch_isaac` in `core/sim/robolab_task.py`
checks for it up front and prints the fix.

```bash
python scripts/run_robolab_mvtoken.py --list-tasks                                     # no Isaac Sim needed
python scripts/run_robolab_mvtoken.py --task RubiksCubeTask --dump-views --probe-axes --no-rollout   # calibration only, no VLM
python scripts/run_robolab_mvtoken.py --version v3 --task RubiksCubeTask --episodes 5
```

`--task` takes the task class name; `--episodes N` reuses one Isaac Sim app and
env across episodes; `--gui` shows the viewport (default headless).

Embodiment. The sim Franka wears the real rig's short yellow fingertips, not
the stock black fingers — for a policy that reads pixels, the stock finger is a
distribution shift in the middle of every frame.
`assets/robolab_franka/panda_short_finger.usda` replaces both finger visuals
and collision meshes (rebuild with
`scripts/trajectory/real2sim/robolab/make_short_finger_asset.py`; set
`ROBOLAB_PANDA_USD` to compare against the stock robot). The camera geometry
also mirrors the real rig rather than RoboLab's DROID default: Panda hand, and
the wrist camera centered between the fingers looking down the grasp axis
(`core/sim/robolab_franka.py`).

Calibration facts (load-bearing; measured tables in the yaml comments):

- RoboLab's relative differential-IK achieves a constant ~28% of any commanded
  delta, so `step_m: 0.072` commanded yields ~20.1 mm measured per decision —
  see `configs/robot_robolab.yaml`, including why settle steps do not help.
- `wrist_rotation_degrees: 270` / `wrist_flip: none` are measured for the Panda
  hand (fingertips at the top of the frame, no mirror). The camera contract
  lives only in this yaml; the runner and the data generators both read it via
  `core.config.camera_contract()` — never restate it elsewhere.

Training-data generation uses the same real2sim core. The RoboLab oracle parses
each task's own subtask declaration, so any single-object pick-and-place task
among the 120 generates without code changes (others raise `UnsupportedTask`).
Generate per task with `real2sim/robolab/record_demos.py` then
`follow_tokenize.py` (commands in
[real2sim/README.md](../scripts/trajectory/real2sim/README.md)), looping over the
task names `--list-tasks` prints to build a set. Quality gate before training:
`python scripts/robolab/check_dataset.py <dir>` (nonzero exit = do not train on
it). Prefer rotation-insensitive objects — the vocabulary has no wrist-rotation
token, so elongated objects are ungraspable.
