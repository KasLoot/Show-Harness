# Quickstart
```bash
python scripts/run_mujoco.py \
  --gui \
  --record \
  --record-global \
  --vlm-backend openai \
  --robot-config configs/robot_mujoco_plug.yaml
```


# Simulators

Show-Harness integrates MuJoCo, ManiSkill, and RoboLab (Isaac Lab). They serve two roles:

1. **Zero-shot evaluation** — run the deployment pipelines (the subgoal planner
   stack or the fine-tuned action model, see `docs/finetuned.md`) in sim
   with no robot attached.
2. **Sim-to-real training data** — generate demonstrations on the same action
   lattice the real robot deploys with (single-axis, ~2 cm per token), so sim
   rollouts mix directly with real teleop rollouts for fine-tuning.

The default integrations keep the deployment contracts: the same nine-token action
vocabulary, the same image transforms (`core/record/images.py`), and measured
per-config step calibration so one token means ~2 cm of physical travel.
MuJoCo also offers an opt-in Cartesian vocabulary for plug insertion, described below.

## MuJoCo + cloud models

Run the zero-shot planner/controller with a Franka Panda in a local MuJoCo
pick-and-place scene. Both MuJoCo tasks default to OpenAI `gpt-5.6-sol` with medium
reasoning. Ollama (`kimi-k3:cloud`) and Gemini remain available with
`--vlm-backend ollama` and `--vlm-backend gemini`.
No local model server, model weights, GPU, or robot
hardware is required. The initial scene contains a red cube and a blue target pad.
The shared hardware imports may print a `pyrealsense2` warning; MuJoCo does not use it.

Both tasks share a furnished laboratory in `assets/mujoco/lab.xml`: a four-legged
workbench, floor, walls, cabinets, and instrument benches. The tabletop remains at
world Z=0 with its original contact geometry and friction. Surrounding furniture
is visual-only. The GUI starts with a wider lab overview; the named VLM cameras
retain their positions, projections, and image order. A separate `lab_overview`
camera supplies the recording-only global panel and presentation renders; it is
not sent to the VLM.

```bash
# Creates/uses .venv with uv; downloads only the official Panda MJCF and meshes.
bash scripts/setup.sh mujoco

# Use an existing exported key, or add this line to gitignored configs/secrets.env:
# OPENAI_API_KEY="your-key"

# Check all six 2 cm moves, empty-grasp recovery, three cameras, and recording (no API calls).
uv run --no-project python scripts/run_mujoco.py --smoke-test --record

# Run the complete model-controlled task; exits 0 only on physical success.
uv run --no-project python scripts/run_mujoco.py
# Add --record to save videos and their annotated timeline.
uv run --no-project python scripts/run_mujoco.py --record
# Add --record-global for a separate clean 1920x1080 global lab video (also enables --record).
uv run --no-project python scripts/run_mujoco.py --record-global
# Use the existing Gemini profile instead (requires GEMINI_API_KEY):
uv run --no-project python scripts/run_mujoco.py --vlm-backend gemini
# Use Kimi K3 via Ollama instead (requires OLLAMA_API_KEY):
uv run --no-project python scripts/run_mujoco.py --vlm-backend ollama

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
`--vlm-backend` selects a profile; `--model`, `--reasoning-effort`, `--vlm-url`,
`--max-steps`, and `--log-dir` override it for one run. The OpenAI profile uses
`vlm_backends.openai.model`, `OPENAI_MODEL`, and `OPENAI_API_KEY`.
Set `vlm_backends.openai.reasoning_effort` or `OPENAI_REASONING_EFFORT` to adjust
reasoning; `--reasoning-effort` takes precedence over the environment and YAML.
The default model supports `none`, `low`, `medium`, `high`, `xhigh`, and `max`.
Set `vlm_backends.gemini.model` or
`GEMINI_MODEL` to change the Gemini model; the Ollama profile uses
`vlm_backends.ollama.model`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL`.
The selected backend determines the endpoint and API key, so `--model` alone
does not switch providers. The model must accept images.

OpenAI uses `/v1/chat/completions`, ordered camera images, JSON responses, and
`reasoning_effort: medium`. The API performs internal reasoning while the harness
keeps its existing one-action JSON response contract (`reasoning_cot: false` controls
the visible response format, not the API reasoning setting). The 8192-token client
budget includes reasoning and the final answer; role-specific budgets still apply.
Requests use `max_completion_tokens` and omit temperature while reasoning is enabled.
See the [GPT-5.6 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
and [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Gemini uses Google's [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai)
with JSON output, `GEMINI_API_KEY`, and low reasoning effort. Gemini 3 calls use
the backend's configured temperature (1.0), including planner/controller calls
that otherwise request zero, following
[Google's temperature guidance](https://ai.google.dev/gemini-api/docs/gemini-3#temperature).

Ollama uses native `/api/chat` with bearer authentication, ordered
base64 camera images, and non-streaming replies. Ollama Cloud currently does not
support structured output constraints, so the harness supplies the JSON contract
in the prompt and validates action tokens locally. `reasoning_effort: low` is sent
as `think: "low"`; `max_tokens` includes the model's reasoning budget.
See [Ollama Cloud](https://docs.ollama.com/cloud),
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs), and
[Kimi K3](https://ollama.com/library/kimi-k3:cloud).

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
`observation_context` and `prompts/controller_mujoco.txt` describe the three model views
and their action directions to the model; update them when changing camera geometry.
For the original task, the planner and controller both receive separate images in
this order: Wrist (A), Front (B), Right Side (C). Plug insertion adds Angled Wrist (D)
and a metric Wrist depth text array, described below. Front is on the +X side and Right Side is on the +Y side;
both are orthographic views tilted 30 degrees downward toward the table. Front's
horizontal position isolates Y alignment; Right Side's isolates X alignment.
In Front, FWD approaches the camera and BACK recedes; RIGHT/LEFT move image-right/left.
In Right Side, FWD/BACK move image-left/right; RIGHT approaches the camera and LEFT
recedes. UP moves vertically away from the table and DOWN moves vertically toward it.
Depth and height can overlap visually, so cross-check the views; orthographic depth
does not change apparent size. Wrist is the primary
guide for fine horizontal alignment. "Above the target" means alignment in the table
plane with positive vertical clearance, rather than alignment in a single image.
Before closing, Wrist must show the graspable part between the fingertips; overlap
in one external view is insufficient. Camera translation parallel to the Wrist image
plane makes stationary targets shift oppositely in the image; motion along the Wrist
line of sight mainly changes apparent size.
Side/front `pos`, `xyaxes`, `projection`, and `fovy` are set in both scene XML files
under `assets/mujoco/`. For these orthographic cameras, `fovy` is the vertical span
in metres; a smaller value zooms in. The old AgentView and separate global cameras have been removed.
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
built-in physical predicate is selected with the scene (`pick_place` or `plug_insert`).
When adding another scene/task, update `MujocoSession.check_success` along with it. This scene uses
stock Panda fingers and has not been calibrated for the released fine-tuned policies.

### Plug insertion and explicit Cartesian movements

```bash
# Same cloud backend, planner, single-action controller, and logger; adds angled RGB and metric depth text.
uv run --no-project python scripts/run_mujoco.py --robot-config configs/robot_mujoco_plug.yaml
# Check all 36 sized translation/rotation commands, empty grasp, and cameras without an API call.
uv run --no-project python scripts/run_mujoco.py --robot-config configs/robot_mujoco_plug.yaml --smoke-test
# View and optionally record on macOS:
.venv/bin/mjpython scripts/run_mujoco.py --robot-config configs/robot_mujoco_plug.yaml --gui --record
```

The scene contains an orange-handled plug on a gray stand, a silver cylindrical pin
20 mm in diameter, and a blue socket with an open black cylindrical bore nominally
26 mm wide and 50 mm deep. Black interior surfaces make the opening distinct without
filling it. The hand must securely grasp the handle, lift it 50 mm (5 cm), verify the
pin clears the stand and tall blue socket, transport the plug, align the pin axis with
the bore, insert, release, and retreat. The round shapes do not
require keyed yaw alignment; rotations remain available to correct tilt when needed.
MuJoCo contacts and
friction hold the plug; there is no grasp attachment, snap-to-socket, or runtime
scripted solution. Success checks depth, radial fit and shaft tilt, settled motion,
open fingers, and gripper withdrawal. Hovering, resting on the rim, or emitting
`DONE` does not count as insertion.

Every model decision still produces one atomic token and executes one fixed movement
before the next observation. The configured sizes are:

| Size | Translation | Rotation |
| --- | --- | --- |
| `SMALL` | 0.002 m (2 mm) | 2° |
| `MEDIUM` | 0.01 m (1 cm) | 10° |
| `LARGE` | 0.05 m (5 cm) | 30° |

Append a size to a translation, for example `MV_FWD_LARGE`, `MV_RIGHT_MEDIUM`,
or `MV_DOWN_SMALL`. Directions use fixed world axes: FWD/BACK = ±X,
RIGHT/LEFT = ±Y, UP/DOWN = ±Z. Rotation tokens are
`ROT_{X|Y|Z}_{POS|NEG}_{SMALL|MEDIUM|LARGE}`, for example `ROT_Z_POS_LARGE` or
`ROT_Y_NEG_SMALL`. POS follows the right-hand rule; NEG reverses it. A rotation
turns around the current TCP without translating it. Rotations compose in order;
translations, including lifts while holding an object, preserve the full orientation.
`GRASP`, `RELEASE`, and `DONE` retain their existing semantics; `STOP` holds position.
Bare translation tokens use `fine_step_m` (2 mm in the plug configuration).

The task and planner context require a separate post-grasp LIFT milestone. With the
default sizes, the VLM chooses one `MV_UP_LARGE` (50 mm) and then inspects fresh images
and measured movement. A commanded lift does not prove the plug rose: completion
requires measured lift, secure carry without slip, and visible clearance between the
pin bottom and the stand/socket along the transport path. Incomplete movement calls
for inspection and justified remaining upward steps; insufficient clearance calls for
an additional safe lift before lateral travel. A completed LIFT is not restarted in
transport or later stages just because a stage/recovery begins. A new secure grasp
after recovery requires a new lift. Current stage and available history supply this
context; no persistent lift gate or automatic motion is added. Customized command
sizes remain authoritative when choosing steps to meet the 50 mm goal.

`configs/robot_mujoco_plug.yaml` layers on the existing MuJoCo config and selects
`scene: plug_insert`. Edit its `cartesian_motion.translation_steps_m` and
`cartesian_motion.rotation_steps_deg` maps to tune the three sizes; both require
finite positive values with small < medium < large; rotation steps must be below
180° so the quaternion endpoint represents the requested turn. The prompt and executor share
these values. `--cartesian-motion` enables the same vocabulary on the original scene
without changing its default behavior. Explicit sizes cannot be combined with
`--variable-step`, which chooses sizes automatically. The extended vocabulary is
for this zero-shot MuJoCo path; real hardware and released fine-tuned policies keep
their existing vocabularies.

`prompts/controller_mujoco_plug.txt` and `cartesian_observation_context` describe
the insertion task and its four RGB images in order: Wrist (A), Front (B), Right Side (C),
Angled Wrist (D), accompanied by a `WRIST_DEPTH_MM` text observation. The original task retains its three-camera prompts, including
`controller_mujoco_cartesian.txt` when Cartesian movements are enabled.

The plug configuration sets top-level `visual_history_steps: 1`, retaining one previous
observation alongside its executed action and measured movement. This sends four current
RGB images plus four historical RGB images after the first action, with a matching metric
depth array and calibration for each observation. Historical depth belongs to the same
before-action moment as its images. Change this number to adjust history length, or set
it to `0` to disable visual history. Requests place CURRENT A-D first, then clearly labeled
HISTORY step sets from oldest to newest. The prompt uses those pairs to assess motion
and correction effects while keeping current images authoritative for the next action.
History continues across normal stage transitions and recovery, and resets on a full
replan or new episode. Each decision still executes one atomic action.
The existing three-action text memory remains enabled. The plug task saves every controller
prompt in `controller_prompts/`, with image source steps and pixel fingerprints so the
current/history pairing can be checked against the saved camera PNGs.
The model compares the pin against the rim and the handle against the fingertip pads
across these images. TCP descent alone is not plug insertion: a stationary pin or a
handle moving upward between the fingers indicates blockage/slippage and calls for
ending descent, restoring safe clearance and secure contact, and realigning.

The added `wrist_insert` camera is attached to the hand with a 65° field of view
and looks obliquely below the TCP. Its image center is not the grasp point. Both
wrist cameras rotate with the hand; measured image-right, image-down, and available
sightline world vectors in the prompt guide world-axis commands. Stationary targets
shift opposite camera translation parallel to each image plane. Front isolates Y
alignment and Right Side isolates X. Grasp checks continue to require independent
evidence from A/B/C; D is supplementary for grasping.

`wrist_depth` supplies metric depth text from the original Wrist camera with the same
projection as A. The plug configuration uses
`wrist_depth: {representation: text, grid_rows: 32, grid_cols: 32, near_m: 0.0, far_m: 0.30}`.
The plug prompts expect `representation: text`; grayscale diagnostics are saved without
switching this setting.
The `WRIST_DEPTH_MM` serialized observation contains a `depth_mm` array, normally
32 × 32 point samples from the raw metric renderer, rounded to 1 mm. Rows run top to
bottom and columns left to right. `source_width_px`/`source_height_px` describe the
source image; `sample_y_px`/`sample_x_px` give the exact zero-based RGB pixel indices sampled by
each array row/column. Each entry measures one pixel, not a tile average. Grids clamp
to smaller source dimensions rather than duplicate samples. Invalid or out-of-range
samples are `null`. This sparse array can miss narrow edges or finger pads and does
not supply a depth value for every RGB pixel.

The planner and controller receive the matching range and camera-to-TCP calibration
(currently 48 mm behind the TCP). Array values measure camera-axis distance in
millimetres, not world Z or TCP clearance; wrist rotation changes the relationship to
vertical height. The prompt asks the model to identify visible surfaces in A, map them
to sampled pixels, and compare handle and fingertip-pad depths before closing. Similar
depths cannot establish a secure body grasp or rule out shallow edge contact. Hidden
or unsampled surfaces have no depth evidence; `null` is not proof of free space.
The fixed-scale grayscale PNG/video remains a local diagnostic, with white near and
black far and no per-frame normalization. It is not sent to the VLM.
The upright handle is 32 mm tall, placing its side-body center 16 mm below its top.
With a downward wrist and a visible upright handle, a centered body grasp therefore
puts the top face approximately 16 mm nearer the camera than the calibrated TCP plane
(about 32 mm camera depth with the current 48 mm TCP calibration). This geometric
check does not apply to a tilted wrist/handle or hidden top; the prompt requires
cross-checking the other views and visible pad depths.

For insertion, D shows the silver pin tip, black bore rim, lateral offset, and tilt.
The prompt tells the controller to align the actual pin axis with the bore, not the handle or housing
center. At hover, different heights produce parallax in D: the prompt asks the model
to project the pin axis to the rim plane, center the full pin within the opening, and
cross-check the orthogonal views, rather than requiring tip/hole pixel coincidence.
The nominal radial clearance is only 3 mm. Transport/alignment DONE requires this
check against the actual bore, and the model must recheck at final approach and before
each downward step. A stage transition or silhouette overlap does not establish fit.
Asymmetric rim visibility and uncertainty comparable to that clearance call for closer
inspection; equal pixel gaps are not required in the oblique view. The existing
one-sentence reasoning must cite the current X/Y pin-to-rim checks for insertion DOWN
or alignment DONE, without changing the JSON response schema.
Visible lateral offset calls for a justified small correction while clear above the rim.
If the rim is hidden before entry, the prompt favors a safe small lift to inspect and
realign instead of blind descent or repeated STOP. History can reference a previously
visible rim against current fixed-camera landmarks; it cannot override current evidence.
Once entry is established, rim occlusion is expected; current external depth, measured
pose, and actual motion support depth judgments without repeated withdrawal for occlusion
alone. Stalled or off-center entry calls for lifting clear before lateral correction.
Small steps and withdrawal after a stalled insertion remain required. Destination
release requires visible seating and support by the socket; the model is instructed
not to release a hovering pin and rely on gravity to complete insertion.
Final RETREAT is a separate stage from the initial post-grasp LIFT. After release,
the VLM chooses one default `MV_UP_LARGE` (50 mm), then checks fresh observations:
the fingers must be open, the plug must remain seated and undisturbed, and the TCP
must be strictly more than 50 mm vertically above the plug's top. The prompt targets
55 mm for margin and requires further justified upward steps if needed. A 50 mm
command is not itself sufficient because the TCP can start below the top. The model
estimates the visible top from RGB and mapped depth, with the 32 mm handle height as
a scale cue; it receives no privileged object pose. A vague visible gap cannot satisfy
the retreat goal. `plug_success.retreat_clearance_m` defaults to `0.05`; the checker,
task, and prompts share that threshold, and the prompt's target is 5 mm above it.
This preserves the strict physical success check.
Each step and `summary.json` now include `success_diagnostics`, with measured values,
thresholds, and failed criteria such as `retreat_clearance`. These diagnostics are
for inspection and do not supply privileged object state to the VLM.
These are visual
instructions for the VLM; no object-state alignment gate or automatic correction is added. A supported-source
release still allows retrying a visibly off-center grasp. The configured TCP floor
and Panda joint limits still apply.
Step records include the selected size, translation distance or rotation angle/vector,
and full measured/target poses. Physical tests verify the action geometry and a
scripted insertion; they do not establish a cloud VLM's task success rate.

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
| `side.mp4` | Orthographic Right Side view, tilted 30 degrees downward |
| `wrist.mp4` | Gripper-mounted camera |
| `front.mp4` | Orthographic Front view, tilted 30 degrees downward |
| `wrist_insert.mp4` | Plug only: oblique hand-mounted Angled Wrist view |
| `wrist_depth.mp4` | Plug only: local diagnostic grayscale depth aligned with Wrist RGB; not sent to the VLM |
| `combined.mp4` | Existing camera/decision canvas intact on the left, with a full-height global lab view appended on the right |
| `global.mp4` | Optional: clean native 1920 × 1080 render from the fixed `lab_overview` camera, without a title or overlays |
| `annotations.jsonl` | Timestamped decisions, full VLM output, execution, recovery, and results |

The camera videos are 512 × 512 at the default resolution. The original task's
four-panel canvas remains 1024 × 1088 on the left; its combined recording is
2080 × 1088 after adding the global view. The plug's original 1024 × 1632 canvas
remains intact on the left, with Wrist Depth and the decision panel in its last
row. Its combined recording is 2624 × 1632, including the 1600 × 1600 global
image and its title bar on the right. This fixed `lab_overview` camera is captured
only by the recorder, independently of the movable GUI. It is absent from VLM
images, depth input, and visual history. Append `--record-global` to `--record` to
write `videos/global.mp4` at the native 1920 × 1080 resolution, with no title or
overlays. The equivalent config is `recording.global_video: true` when recording
is enabled. The default `--record` output does not include this separate file.
All videos in a run, including the optional global video, have identical frame
counts and run at 30 fps with the same frame timing. Frames are captured during
physics steps, so the recordings show the motion between actions. Cloud waiting
time is omitted; each action's result is held for one second for readability.
These video pauses do not advance physics.
Change `recording.fps` and `recording.decision_hold_s` in the config to tune
recording. MuJoCo uses this synchronized recorder instead of the generic per-decision
analysis video. Without either recording flag, no videos are created; ordinary
step logs and observation images are still saved. `summary.json` and the printed
`video_path` point to the combined video
when recording is enabled and use an empty string otherwise.

Each annotation includes UTC, simulation time, video time, frame index, stage,
decision source, and explanatory text. `decision` events precede the motion;
`step_complete` events include the action result and `[frame_start, frame_end)`
interval in every video (indices start at zero). `vlm_output` retains the full
answer even when the canvas shortens long text. Gripper commands, including an
automatic reopen after an empty grasp, are annotated separately. Interruptions
and errors finalize the videos and preserve the log; frames stream to disk as
playable fragmented MP4s while the run is in progress.

Wrist, Front, and Right Side feed the model in that order, with Angled Wrist appended
for plug insertion, independently of the interactive viewer. Plug requests also carry
the matching metric depth text. The recorder keeps those RGB views plus the local
grayscale depth diagnostic. Observations are saved in `images/side`, `images/wrist`, and
`images/front`, plus `images/wrist_insert` and `images/wrist_depth` for the plug task.
Raw depth arrays in metres and their serialized model text in millimetres are saved as
`depth/wrist/0000.npy` and `depth/wrist/0000.txt` for each step. Initial planner depth
is saved as `planner_wrist_depth_m.npy` and `planner_wrist_depth.txt`.
`--smoke-test --record` also writes example recordings to
`rollouts/mujoco/smoke_test/videos/`, marked `smoke_test` rather than VLM decisions.

### Replay a saved MuJoCo run

Replay reconstructs the scene and re-executes logged actions through the same smooth
actuator motion and MuJoCo physics, without calling a VLM. Older logs contain actions
and measured TCP waypoints, rather than every physics state: playback reconstructs
the motion between observations instead of joining endpoints with straight lines.
It is not an exact restoration of a saved per-timestep simulation. Use `--verify` to
check the replayed TCP poses against the logged waypoints and detect divergence;
this does not verify full object states. Replay writes `replay_report.json` into
its separate output directory.

Replace `RUN_DIR` with the saved run directory:

```bash
# Inspect the recorded actions/configuration without executing the replay.
uv run --no-project python scripts/mujoco/replay.py RUN_DIR --dry-run
# macOS: smooth interactive playback, pose verification, and keep the final view open.
.venv/bin/mjpython scripts/mujoco/replay.py RUN_DIR --gui --verify --hold-final
# Save new synchronized replay videos into a separate, new output directory.
uv run --no-project python scripts/mujoco/replay.py RUN_DIR --record --output-dir NEW_REPLAY_DIR --verify
```

On other platforms, the normal Python launcher can use `--gui`. Replay output belongs
in a new directory so the original run and its recordings remain available for comparison.

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
