# MuJoCo: RoboLab scene, Show-Harness VLM loop, continuous recording

This backend converts **RoboLab's `RubiksCubeTask`** ("Put the cube in the bowl")
to MuJoCo and runs it through Show-Harness's existing zero-shot planner,
controller, plugins, and Franka action interpreter. The default model is
`gemini-3.8-flash` through Google's API. Isaac Sim, torch, and a local VLM server
are not required.

The current user-requested profile uses a **40 mm cube**, retaining its **0.2 kg
mass**, and an **80 N closing-force target per finger**. These are explicit
runtime overrides of the source scene/actuation settings. Source mesh files
remain unchanged; each run records its effective settings and compiled model.

See [the validation report](mujoco_validation.md) for measured results, including
the failed final trial. Successful demonstrations do not establish deployment reliability.
The [controlled ablation report](mujoco_ablation.md) contains 18 episodes testing
camera count, camera descriptions, and the visual grasp condition one factor at
a time. None of those changes established a reliable grasp-centering fix.
Those historical results used the earlier cube and motion settings; they are
not results for the smaller-cube/stronger-grasp profile.
See [motion and recording validation](mujoco_motion_validation.md) for the current
profile's live run, synchronization checks, and replay measurements.

## Setup and run

```bash
bash scripts/setup.sh mujoco

# Render both cameras and measure action distances; no API calls.
.venv/bin/python scripts/run_mujoco.py --no-vlm --probe-axes

# Uses GEMINI_API_KEY from your environment or configs/secrets.env.
.venv/bin/python -u scripts/run_mujoco.py --gui

# The previously tested three-view experimental input, now with continuous motion video:
.venv/bin/python -u scripts/run_mujoco.py --gui --extra-view side \
  --describe-side-camera --side-grasp-check

# Optional multi-camera motion video; the default records the fixed front view.
.venv/bin/python -u scripts/run_mujoco.py --extra-view side --video-layout multiview

# Independent episodes, each starting from the authored layout:
.venv/bin/python -u scripts/run_mujoco.py --episodes 3 --max-steps 120
```

Setup downloads a pinned asset dependency closure (about 27 MB for RoboLab),
verifies Git LFS hashes, and converts the USD scene offline. The bowl's convex
decomposition can take several CPU minutes on the first run; it is cached.
`models/mujoco/` also contains the Panda arm assets and generated meshes.
The runtime only needs the generated meshes/JSON/XML, not the USD converter.

Headless runs default to EGL. `MUJOCO_GL=osmesa` can be used on systems with
Mesa OSMesa installed. `--gui` uses MuJoCo's viewer and requires a display.
Configuration: [`configs/robot_mujoco.yaml`](../configs/robot_mujoco.yaml).
CLI overrides include `--model`, `--max-steps`, `--episodes`, `--log-dir`,
`--robot-config`, `--cube-size 0.04` (metres), `--grip-force 80` (N per finger),
`--video-fps 30`, and `--video-layout front|multiview`.
Only `RubiksCubeTask` is ported at present.

## Movement completion and grasp force

Each Cartesian target is converted to a smooth, speed-limited joint-reference
trajectory. MuJoCo's actuators and physics move the arm through intermediate
states. The live controller never assigns joint positions to teleport to a goal.
Joint feedback compensates steady payload sag. A move returns only after its
position error is within **0.2 mm**, orientation error within **0.005 rad**, and
arm joint speed below **0.02 rad/s** for **0.1 s**. A motion that cannot settle
times out with an error, preserving its partial trajectory and video.

The same action lock covers gripper closure and an automatic empty-grasp reopen.
Camera capture waits for that lock; the HTTP wrapper also rejects an observation
whose simulation timestamp is older than the current state. Consequently the
next VLM request uses images captured after the previous action completes.
Workspace clipping is recorded and the cached command target is resynchronized
with the measured pose.

The gripper ramps its reference at 0.04 m/s and limits the closing force to its
configured target, with native velocity damping. Completion uses sustained force
and settled jaw width; opening waits for the measured open position. Force is
maintained during subsequent arm motion. The old position servo's 200 N limit
was not its normal gripping force: with a 40 mm opening, its 2000 N/m gain would
produce approximately 40 N per finger. The new default targets approximately
80 N per finger. Increasing only that old limit would not increase this force.
The closed mechanical stops are stiffened for the stronger actuator, and encoder
feedback removes preload near the fully closed position on an empty grasp.

Settings live under `motion_control` and `gripper_control`. Setting
`cube_size_m: null` restores the source cube size. Changing cube size does not
change its mass or the bowl/table geometry.

## Recorded files and offline replay

Every run directory contains:

| File | Contents |
| --- | --- |
| `rollout_success.mp4` / `rollout_failure.mp4` | Continuous motion video at 30 fps by default, sampled during physics steps |
| `decision_frames.mp4` | Original per-decision image/telemetry montage, retained separately |
| `action_endpoints.jsonl` | Before/after measured end-effector and joint poses for every executed atomic action, including recovery/queued actions |
| `request_timing.jsonl` | Request ID, observation timestamp, completion status and API wall time |
| `requests/NNNN/` | Exact images, prompt, response and parameters for each HTTP attempt |
| `trajectory_manifest.json` + `trajectory/chunk_*.npz` | Full integration state and applied controls/inputs at every 2 ms physics step |
| `model.mjb.gz` | Compressed, self-contained compiled model, including meshes/textures, for replay |

`--no-vlm --probe-axes` produces `motion_check.mp4` and the same pose/trajectory
records for the six calibration moves. `motion_live.mp4` is streamed while a run
is active. Trajectory chunks are flushed after actions and before API calls.
Frames include camera labels and the active action for viewing only; the VLM
continues to receive unannotated camera images.

Each endpoint's `after` block contains:

* `eef_pose6d_xyz_rpy`: `[x, y, z, roll, pitch, yaw]`, robot-base frame, metres/radians;
* `eef_pose_xyz_quat_xyzw`: the same pose with quaternion orientation;
* `arm_joint_positions_rad`: the seven Panda joints, ordered `joint1` through `joint7`;
* `finger_joint_positions_m`: the two finger slides;
* joint velocities, jaw width, finger actuator forces and simulation time.

The end-effector pose refers to `panda_hand` (the model's `eef` site). The
fingertips extend approximately 132.3 mm along that hand frame's grasp axis.

`request_id` associates an action with the latest HTTP request. Queued moves and
automatic recovery can share a request ID; startup actions have `null`.
The `motions` list records requested/applied targets, convergence errors and
completion or failure status. Euler angles wrap; use the quaternion or recorded
joint states for orientation replay.

### Endpoint playback (default)

Playback reads `action_endpoints.jsonl` and generates a **new smooth trajectory**
through the recorded robot joint poses. Recorded end-effector poses check the
kinematics and define movement directions. The old inference times, movement
durations, per-step controls and subsequent physics states do not determine
playback motion. Only the initial physics state is restored to initialize the
scene; MuJoCo then simulates the entire new execution.

```bash
# Watch the newly planned trajectory and save replay_endpoints.mp4:
.venv/bin/python scripts/replay_mujoco.py <run-directory> --gui --linear-speed 0.10

# Watch without offscreen rendering or video encoding:
.venv/bin/python scripts/replay_mujoco.py <run-directory> --gui --linear-speed 0.10 --no-video

# Omit recorded failed empty grasps and join movement through those attempts:
.venv/bin/python scripts/replay_mujoco.py <run-directory> --gui --playback-type no_fail --no-video

# Generate a video from all cameras in the recorded model:
.venv/bin/python scripts/replay_mujoco.py <run-directory> --linear-speed 0.10 --layout multiview
```

`--linear-speed` is a **physical tool travel speed in metres/second**, not a
multiplier of the original run. The default is **0.10 m/s (10 cm/s)**. GUI and
video playback use the newly planned simulation time at 1×. No VLM/API key is
needed. This is also available explicitly as `--mode endpoints`.

`--playback-type` selects which actions to include:

* **`full` (default):** retain all recorded actions, including failed grasp
  closes and their automatic reopens.
* **`no_fail`:** skip completed `GRASP` actions explicitly marked
  `grasp_empty: true` by the live controller. This removes the attempt's
  close/reopen and settling pose correction, allowing adjacent moves in the
  same direction to join. Subsequent positioning corrections, successful grasps
  and normal releases remain in the path. Missing outcomes and later object
  drops are not inferred as empty grasps. Incomplete/error records are still
  rejected.

The filter uses the saved controller outcome; it does not query a VLM or object
coordinates. `no_fail` is available only in endpoint mode. Diagnostic
`states`/`controls` replay always retains the original sequence.

Consecutive moves within 5 degrees of the same direction are joined into one
continuous path. Joint interpolation passes through their recorded waypoints;
kinematic arc-length parameterization sets a constant nominal tool cruise speed.
There is no stop for an intermediate request or planner `DONE`. Turns, reversals
and gripper events use smooth acceleration/deceleration ramps. Short moves may
not reach full cruise speed. This keeps velocity continuous while preserving
the path's corners and grip locations. Physical tracking and contact forces can
cause small deviations from the commanded speed.

Additional limits are `--angular-speed` (default 0.5 rad/s), `--joint-speed`
(1 rad/s), `--joint-acceleration` (4 rad/s²), and `--ramp-time` (0.25 s).
The ramp time is a minimum for full-speed motion and increases if joint
acceleration requires it; short moves use shorter ramps at reduced peak speed.
The planner respects the most restrictive speed limit. Retained gripper close/open
events keep their order and wait for fresh
force/width feedback. The arm tracks the new references using native actuators,
velocity/inertial feedforward and joint feedback. It never teleports between
recorded robot positions.

Playback saves:

* `replay_endpoints.mp4`: the new continuous trajectory, unless `--no-video`;
* `replay_endpoints.json`: speed settings, completed events, endpoint errors,
  measured cruise speeds and independent task evaluation when provenance exists;
  it also records `playback_type`, `skipped_action_count` and `skipped_actions`
  with original action/request IDs and reasons;
* `replay_endpoints_trajectory.npz`: new timestamps, commanded/actual joint
  positions and velocities, reference accelerations, tool positions/speeds,
  phases and event indices at every physics step. Units are seconds, radians,
  metres and their time derivatives. `cruise` marks the constant-speed portion
  of a path; angular/joint limits can make its linear speed lower than requested.

With `--playback-type no_fail`, the default output names become
`replay_endpoints_no_fail.mp4`, `replay_endpoints_no_fail.json` and
`replay_endpoints_no_fail_trajectory.npz`. This preserves the full playback's
outputs. `--output` can supply another name.

Retiming is a new physical execution, so object/contact motion can differ from
the original recording. Incomplete action records or inconsistent robot poses
are rejected rather than silently treated as successful targets.

### Diagnostic recording replay

The original replay methods remain available explicitly for inspecting the live
run or checking deterministic physics. `--speed` applies only to these modes;
endpoint playback rejects it and directs you to `--linear-speed`.

```bash
# Inspect the exact recorded state sequence at twice its original speed:
.venv/bin/python scripts/replay_mujoco.py <run-directory> --mode states --gui --speed 2 --no-video

# Re-simulate recorded actuator inputs and check numerical agreement:
.venv/bin/python scripts/replay_mujoco.py <run-directory> --mode controls --no-video
```

State playback reproduces recorded robot **and object** motion. Controls mode
advances physics using the recorded inputs. Both save a JSON replay report and,
when video is enabled, `replay_states.mp4` or `replay_controls.mp4`.
Binary models require a compatible MuJoCo version; the recorded version is in
the manifest. The original asset cache is not needed to load the snapshot.
Re-rendered pixels can differ across rendering contexts even when physics states
match. The saved `requests/NNNN/image_*.png` files remain the authoritative images
sent to the VLM.

The live simulator still waits for the VLM between actions. Its recorded video
omits those wall-clock waits; endpoint playback additionally replaces the old
movement timing with the newly planned trajectory.
Rendering/encoding speed depends on the machine; `--no-video` avoids that cost
when watching in the GUI or checking control replay. The default front-only
video costs less to render than the multi-camera layout, independently of which
images are sent to the VLM.

The trace stores MuJoCo's integration state (including warm-start information)
and per-step user inputs so control replay can reproduce the dynamics. See the
[MuJoCo state documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#state-and-control).

## What comes from the original sources

| Component | Source |
| --- | --- |
| Instruction, cube/bowl/table layout, object masses, textures and meshes | [RoboLab task](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/robolab/tasks/benchmark/rubiks_cube_task.py) and `assets/scenes/rubiks_cube_bowl.usda` at that revision |
| Panda arm MJCF | MuJoCo Menagerie `franka_emika_panda`, revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`, derived from Franka's URDF |
| Yellow fingertips, brackets and their collision hull | This repo's `assets/panda_short_finger.stl` and `make_short_finger_asset.py` geometry/transforms |
| Robot home pose, camera extrinsics/intrinsics, wrist mount, arm actuator gains | `core/sim/robolab_franka.py`; gripper control is overridden as described above |
| Camera rotation, crop, and letterbox | `configs/robot_robolab.yaml` through `camera_contract()` |
| Planner/controller and policy plugins | `core.launch.make_runner`, `RealEpisodeRunner`, `prompts/controller.txt`, and the original plugin prompts |
| Movement vocabulary, metric moves, gripper/recovery behavior | `FrankaAtomicController` and `configs/primitives_franka.yaml` |

The source manifest records asset hashes. Generated `provenance.json` records
object poses, bounds, masses, camera constants, source revision and conversion
differences. The source scene has a **red bowl and textured Rubik's cube**; this
is separate from the ManiSkill orange-block/coaster task. The earlier custom
orange-block prototype has been replaced.

## Policy and real-robot transfer

The simulator exposes the same interface as the physical Franka session:

* unannotated front and wrist RGB images;
* measured end-effector pose and gripper width;
* the original action interpreter's measured displacement and grip feedback.

The policy inherits `configs/robot_franka.yaml`'s body, without its hardware site
files. Subgoal planning, multi-view instructions, proprioception, action history,
adaptive steps, action chunking and recovery remain active. MuJoCo now supplies
the smooth trajectory and completion checks, so the inherited smoothing plugin
and repeated fixed-duration settling calls are disabled for this backend.
GUI movement is paced at simulation speed; headless physics can run faster.
The model generates the subgoals and selects the semantic actions; its `DONE`
decisions advance the stages, exactly as in the original real-robot runner.

`coords`, `affordance` and `video_ref` are disabled and rejected if enabled. No
crosshairs, object-coordinate inputs, target tracking, scripted task trajectories,
geometric release gates or simulator success feedback enter the policy. Only the
normal robot workspace/floor limits and original sensor-based recovery apply.

## Evaluation and fidelity limits

After the VLM finishes or exhausts the decision budget, an independent evaluator
checks the original task's open-top container-hull/contact/gripper-detachment
conditions. It uses the source hull algorithm in NumPy/SciPy rather than torch.
It is never queried during policy decisions. Both finger contacts are checked
for detachment (the original Franka contact alias names its left finger).

Logs under `rollouts/mujoco_robolab/` include exact prompts, raw input PNGs,
action telemetry, source provenance, a video, and `evaluation.json`.
`model_declared_success` preserves the VLM's conclusion; `success` and the video
filename report the independent evaluation. Exit codes: 0 for successful
episodes/offline checks, 2 for task failures, 1 for runtime/configuration errors,
and 130 for interruption (remaining episodes are cancelled).

The imported scene is the basis for the configured size/force overrides; it is
not bit-identical PhysX dynamics or RTX rendering. MuJoCo contacts use tuned solver parameters; the bowl
uses 64 CoACD convex pieces rather than PhysX's collision cooking. Object textures
are retained; other mesh collisions use convex hulls. Omniverse MDL furniture
shaders use diffuse-color approximations.
Arm inertias/kinematics come from Menagerie's Panda; finger inertias are derived
from the supplied collision geometry at the original density. Robot gravity is
compensated and arm self-collision is disabled, matching the RoboLab configuration.

The low-level servo clock is calibrated for measured 2 cm moves. Runs use the
zero-shot harness's decision budget, not RoboLab's original 40-second policy
benchmark limit. Report these results as **Gemini/Show-Harness in MuJoCo**, not
as a reproduction of the paper's benchmark scores. No Isaac runtime comparison
has been performed on this machine.

Validation (requires prepared assets):

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_mujoco.py -q
```

Physics fixtures use predefined actions to verify the gripper and hollow bowl.
They do not count as VLM successes; live evaluations are reported separately.
