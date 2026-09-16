# MuJoCo motion and recording validation

Validated on 2026-09-16. The episode logger uses UTC+8, so this run's directory
is dated `0917`. This profile intentionally changes the earlier ablation setup.

## Requested changes

* Cube: approximately **40 × 39.65 × 39.78 mm**, uniformly scaled from the source
  mesh; mass remains **0.2 kg**. Visual and collision geometry scale together.
* Grip: **80 N closing-force target per finger**, with smooth travel, native
  damping, closed-stop protection, and measured-width settling.
* Motion: smooth actuator-driven trajectories with joint feedback. The live
  controller does not assign robot joint positions. Observations wait for the
  entire previous action, including an automatic empty-close reopen.
* Recording: end-effector XYZ/RPY and quaternion poses, seven arm joints, two
  finger joints, force feedback, request IDs and timestamps after each action;
  full state/inputs at every physics step; continuous simulation-clock video.

## Automated checks

**83 passed, 1 skipped** after the `full`/`no_fail` playback update (endpoint
playback initially passed 80 tests; the earlier live motion change passed 75).
Ruff, whitespace and repository leak checks passed.
Relevant checks cover matching visual/collision scaling, unchanged cube mass,
stronger measured grip force, closed travel stops, intermediate motion states,
settled target poses, observation blocking, rejection of stale requests,
workspace-target resynchronization, endpoint units/schema, video timing,
API-free replay, and preserving recordings on controller/rendering errors.

## Final live Gemini run

Task: **Put the cube in the bowl**. Model: **gemini-3.8-flash**, medium reasoning.
The run used the existing three-view experimental prompt flags.

| Measurement | Result |
| --- | ---: |
| Independent physical task result | Success |
| Controller steps | 34 |
| Empty grasps / dropped grasps | 1 / 0 |
| HTTP requests / action endpoint records | 35 / 35 |
| Recorded physics states | 34,735 |
| Continuous video | 2,085 frames, 30 fps, 640 × 512 |
| Video duration / simulated motion duration | 69.50 s / 69.468 s |
| Live episode wall time | 222.25 s |
| Maximum completed-move position error | 0.159 mm |
| Measured force after the nonempty grasp | approximately 79.87 N and 80.00 N |
| Minimum finger joint position | −3.0e−10 m (numerical zero) |

All requests were made after completed controls. Every request's observation
timestamp matched the current simulation timestamp, and simulation time did not
advance while awaiting its API response. Endpoints contain all six Cartesian
pose components and all seven arm joints.

The remaining empty grasp was a VLM decision; stronger force cannot grasp an
object when the fingers are above it. This is one functional validation run,
not a general task-reliability measurement.

Local artifacts (generated run data is ignored by Git):

* [Continuous motion video](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/rollout_success.mp4)
* [Pose/joint endpoints](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/action_endpoints.jsonl)
* [Request timing](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/request_timing.jsonl)
* [Trajectory manifest](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/trajectory_manifest.json)
* [Physical evaluation](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/evaluation.json)
* [Measurements](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/validation.json)

## Diagnostic control replay (original validation)

Re-simulating the recorded controls with video rendering disabled processed the
69.468 seconds of dynamics in **2.86 seconds** (simulation loop time), with
**zero API calls** and **zero maximum joint-state difference** from the recorded
trajectory on this machine. Model loading/decompression is additional startup
work. State playback is also available, and GUI replay can run at a chosen speed.

```bash
.venv/bin/python scripts/replay_mujoco.py \
  rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10 \
  --mode states --gui --speed 2 --no-video
```

Rendered pixels are not guaranteed to be bit-identical after a model reload;
the original request PNGs are saved separately. The exact-reproduction check
above concerns the physical state trajectory, not GPU rasterization.

## Endpoint playback (new default)

The default playback now plans a new trajectory from the saved poses and joint
positions. It restores one initial scene state, joins nearly collinear movement
waypoints, cruises at a configured physical tool speed, and uses smooth ramps
at direction changes and gripper events. It does not use the recorded per-step
controls, motion duration or inference timing. Stops are retained at corners
and grasp/release events; intermediate decisions and planner `DONE` do not add
a pause.

The same 35 endpoint records were re-executed with a **0.10 m/s** tool-speed
setting, using no API calls. The cube was physically grasped and placed in the
bowl successfully. This is an offline execution of the saved action path;
it does not count as another live VLM success.

| Measurement | Result |
| --- | ---: |
| New simulated trajectory duration | 19.648 s |
| New video | 591 frames, 30 fps, 19.70 s |
| Commanded tool speed during cruise | 0.09993–0.10004 m/s |
| Actual median tool speed during cruise | 0.10005 m/s |
| Actual 5th–95th percentile cruise speed | 0.09980–0.10698 m/s |
| Largest transient tool tracking error | 3.72 mm, during the loaded lift |
| Largest completed-event position error | 0.496 mm |
| Largest completed-event orientation error | 0.00118 rad |
| Source physics states used | 1, initialization only |

Tool speed is constant in the commanded cruise portion, with smooth ramps at
starts/stops. Contact forces cause physical tracking deviations, particularly
when first lifting the cube. Short moves and angular/joint constraints can
reduce the attainable peak speed.

New tests verify that changing old timestamps and sample indices leaves the
playback unchanged, removing an endpoint changes the physical destination,
no dense states beyond initialization are read, and live joint positions are
only updated by physics. Additional checks cover constant-speed passage through
joined waypoints, bounded joint acceleration, preservation of empty-grasp
close/reopen events, and rejection of incomplete endpoints.

* [New endpoint-playback video](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/replay_endpoints.mp4)
* [New trajectory report](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/replay_endpoints.json)
* [Final validation](../rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10/endpoint_playback_validation.json)

```bash
.venv/bin/python scripts/replay_mujoco.py \
  rollouts/mujoco_motion_live_final/gemini-3.8-flash/0917/task_0/01-05-10 \
  --gui --linear-speed 0.10 --no-video
```

Remove `--no-video` to save a new playback video. GUI-only playback avoids the
offscreen-rendering cost; the saved video always follows the newly planned
simulation timeline. To change movement speed, set `--linear-speed` in metres
per second. `--speed` is reserved for the diagnostic `states`/`controls` modes.

## Filtering failed grasps

Validated `--playback-type full|no_fail` on the user's saved run
`rollouts/mujoco_robolab/gemini-3.8-flash/0917/task_0/01-46-06`, using the requested
`--linear-speed 1.0` with the existing angular/joint/acceleration limits.

| Measurement | `full` | `no_fail` |
| --- | ---: | ---: |
| Simulated duration | 16.038 s | 13.386 s |
| Skipped empty-grasp attempts | 0 | 1 |
| Gripper close/open events | 4 | 2 |
| Independent task result | Success | Success |

Action **13**, request **14**, was explicitly marked `grasp_empty: true`. The
filtered playback omits its close, automatic reopen and small settling pose
change. Downward moves 7–12 and 14 form one continuous movement. Positioning
corrections 15–16, successful grasp 17 and release 32 remain. `full` reproduces
every trajectory sample of the pre-change playback exactly.

Tests cover joining across removed attempts, successful grasp/release
preservation, missing outcomes, incomplete records, first/last failed grasps,
all-failed logs, separate output filenames and rejection of filtering in
diagnostic state/control replay. Source logs and live inference are unchanged.

```bash
.venv/bin/python scripts/replay_mujoco.py \
  rollouts/mujoco_robolab/gemini-3.8-flash/0917/task_0/01-46-06 \
  --gui --linear-speed 1.0 --playback-type no_fail --no-video
```

* [Filtered playback video](../rollouts/mujoco_robolab/gemini-3.8-flash/0917/task_0/01-46-06/replay_endpoints_no_fail.mp4)
* [Filtered playback report](../rollouts/mujoco_robolab/gemini-3.8-flash/0917/task_0/01-46-06/replay_endpoints_no_fail.json)
* [Filter validation](../rollouts/mujoco_robolab/gemini-3.8-flash/0917/task_0/01-46-06/playback_filter_validation.json)

See [run/replay commands and file formats](mujoco.md).
