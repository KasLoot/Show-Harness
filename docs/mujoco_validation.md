# MuJoCo validation results

Validation performed in this workspace on 2026-09-15. The repository's logger
uses UTC+8, so the episode directories below are dated `0916`.

These are the initial port checks. See the later
[controlled ablation report](mujoco_ablation.md) for 18 fresh episodes, saved
three-view inputs, and the measured grasp-offset failure mechanism.

Links into `rollouts/` and `models/` below refer to local generated artifacts,
which are ignored by Git. The ablation report includes a versioned results
summary and example images for readers of a fresh checkout.

## Result

The original RoboLab cube/bowl scene and Show-Harness VLM harness are working
in MuJoCo. The VLM uses raw images and robot proprioception, without target
coordinates, visual markers, reference trajectories, or a geometric placement
controller. **Task execution is still inconsistent and needs further reliability
work before real-robot deployment.**

Three consecutive episodes succeeded. A subsequent run preserving the original
ground-plane visibility failed: the model carried the cube too low, pushed the
movable bowl across the table, and requested an unreachable arm pose. The small
sample and intervening rendering correction do not support a general success-rate
claim. All failures are retained.

## Live Gemini trials

Model: `gemini-3.8-flash`, Google API, medium reasoning effort. Same authored
initial layout, 120-step budget, original subgoal/controller prompts and policy
plugins. Step counts include subgoal completion and recovery steps.

| Trial | Result | Steps | Evidence |
| --- | --- | ---: | --- |
| Batch 1 | Success | 68 | [Evaluation](../rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_0/00-45-03/evaluation.json), [video](../rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_0/00-45-03/rollout_success.mp4) |
| Batch 2 | Success | 55 | [Evaluation](../rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_1/00-49-55/evaluation.json), [video](../rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_1/00-49-55/rollout_success.mp4) |
| Batch 3 | Success | 96 | [Evaluation](../rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_2/00-53-57/evaluation.json), [video](../rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_2/00-53-57/rollout_success.mp4) |
| Final source-visibility check | Runtime failure; placement unfinished | 80th action attempted | [Console log](../rollouts/mujoco_robolab_final/gemini-3.8-flash/0916/task_0/01-02-34/console.log), [video](../rollouts/mujoco_robolab_final/gemini-3.8-flash/0916/task_0/01-02-34/rollout_failure.mp4) |

The three-run batch rendered the ground plane gray. The source USD marks it
invisible, which was corrected before the final trial. The collision geometry,
robot, task assets, and policy were unchanged between those trials.

The final trial stopped at `MuJoCo IK cannot reach [0.7065, 0.39, 0.1967]`.
Its video contains 79 executed-step records. This run exposed a reporting gap:
the inherited runner's exception summary said `max_steps_exceeded` and bypassed
independent evaluation. The adapter now records `runtime_error` and independent
scoring on that path; a regression test covers it. The historical raw run was
not rewritten or relabeled as a success.

An earlier development trial, before the final asset/evaluator conversion, also
failed after 58 steps. It remains under
`rollouts/mujoco_robolab/gemini-3.8-flash/0916/task_0/00-37-14` and is excluded
from the three-run batch above.

## Automated checks

* Full suite: **55 passed, 1 skipped**.
* Ruff, whitespace checks, shell syntax check and repository leak check passed.
* Authored object positions, masses, robot home pose and invisible ground checked.
* Original planner/controller/interpreter classes and controller prompt verified.
* Model observations checked against the exact unannotated renderer captures.
* Physical grasp/lift and placement into the hollow bowl passed.
* Unreachable IK targets, runtime-error reporting, cancellation and key redaction checked.

The scripted physics fixture verifies mechanics; it is not counted as a VLM trial.

## Motion calibration

Command: one 20 mm move on each base-frame axis. Measured main-axis displacement:

| Action | Measured magnitude |
| --- | ---: |
| MV_FWD | 19.922 mm |
| MV_BACK | 19.827 mm |
| MV_LEFT | 19.919 mm |
| MV_RIGHT | 19.835 mm |
| MV_UP | 19.922 mm |
| MV_DOWN | 19.844 mm |

Maximum off-axis component: **0.096 mm**.
[Calibration JSON](../rollouts/mujoco_robolab/physics_check/0916/task_0/00-57-54/calibration.json).

## Resources and provenance

Measured with `/usr/bin/time -v` on this machine:

* Three-episode batch: 16m 16s elapsed, 2% CPU, peak host RSS 1.53 GiB.
* Final single episode: 6m 49s elapsed, 3% CPU, peak host RSS 1.17 GiB.
* These are host-memory measurements, not GPU VRAM measurements. Model inference
  ran on Google's API; no local model weights were loaded.
* RoboLab source asset closure: about 27 MB; converted scene about 45 MB;
  Panda arm assets about 33 MB, all cached under `models/mujoco/`.

RoboLab revision: `ad45d4f974725d020f82c2b0d77d78533aeba2b3`.
Menagerie revision: `8161bba264d7fa7c99ca301e91e7fb44737676ad`.
Show-Harness source checkout: `137d571`.
MuJoCo version used: `3.13.0`.

See [setup and fidelity notes](mujoco.md), the generated
[source manifest](../models/mujoco/robolab/manifest.json), and
[conversion provenance](../models/mujoco/rubiks_cube_bowl/provenance.json).
