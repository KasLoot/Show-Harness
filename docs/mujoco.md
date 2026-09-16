# MuJoCo: original RoboLab scene, original Show-Harness VLM loop

This backend converts **RoboLab's `RubiksCubeTask`** ("Put the cube in the bowl")
to MuJoCo and runs it through Show-Harness's existing zero-shot planner,
controller, plugins, and Franka action interpreter. The default model is
`gemini-3.8-flash` through Google's API. Isaac Sim, torch, and a local VLM server
are not required.

See [the validation report](mujoco_validation.md) for measured results, including
the failed final trial. Successful demonstrations do not establish deployment reliability.
The [controlled ablation report](mujoco_ablation.md) contains 18 episodes testing
camera count, camera descriptions, and the visual grasp condition one factor at
a time. None of those changes established a reliable grasp-centering fix.

## Setup and run

```bash
bash scripts/setup.sh mujoco

# Render both cameras and measure action distances; no API calls.
.venv/bin/python scripts/run_mujoco.py --no-vlm --probe-axes

# Uses GEMINI_API_KEY from your environment or configs/secrets.env.
.venv/bin/python -u scripts/run_mujoco.py --gui

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
CLI overrides: `--model`, `--max-steps`, `--episodes`, `--log-dir`, `--robot-config`.
Only `RubiksCubeTask` is ported at present.

## What comes from the original sources

| Component | Source |
| --- | --- |
| Instruction, cube/bowl/table layout, object masses, textures and meshes | [RoboLab task](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/robolab/tasks/benchmark/rubiks_cube_task.py) and `assets/scenes/rubiks_cube_bowl.usda` at that revision |
| Panda arm MJCF | MuJoCo Menagerie `franka_emika_panda`, revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`, derived from Franka's URDF |
| Yellow fingertips, brackets and their collision hull | This repo's `assets/panda_short_finger.stl` and `make_short_finger_asset.py` geometry/transforms |
| Robot home pose, camera extrinsics/intrinsics, wrist mount, actuator gains | `core/sim/robolab_franka.py` |
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
adaptive steps, action chunking, smoothing and recovery remain active. For
simulation, controller delays advance physics instead of sleeping in real time.
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

This preserves authored geometry and policy behavior, not bit-identical PhysX
dynamics or RTX rendering. MuJoCo contacts use tuned solver parameters; the bowl
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
