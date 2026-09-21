#!/usr/bin/env python3
"""Replay logged MuJoCo actions with smooth physics, without a VLM or hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.sim.mujoco_replay import execute_replay, load_replay_plan, validate_tolerances


def hold_final_view(session) -> None:
    """Keep the passive viewer responsive without advancing simulated time."""
    viewer = session._viewer
    while viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(1 / 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="MuJoCo run containing metadata.json and steps.jsonl")
    parser.add_argument("--gui", action="store_true", help="Show smooth simulation; on macOS use .venv/bin/mjpython")
    parser.add_argument("--record", action="store_true", help="Write synchronized smooth camera videos in a NEW output directory")
    parser.add_argument("--output-dir", type=Path, help="New directory for replay report/videos; default is a timestamped sibling of the source run")
    parser.add_argument("--verify", action="store_true", help="Abort if measured pre/post TCP poses differ from the log (does not verify full object state)")
    parser.add_argument("--position-tolerance-m", type=float, default=0.0001)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--hold-final", action="store_true", help="Keep --gui open at the final state without advancing physics")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and actions without loading MuJoCo, creating output, or executing physics")
    args = parser.parse_args(argv)
    if args.hold_final and not args.gui:
        parser.error("--hold-final requires --gui")
    try:
        validate_tolerances(args.position_tolerance_m, args.rotation_tolerance_deg)
        plan = load_replay_plan(args.run_dir, require_poses=args.verify)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"[replay] {len(plan.records)} recorded actions; scene={plan.config.get('scene', 'pick_place')}")
    print("[replay] Action replay through current MuJoCo physics; the run has no full simulator-state trajectory.")
    if args.dry_run:
        print("[replay] Dry-run passed; no simulation or output files created.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or plan.run_dir.with_name(f"{plan.run_dir.name}_replay_{stamp}")).resolve()
    if output == plan.run_dir or plan.run_dir in output.parents:
        parser.error("--output-dir must be separate from the original run")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output-dir already exists; choose a new directory to preserve existing files")
    print(f"[replay] Output: {output}")
    # Lazy imports keep --help/--dry-run independent of MuJoCo and its renderer.
    # make_controller only constructs local motion control; run_mujoco.main,
    # secrets loading, VLM construction, and robot hardware are never invoked.
    from core.record.mujoco_recorder import MujocoRecorder
    from core.sim.mujoco_session import MujocoSession
    from scripts.run_mujoco import make_controller

    session = None
    report = {"source_run": str(plan.run_dir), "status": "initializing"}
    exit_code = 0
    try:
        session = MujocoSession(plan.config, gui=args.gui)
        controller = make_controller(session, plan.config,
                                     variable_step=plan.variable_step)
        if args.record:
            session.recorder = MujocoRecorder(session, output / "videos", model="offline action replay",
                                              task=str(plan.config.get("task", "")),
                                              **plan.config.get("recording", {}))
        execute_replay(plan, session, controller, verify=args.verify,
                       position_tolerance_m=args.position_tolerance_m,
                       rotation_tolerance_deg=args.rotation_tolerance_deg, report=report)
        if session.recorder is not None:
            session.recorder.finish(report["task_success"], "action_replay_complete")
            report["video_path"] = str(session.recorder.paths["combined"])
        print(f"[replay] Completed {report['actions_replayed']} actions. "
              f"Physical task success: {report['task_success']}")
        if report["comparisons"]:
            print(f"[replay] Largest TCP error: {report['max_position_error_m'] * 1000:.4f} mm, "
                  f"{report['max_rotation_error_deg']:.4f} degrees.")
        else:
            print("[replay] No recorded TCP poses were available for comparison.")
        if args.hold_final:
            print("[replay] Final state held; close the viewer or press Ctrl+C to exit.")
            hold_final_view(session)
    except KeyboardInterrupt:
        if report.get("status") != "completed":
            report["status"] = "interrupted"
            exit_code = 130
        print("[replay] Viewer closed or replay interrupted.")
    except Exception as exc:
        if report.get("status") in ("initializing", "running"):
            report["status"] = "error"
        report["error"] = str(exc)
        print(f"[replay] {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if session is not None:
            if session.recorder is not None:
                report["video_path"] = str(session.recorder.paths["combined"])
            session.close()
        (output / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
