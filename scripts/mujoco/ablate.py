"""Run a frozen, interleaved one-factor camera ablation.

python scripts/mujoco/ablate.py --study rollouts/mujoco_ablations/<study> --prepare-only
python -u scripts/mujoco/ablate.py --study rollouts/mujoco_ablations/<study>

This is an experiment runner, not a task policy. All actions remain selected by
the existing Show-Harness pipeline; object-state diagnostics run after episodes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.record.episode_logger import _redact_sensitive_metadata
from core.sim.mujoco_ablation import (
    AblationInstrumentation, SIDE_CAMERA_DESCRIPTION, SIDE_GRASP_RULE_TEMPLATE, side_camera_spec,
)
from scripts.run_mujoco import parse_args as run_args, resolve_config, run_episode


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def runtime_manifest():
    files = []
    for base in ["core", "plugins", "interpreters", "prompts"]:
        files.extend(p for p in (ROOT / base).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".txt"))
    files.extend(ROOT / path for path in ["configs/robot_mujoco.yaml", "configs/robot_franka.yaml",
                 "configs/robot_robolab.yaml", "configs/primitives_franka.yaml", "scripts/run_mujoco.py"])
    files.extend((ROOT / "scripts/mujoco").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))}


def check_frozen(study):
    source = json.loads((study / "runtime_source_manifest.json").read_text())
    if runtime_manifest() != source:
        raise RuntimeError("Source/config changed after study freeze. Start a new study; do not mix versions.")
    assets = json.loads((study / "asset_manifest.json").read_text())
    for path, expected in assets.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen asset changed: {path}")


def prepare(study, repeats, factor="side-image"):
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    study.mkdir(parents=True, exist_ok=True)
    args = run_args([])
    cfg = resolve_config(args)
    cfg["observer_cameras"] = [side_camera_spec()]
    if not (study / "asset_manifest.json").exists():
        assets = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT / "models/mujoco").rglob("*")) if p.is_file()}
        write_json(study / "asset_manifest.json", assets)
        source = runtime_manifest()
        write_json(study / "baseline_source_manifest.json", source)
        for path in source:
            target = study / "baseline_source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, target)
    if factor not in ("side-image", "side-description", "side-grasp-check"):
        raise ValueError(f"Unknown ablation factor: {factor}")
    if factor == "side-image":
        names = ["baseline", "side"]
        conditions = {"baseline": "Front + wrist (2 images)",
                      "side": "Same request, with one fixed side image appended (3 images)"}
        treatment = "transmit_extra_side_image"
    elif factor == "side-description":
        names = ["side", "side_described"]
        conditions = {"side": "Front + wrist + side (3 images), original prompt",
                      "side_described": "Same three images and prompt, plus one camera-description paragraph"}
        treatment = "append_fixed_camera_description"
    else:
        names = ["side_described", "side_grasp_checked"]
        conditions = {"side_described": "Three images and camera description, original GRASP condition",
                      "side_grasp_checked": "Same inputs, replacing only the visual GRASP condition"}
        treatment = "replace_visual_grasp_condition"
    condition_parameters = {name: {
        "send_side": name != "baseline",
        "describe_side": name in ("side_described", "side_grasp_checked"),
        "check_side_grasp": name == "side_grasp_checked",
    } for name in names}
    order = []
    for pair in range(repeats):
        for condition in (names if pair % 2 == 0 else names[::-1]):
            order.append({"trial": len(order), "pair": pair, "condition": condition})
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": cfg["task"], "model": cfg["vlm"]["model"], "max_steps": cfg["max_steps"],
        "factor": factor,
        "conditions": conditions,
        "condition_parameters": condition_parameters,
        "sole_treatment": treatment,
        "unchanged": ["original prompt templates and text", "model and inference settings", "initial state",
                      "physics and assets", "actuators", "action step sizes", "policy plugins",
                      "existing two camera inputs", "task scoring", "step budget"],
        "common_instrumentation": "Both conditions render all three cameras. "
            "Every HTTP POST is logged. Raw simulation states are saved, but never passed to the policy.",
        "camera": side_camera_spec(),
        "order": order,
        "primary_metric": "Independent physical success; runtime errors count as failures",
        "secondary_metrics": ["empty/lost grasps", "grasp attempts", "steps", "API calls",
                              "post-run jaw-axis XY error at grasp", "wall time", "token usage"],
        "scope": f"{repeats} repetitions per condition are an exploratory screen, not a deployment reliability estimate. "
            "The initial scene is identical, but the hosted VLM is not seed-controlled.",
        "camera_label_policy": ("No prompt labels or camera-use instructions are added; this is an image-only treatment."
            if factor == "side-image" else "Only the side_described condition appends the camera_description below. "
            "No grasp rules, action increments, or other prompt text are changed."),
    }
    if factor != "side-image":
        protocol["camera_description"] = SIDE_CAMERA_DESCRIPTION
    if factor == "side-grasp-check":
        protocol["unchanged"][0] = "all prompt text except one GRASP condition; camera description unchanged"
        protocol["unchanged"][7] = "all three camera inputs"
        protocol["camera_label_policy"] = "Both conditions append the same camera description. " \
            "Only side_grasp_checked replaces the GRASP condition; direction rules remain original."
        protocol["grasp_condition_template"] = SIDE_GRASP_RULE_TEMPLATE
    write_json(study / "protocol.json", protocol)
    write_json(study / "resolved_config.json", _redact_sensitive_metadata(cfg))
    source = runtime_manifest()
    write_json(study / "runtime_source_manifest.json", source)
    for path in source:
        target = study / "runtime_source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    return protocol


def diagnose_grasps(run_dir):
    """Offline geometry diagnostics using saved states, after the policy has ended."""
    import mujoco
    import numpy as np

    path = run_dir / "physics_states.jsonl"
    if not path.exists():
        return []
    events = [json.loads(row) for row in path.read_text().splitlines()]
    model = mujoco.MjModel.from_xml_path(str(run_dir / "model_snapshot.xml"))
    data = mujoco.MjData(model)
    meta = json.loads((run_dir / "metadata.json").read_text())
    center_local = np.asarray(meta["provenance"]["objects"]["rubiks_cube"]["hull_centroid"])
    rows = []
    for event in events:
        if event["token"] != "GRASP":
            continue
        state = event["before"]
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        data.ctrl[:] = state["ctrl"]
        mujoco.mj_forward(model, data)
        cube = data.body("rubiks_cube")
        center = cube.xpos + cube.xmat.reshape(3, 3) @ center_local
        delta = data.site("eef").xpos[:2] - center[:2]
        rows.append({"action": event["action"], "request": event["request"],
                     "jaw_axis_minus_cube_xy_mm": (delta * 1000).tolist(),
                     "xy_error_mm": float(np.linalg.norm(delta) * 1000),
                     "empty": bool(event.get("grasp_empty"))})
    write_json(run_dir / "grasp_diagnostics.json", rows)
    return rows


def summarize_trial(run_dir, entry, error=None):
    evaluation_path = run_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    records_path = run_dir / "steps.jsonl"
    records = [json.loads(x) for x in records_path.read_text().splitlines()] if records_path.exists() else []
    diagnostics = diagnose_grasps(run_dir)
    requests = [json.loads(p.read_text()) for p in sorted((run_dir / "requests").glob("*/request.json"))]
    usage = [r["usage"] for r in requests if isinstance(r.get("usage"), dict)]
    return {**entry, "run_dir": str(run_dir), "success": bool(evaluation.get("success")) and error is None,
            "physical_success": evaluation.get("success"), "error": error,
            "model_declared_success": evaluation.get("model_declared_success"),
            "steps": len(records), "grasp_attempts": len(diagnostics),
            "empty_grasps": sum((r.get("recovery") or {}).get("event") == "empty_grasp" for r in records),
            "lost_grasps": sum((r.get("recovery") or {}).get("event") == "lost_grasp" for r in records),
            "grasp_xy_error_median_mm": statistics.median([r["xy_error_mm"] for r in diagnostics]) if diagnostics else None,
            "first_grasp_xy_error_mm": diagnostics[0]["xy_error_mm"] if diagnostics else None,
            "api_calls": len(requests), "input_tokens": sum(r.get("prompt_tokens", 0) for r in usage),
            "output_tokens": sum(r.get("completion_tokens", 0) for r in usage),
            "usage_records": len(usage),
            "image_counts": sorted(set(len(r["images"]) for r in requests))}


def write_summary(study, results):
    summary = {"trials": results, "conditions": {}}
    for name in dict.fromkeys(r["condition"] for r in results):
        rows = [r for r in results if r["condition"] == name]
        if not rows:
            continue
        summary["conditions"][name] = {
            "runs": len(rows), "successes": sum(r["success"] for r in rows),
            "median_steps": statistics.median(r["steps"] for r in rows),
            "total_grasp_attempts": sum(r["grasp_attempts"] for r in rows),
            "empty_grasps": sum(r["empty_grasps"] for r in rows),
            "lost_grasps": sum(r["lost_grasps"] for r in rows),
        }
    write_json(study / "results.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--factor", choices=["side-image", "side-description", "side-grasp-check"], default=None,
                        help="One factor per study; defaults to side-image for a new study.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-new-trials", type=int, help="Run a prefix; resume with the same protocol later.")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    study = args.study.resolve()
    protocol_path = study / "protocol.json"
    protocol = (json.loads(protocol_path.read_text()) if protocol_path.exists()
                else prepare(study, args.repeats, args.factor or "side-image"))
    if args.factor is not None and args.factor != protocol.get("factor", "side-image"):
        raise ValueError("Requested factor differs from the existing frozen protocol")
    check_frozen(study)
    if args.prepare_only:
        print(f"Frozen protocol: {protocol_path}", flush=True)
        return 0
    completed = sorted((study / "trials").glob("*.json")) if (study / "trials").exists() else []
    results = [json.loads(p.read_text()) for p in completed]
    finished = {r["trial"] for r in results}
    (study / "trials").mkdir(exist_ok=True)
    fresh = 0
    for entry in protocol["order"]:
        if entry["trial"] in finished:
            continue
        if args.max_new_trials is not None and fresh >= args.max_new_trials:
            break
        check_frozen(study)
        runner_args = run_args([])
        cfg = resolve_config(runner_args)
        cfg["observer_cameras"] = [protocol["camera"]]
        baseline = json.loads((study / "resolved_config.json").read_text())
        if _redact_sensitive_metadata(cfg) != baseline:
            raise RuntimeError("Resolved policy config differs from the frozen protocol")
        cfg = copy.deepcopy(cfg)
        cfg["log_dir"] = str(study / "runs" / f"{entry['trial']:02d}-{entry['condition']}")
        audit = AblationInstrumentation(**protocol["condition_parameters"][entry["condition"]])
        print(f"[study] Trial {entry['trial']+1}/{len(protocol['order'])}: {entry['condition']}", flush=True)
        error = None
        started = time.monotonic()
        try:
            result = run_episode(runner_args, cfg, entry["trial"], instrumentation=audit)
            if result.get("model_result", {}).get("end_reason") == "interrupted":
                print("[study] Interrupted; no further trials will run.", flush=True)
                return 130
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            error = str(exc).replace(str(cfg["vlm"]["api_key"]), "[REDACTED]")
            print(f"[study] Trial failed: {error}", flush=True)
            if audit.run_dir is None:
                raise
            (audit.run_dir / "error.txt").write_text(error + "\n" + traceback.format_exc().replace(
                str(cfg["vlm"]["api_key"]), "[REDACTED]"))
        summary = summarize_trial(audit.run_dir, entry, error)
        summary["wall_time_s"] = time.monotonic() - started
        write_json(study / "trials" / f"{entry['trial']:02d}.json", summary)
        results.append(summary)
        write_summary(study, results)
        print(f"[study] {entry['condition']}: success={summary['success']}, "
              f"steps={summary['steps']}, empty/lost={summary['empty_grasps']}/{summary['lost_grasps']}", flush=True)
        fresh += 1
    print(f"[study] Results: {study / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
