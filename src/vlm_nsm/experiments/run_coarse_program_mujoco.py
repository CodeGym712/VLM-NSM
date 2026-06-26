"""Coarse Program MuJoCo Runner — execute CoarseTaskProgram through atomic skills.

Flow:
  CoarseTaskProgram (JSON)
  → CoarseOptionVerifier (precondition/cyclic checks)
  → CoarsePlannerBridge.execution_sequence() (→ atomic_pick/atomic_place)
  → AtomicPickSkill / AtomicPlaceSkill (robosuite PID execution)
  → OptionResult trace
  → final goal check
  → summary.json

Usage:
  python -m vlm_nsm.experiments.run_coarse_program_mujoco \\
      --program path/to/coarse_program.json \\
      --task stacking_2block --seed 0 --source oracle \\
      --output generated/results/upper_ns/coarse_program_mujoco/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.skills.atomic_pick_skill import AtomicPickSkill
from vlm_nsm.embodied.skills.atomic_place_skill import AtomicPlaceSkill
from vlm_nsm.embodied.skills.base_skill import SkillParams
from vlm_nsm.language.coarse_dsl import CoarseTaskProgram
from vlm_nsm.neuro_symbolic.verifier_bank import (
    CoarseOptionVerifier, standard_verifier_bank, VerificationReport,
)
from vlm_nsm.neuro_symbolic.coarse_planner_bridge import CoarsePlannerBridge
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied


# ── JSON Serialization Helpers ─────────────────────────────────────────────────

def _make_json_safe(obj):
    """Recursively convert non-JSON-serializable types (sets, ndarrays) to lists."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    return obj


# ── Goal Check ────────────────────────────────────────────────────────────────

def _check_goal(env, goal_preds: list[list[str]]) -> dict:
    """Check goal predicates against current physical state."""
    try:
        poses = env.get_all_object_poses()
        targets = dict(poses)
        targets["table"] = env.get_object_pose("table")
        physical = physical_state_to_predicates(
            poses, targets,
            {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05})
        ok = goal_satisfied(physical, goal_preds, mode="symbolic")
        return {"goal_ok": ok, "physical_predicates": physical}
    except Exception as e:
        return {"goal_ok": False, "error": str(e)}


# ── Pre-execution Verification ────────────────────────────────────────────────

def verify_coarse_program(steps: list, objects: list[str]) -> dict:
    """Verify a coarse program before execution.

    Returns:
        {"passed": bool, "failures": [...], "warnings": [...], "n_forbidden": int}
    """
    result = {
        "passed": True,
        "failures": [],
        "warnings": [],
        "n_forbidden": 0,
    }

    # 1. Static structural check
    valid_objects = set(objects)
    from vlm_nsm.neuro_symbolic.verifier_bank import CoarseOptionVerifier
    static_ok, static_errors = CoarseOptionVerifier.check_static(
        steps, valid_objects)
    if not static_ok:
        result["passed"] = False
        result["failures"].extend(static_errors)

    # 2. Dynamic state-tracking check
    verifier = CoarseOptionVerifier(valid_objects=objects)
    report = verifier.check(coarse_steps=steps)
    result["passed"] = result["passed"] and report.passed
    for check, reason in report.failures:
        result["failures"].append(f"{check}: {reason}")
        # Count forbidden ops
        if "forbidden" in check:
            result["n_forbidden"] += 1
    for check, reason in report.warnings:
        result["warnings"].append(f"{check}: {reason}")

    # 3. CoarseTaskProgram structural validation
    try:
        program = CoarseTaskProgram.from_json({"steps": steps})
        if program is None:
            result["passed"] = False
            result["failures"].append("coarse_dsl: from_json returned None (structural invalid)")
    except Exception as e:
        result["passed"] = False
        result["failures"].append(f"coarse_dsl: {e}")

    return result


# ── Main Execution ────────────────────────────────────────────────────────────

def run_coarse_program(env, pid, coarse_steps: list,
                       objects: list[str]) -> dict:
    """Execute a coarse program through atomic skills.

    Args:
        env: RobosuiteEnvWrapper.
        pid: OperationalSpacePID controller.
        coarse_steps: List of coarse step dicts.
        objects: Valid object IDs.

    Returns:
        Trace dict with per-step results, final check, and summary stats.
    """
    bridge = CoarsePlannerBridge()
    sequence = bridge.execution_sequence(coarse_steps, objects)

    if not sequence:
        return {
            "status": "empty_sequence",
            "success": False,
            "failure_type": "empty_sequence",
            "trace": [],
            "n_pick": 0, "n_place": 0, "n_total_steps": 0,
        }

    # Skills
    pick_skill = AtomicPickSkill()
    place_skill = AtomicPlaceSkill()

    trace: list[dict] = []
    n_pick = 0
    n_place = 0
    n_ok_pick = 0
    n_ok_place = 0
    total_steps = 0
    failed = False
    final_failure = None

    for i, exec_step in enumerate(sequence):
        skill_name = exec_step["skill"]
        obj = exec_step.get("object", "")
        tgt = exec_step.get("target", "")

        if skill_name == "atomic_pick":
            n_pick += 1
            params = SkillParams(target_object=obj)
            result = pick_skill.execute_with_pid(env, pid, params)
            total_steps += result.steps
            entry = {
                "step_idx": i, "skill": "atomic_pick",
                "object": obj, "success": result.success,
                "steps": result.steps, "failure_type": result.failure_type,
                "diagnostics": result.diagnostics,
            }
            if result.success:
                n_ok_pick += 1
            else:
                failed = True
                final_failure = result.failure_type

        elif skill_name == "atomic_place":
            n_place += 1
            params = SkillParams(target_object=obj, target_location=tgt)
            result = place_skill.execute_with_pid(env, pid, params)
            total_steps += result.steps
            entry = {
                "step_idx": i, "skill": "atomic_place",
                "object": obj, "target": tgt,
                "success": result.success,
                "steps": result.steps, "failure_type": result.failure_type,
                "diagnostics": result.diagnostics,
            }
            if result.success:
                n_ok_place += 1
            else:
                failed = True
                final_failure = result.failure_type

        else:
            entry = {
                "step_idx": i, "skill": skill_name,
                "success": False,
                "failure_type": f"unknown_skill_{skill_name}",
            }
            failed = True
            final_failure = f"unknown_skill_{skill_name}"

        trace.append(entry)
        if failed:
            break

    # Final goal check
    goal_preds = []
    try:
        program = CoarseTaskProgram.from_json({"steps": coarse_steps})
        if program:
            goal_preds = program.to_goal_predicates()
    except Exception:
        pass

    final_check = _check_goal(env, goal_preds) if goal_preds else {"goal_ok": True}

    # Build summary
    success = (not failed) and final_check.get("goal_ok", False)
    if failed and not final_failure:
        final_failure = "goal_check_failed"

    return {
        "status": "success" if success else "failed",
        "success": success,
        "failure_type": final_failure,
        "trace": trace,
        "n_pick": n_pick,
        "n_place": n_place,
        "n_ok_pick": n_ok_pick,
        "n_ok_place": n_ok_place,
        "n_total_steps": total_steps,
        "goal_predicates_expected": goal_preds,
        "final_check": final_check,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Execute CoarseTaskProgram through atomic skills in robosuite")
    ap.add_argument("--program", type=str, required=True,
                    help="Path to coarse program JSON or JSON string")
    ap.add_argument("--task", type=str, default="stacking_2block",
                    help="Task name: stacking_2block | stacking_3block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", type=str, default="oracle",
                    help="Program source: oracle | qwen")
    ap.add_argument("--output", type=str,
                    default="generated/results/upper_ns/coarse_program_mujoco/")
    ap.add_argument("--episodes", type=int, default=1,
                    help="Number of episodes (repeats with different seeds)")
    ap.add_argument("--reject-invalid", action="store_true", default=True,
                    help="Reject invalid programs before execution")
    args = ap.parse_args()

    # Load program
    program_path = Path(args.program)
    if program_path.exists():
        coarse_data = json.loads(program_path.read_text())
    else:
        coarse_data = json.loads(args.program)

    coarse_steps = coarse_data.get("steps", [])
    if not coarse_steps:
        print("ERROR: empty steps in program")
        sys.exit(1)

    # Setup objects
    cubes = ["cube1", "cube2"] if "2block" in args.task else ["cube1", "cube2", "cube3"]
    all_objects = cubes + ["table"]

    if len(cubes) >= 3:
        import vlm_nsm.executors.stack_three_cube_env as _  # noqa

    # Pre-execution verification
    verif_result = verify_coarse_program(coarse_steps, all_objects)
    print(f"Verification: passed={verif_result['passed']} "
          f"failures={len(verif_result['failures'])} "
          f"forbidden={verif_result['n_forbidden']}")
    for f in verif_result["failures"][:10]:
        print(f"  FAIL: {f}")

    if args.reject_invalid and not verif_result["passed"]:
        # Save rejection result
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        rejection = _make_json_safe({
            "task": args.task,
            "source": args.source,
            "status": "rejected",
            "verification": verif_result,
            "coarse_steps": coarse_steps,
        })
        out_path = out_dir / f"rejected_{args.source}_seed{args.seed}.json"
        out_path.write_text(json.dumps(rejection, indent=2, ensure_ascii=False))
        print(f"Program REJECTED by verifier. Saved to {out_path}")
        sys.exit(0)

    # Setup env
    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda",
        cube_names=cubes, render_mode="none")
    env.env.horizon = 10000 if len(cubes) >= 3 else 5000

    # Run episodes
    all_episodes = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        np.random.seed(ep_seed)

        pid = OperationalSpacePID()
        env.reset()
        pid.reset()

        trace = run_coarse_program(env, pid, coarse_steps, all_objects)
        trace["episode"] = ep
        trace["seed"] = ep_seed
        trace["source"] = args.source
        trace["task"] = args.task
        trace["verification"] = verif_result

        all_episodes.append(trace)

        status = "✅" if trace["success"] else "❌"
        print(f"[ep {ep} seed={ep_seed}] {status} "
              f"pick={trace['n_ok_pick']}/{trace['n_pick']} "
              f"place={trace['n_ok_place']}/{trace['n_place']} "
              f"steps={trace['n_total_steps']} "
              f"failure={trace.get('failure_type', 'none')}")

    # Save summary
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_success = sum(1 for ep in all_episodes if ep["success"])
    summary = _make_json_safe({
        "task": args.task,
        "source": args.source,
        "n_episodes": args.episodes,
        "n_success": n_success,
        "success_rate": n_success / args.episodes if args.episodes else 0,
        "verification": verif_result,
        "coarse_steps": coarse_steps,
        "episodes": all_episodes,
    })

    out_path = out_dir / f"summary_{args.source}_{args.task}_seed{args.seed}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary: {n_success}/{args.episodes} ({n_success/args.episodes:.0%}) "
          f"saved to {out_path}")

    # Aggregate if multiple episodes
    if args.episodes > 1:
        agg = {
            "task": args.task,
            "source": args.source,
            "n_episodes": args.episodes,
            "n_success": n_success,
            "success_rate": n_success / args.episodes,
            "failure_types": {},
            "mean_steps": sum(ep["n_total_steps"] for ep in all_episodes) / args.episodes,
            "mean_pick_steps": 0,
            "mean_place_steps": 0,
        }
        for ep in all_episodes:
            ft = ep.get("failure_type", "none")
            agg["failure_types"][ft] = agg["failure_types"].get(ft, 0) + 1
            for t in ep.get("trace", []):
                if t.get("skill") == "atomic_pick":
                    agg["mean_pick_steps"] += t.get("steps", 0)
                elif t.get("skill") == "atomic_place":
                    agg["mean_place_steps"] += t.get("steps", 0)
        agg["mean_pick_steps"] /= max(1, args.episodes * sum(
            1 for ep in all_episodes
            for t in ep.get("trace", []) if t.get("skill") == "atomic_pick"))
        agg["mean_place_steps"] /= max(1, args.episodes * sum(
            1 for ep in all_episodes
            for t in ep.get("trace", []) if t.get("skill") == "atomic_place"))

        agg_path = out_dir / f"aggregate_{args.source}_{args.task}.json"
        agg_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        print(f"Aggregate saved to {agg_path}")


if __name__ == "__main__":
    main()
