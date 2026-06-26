"""Task 3: Oracle Coarse Program Evaluation — 20ep 2-block, 20ep 3-block, invalid rejection.

Output: generated/results/upper_ns/coarse_oracle_eval.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.skills.atomic_pick_skill import AtomicPickSkill
from vlm_nsm.embodied.skills.atomic_place_skill import AtomicPlaceSkill
from vlm_nsm.embodied.skills.base_skill import SkillParams
from vlm_nsm.language.coarse_dsl import CoarseTaskProgram
from vlm_nsm.neuro_symbolic.coarse_planner_bridge import CoarsePlannerBridge
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def _check_goal(env, goal_preds):
    try:
        poses = env.get_all_object_poses()
        targets = dict(poses)
        targets["table"] = env.get_object_pose("table")
        physical = physical_state_to_predicates(poses, targets,
            {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05})
        return goal_satisfied(physical, goal_preds, mode="symbolic")
    except Exception:
        return False


def verify_coarse_program(steps, objects):
    from vlm_nsm.neuro_symbolic.verifier_bank import CoarseOptionVerifier
    valid_objs = set(objects)
    static_ok, static_errors = CoarseOptionVerifier.check_static(steps, valid_objs)
    verifier = CoarseOptionVerifier(valid_objects=objects)
    report = verifier.check(coarse_steps=steps)
    passed = static_ok and report.passed
    failures = list(static_errors)
    for check, reason in report.failures:
        failures.append(f"{check}: {reason}")
    # CoarseTaskProgram
    try:
        program = CoarseTaskProgram.from_json({"steps": steps})
        if program is None:
            passed = False
            failures.append("coarse_dsl: from_json returned None")
    except Exception as e:
        passed = False
        failures.append(f"coarse_dsl: {e}")
    n_forbidden = sum(1 for f in failures if "forbidden" in f.lower() or "phase-level" in f.lower())
    return {"passed": passed, "failures": failures, "n_forbidden": n_forbidden}


def run_one_episode(env, pid, coarse_steps, objects):
    bridge = CoarsePlannerBridge()
    sequence = bridge.execution_sequence(coarse_steps, objects)
    if not sequence:
        return {"status": "empty", "success": False, "failure_type": "empty_sequence",
                "n_pick": 0, "n_place": 0, "n_ok_pick": 0, "n_ok_place": 0,
                "n_total_steps": 0, "trace": []}

    pick_skill = AtomicPickSkill()
    place_skill = AtomicPlaceSkill()
    trace, n_pick, n_place, n_ok_pick, n_ok_place, total_steps = [], 0, 0, 0, 0, 0
    failed, final_failure = False, None

    for i, exec_step in enumerate(sequence):
        skill_name = exec_step["skill"]
        obj = exec_step.get("object", "")
        tgt = exec_step.get("target", "")

        if skill_name == "atomic_pick":
            n_pick += 1
            params = SkillParams(target_object=obj)
            result = pick_skill.execute_with_pid(env, pid, params)
            total_steps += result.steps
            entry = {"step_idx": i, "skill": "atomic_pick", "object": obj,
                     "success": result.success, "steps": result.steps,
                     "failure_type": result.failure_type,
                     "diagnostics": _make_json_safe(result.diagnostics)}
            if result.success:
                n_ok_pick += 1
            else:
                failed = True; final_failure = result.failure_type

        elif skill_name == "atomic_place":
            n_place += 1
            params = SkillParams(target_object=obj, target_location=tgt)
            result = place_skill.execute_with_pid(env, pid, params)
            total_steps += result.steps
            entry = {"step_idx": i, "skill": "atomic_place", "object": obj,
                     "target": tgt, "success": result.success,
                     "steps": result.steps, "failure_type": result.failure_type,
                     "diagnostics": _make_json_safe(result.diagnostics)}
            if result.success:
                n_ok_place += 1
            else:
                failed = True; final_failure = result.failure_type
        else:
            entry = {"step_idx": i, "skill": skill_name, "success": False,
                     "failure_type": f"unknown_skill_{skill_name}"}
            failed = True; final_failure = f"unknown_skill_{skill_name}"

        trace.append(entry)
        if failed:
            break

    # Goal check
    goal_preds = []
    try:
        program = CoarseTaskProgram.from_json({"steps": coarse_steps})
        if program:
            goal_preds = program.to_goal_predicates()
    except Exception:
        pass
    final_ok = _check_goal(env, goal_preds) if goal_preds else True
    success = (not failed) and final_ok
    if failed and not final_failure:
        final_failure = "goal_check_failed"

    return _make_json_safe({
        "status": "success" if success else "failed",
        "success": success, "failure_type": final_failure,
        "trace": trace, "n_pick": n_pick, "n_place": n_place,
        "n_ok_pick": n_ok_pick, "n_ok_place": n_ok_place,
        "n_total_steps": total_steps,
        "goal_predicates_expected": goal_preds,
        "final_goal_ok": final_ok,
    })


def main():
    out_dir = Path("generated/results/upper_ns")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── Part 1: Invalid program rejection ──
    print("=" * 60)
    print("PART 1: Invalid Program Rejection Tests")
    print("=" * 60)

    invalid_programs = {
        "self_place": [
            {"op": "pick", "object": "cube1"},
            {"op": "place", "object": "cube1", "target": "cube1"},
        ],
        "reach_forbidden": [
            {"op": "reach", "object": "cube1"},
            {"op": "grasp", "object": "cube1"},
        ],
        "double_pick": [
            {"op": "pick", "object": "cube1"},
            {"op": "pick", "object": "cube2"},
        ],
        "place_occupied": [
            {"op": "pick", "object": "cube1"},
            {"op": "place", "object": "cube1", "target": "cube2"},
            {"op": "pick", "object": "cube3"},
            {"op": "place", "object": "cube3", "target": "cube2"},
        ],
        "unknown_object": [
            {"op": "pick", "object": "cube99"},
        ],
        "place_without_pick": [
            {"op": "place", "object": "cube1", "target": "cube2"},
        ],
        "cyclic_stack": [
            {"op": "pick", "object": "cube1"},
            {"op": "place", "object": "cube1", "target": "cube2"},
            {"op": "pick", "object": "cube2"},
            {"op": "place", "object": "cube2", "target": "cube1"},
        ],
    }

    rejection_results = {}
    for name, steps in invalid_programs.items():
        objects = ["cube1", "cube2", "cube3", "table"]
        result = verify_coarse_program(steps, objects)
        status = "REJECTED" if not result["passed"] else "PASSED (BUG)"
        rejection_results[name] = result
        print(f"  {name}: {status} failures={len(result['failures'])}")
        for f in result["failures"][:3]:
            print(f"    {f}")

    n_rejected = sum(1 for r in rejection_results.values() if not r["passed"])
    print(f"  Rejection rate: {n_rejected}/{len(invalid_programs)} ({n_rejected/len(invalid_programs):.0%})")

    # ── Part 2: 20ep oracle 2-block ──
    print("\n" + "=" * 60)
    print("PART 2: Oracle 2-Block (20 episodes)")
    print("=" * 60)

    oracle_2block_steps = [
        {"op": "pick", "object": "cube1"},
        {"op": "place", "object": "cube1", "target": "cube2"},
    ]
    objects_2b = ["cube1", "cube2", "table"]

    verif = verify_coarse_program(oracle_2block_steps, objects_2b)
    print(f"  Verification: passed={verif['passed']}")

    env_2b = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                                  cube_names=["cube1", "cube2"], render_mode="none")
    env_2b.env.horizon = 5000

    episodes_2b = []
    for ep in range(20):
        np.random.seed(ep)
        pid = OperationalSpacePID()
        env_2b.reset()
        pid.reset()
        result = run_one_episode(env_2b, pid, oracle_2block_steps, objects_2b)
        result["episode"] = ep; result["seed"] = ep
        episodes_2b.append(result)
        status = "✅" if result["success"] else "❌"
        print(f"  [ep {ep:02d}] {status} pick={result['n_ok_pick']}/{result['n_pick']} "
              f"place={result['n_ok_place']}/{result['n_place']} steps={result['n_total_steps']} "
              f"failure={result.get('failure_type', 'none')}")

    n_ok_2b = sum(1 for e in episodes_2b if e["success"])
    print(f"  2-block oracle: {n_ok_2b}/20 ({n_ok_2b/20:.0%})")

    # ── Part 3: 20ep oracle 3-block ──
    print("\n" + "=" * 60)
    print("PART 3: Oracle 3-Block (20 episodes)")
    print("=" * 60)

    oracle_3block_steps = [
        {"op": "pick", "object": "cube2"},
        {"op": "place", "object": "cube2", "target": "cube3"},
        {"op": "pick", "object": "cube1"},
        {"op": "place", "object": "cube1", "target": "cube2"},
    ]
    objects_3b = ["cube1", "cube2", "cube3", "table"]

    verif_3b = verify_coarse_program(oracle_3block_steps, objects_3b)
    print(f"  Verification: passed={verif_3b['passed']}")

    import vlm_nsm.executors.stack_three_cube_env as _  # noqa
    env_3b = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                                  cube_names=["cube1", "cube2", "cube3"], render_mode="none")
    env_3b.env.horizon = 10000

    episodes_3b = []
    for ep in range(20):
        np.random.seed(ep)
        pid = OperationalSpacePID()
        env_3b.reset()
        pid.reset()
        result = run_one_episode(env_3b, pid, oracle_3block_steps, objects_3b)
        result["episode"] = ep; result["seed"] = ep
        episodes_3b.append(result)
        status = "✅" if result["success"] else "❌"
        print(f"  [ep {ep:02d}] {status} pick={result['n_ok_pick']}/{result['n_pick']} "
              f"place={result['n_ok_place']}/{result['n_place']} steps={result['n_total_steps']} "
              f"failure={result.get('failure_type', 'none')}")

    n_ok_3b = sum(1 for e in episodes_3b if e["success"])
    print(f"  3-block oracle: {n_ok_3b}/20 ({n_ok_3b/20:.0%})")

    # ── Save ──
    all_results["invalid_rejection"] = rejection_results
    all_results["oracle_2block"] = {
        "steps": oracle_2block_steps, "n_episodes": 20,
        "n_success": n_ok_2b, "success_rate": n_ok_2b / 20,
        "verification": verif, "episodes": episodes_2b,
    }
    all_results["oracle_3block"] = {
        "steps": oracle_3block_steps, "n_episodes": 20,
        "n_success": n_ok_3b, "success_rate": n_ok_3b / 20,
        "verification": verif_3b, "episodes": episodes_3b,
    }

    out_path = out_dir / "coarse_oracle_eval.json"
    out_path.write_text(json.dumps(_make_json_safe(all_results), indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Invalid rejection: {n_rejected}/{len(invalid_programs)}")
    print(f"  2-block oracle:    {n_ok_2b}/20 ({n_ok_2b/20:.0%})")
    print(f"  3-block oracle:    {n_ok_3b}/20 ({n_ok_3b/20:.0%})")

    return n_ok_2b, n_ok_3b


if __name__ == "__main__":
    main()
