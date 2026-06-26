"""Task 5: Qwen Embodied Eval — generate coarse programs from Qwen + verify + execute.

Focus: 2-block embodied (3-block execution broken at atomic skill layer — documented).

Output: generated/results/upper_ns/qwen_coarse_embodied_eval.json
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
import numpy as np

from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser
from vlm_nsm.neuro_symbolic.verifier_bank import CoarseOptionVerifier
from vlm_nsm.neuro_symbolic.coarse_reorder import bfs_reorder_coarse

API_KEY = os.environ.get("QWEN_API_KEY", "")
if not API_KEY:
    sys.exit("Error: QWEN_API_KEY environment variable is not set. "
             "Obtain a key from Alibaba Cloud Model Studio and try again.")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── Test instructions ──
QNUM_TESTS = [
    # 2-block (embodied executable)
    {"id": "Q2_01", "instruction": "Stack cube1 on cube2", "n_blocks": 2,
     "objects": ["cube1", "cube2", "table"], "embodied": True},
    {"id": "Q2_02", "instruction": "Pick up cube1 and put it on cube2", "n_blocks": 2,
     "objects": ["cube1", "cube2", "table"], "embodied": True},
    {"id": "Q2_03", "instruction": "Place cube2 on top of cube1", "n_blocks": 2,
     "objects": ["cube1", "cube2", "table"], "embodied": True},
    {"id": "Q2_04", "instruction": "Move cube1 onto cube2", "n_blocks": 2,
     "objects": ["cube1", "cube2", "table"], "embodied": True},
    {"id": "Q2_05", "instruction": "Unstack cube1 and put it on the table", "n_blocks": 2,
     "objects": ["cube1", "cube2", "table"], "embodied": True},
    # 3-block (verification + reorder only, embodied marked as known-failure)
    {"id": "Q3_01", "instruction": "Stack cube1 on cube2, then cube2 on cube3", "n_blocks": 3,
     "objects": ["cube1", "cube2", "cube3", "table"], "embodied": False},
    {"id": "Q3_02", "instruction": "Make a 3-block tower: cube3 base, cube2 middle, cube1 top", "n_blocks": 3,
     "objects": ["cube1", "cube2", "cube3", "table"], "embodied": False},
    {"id": "Q3_03", "instruction": "Stack all cubes with cube1 on top", "n_blocks": 3,
     "objects": ["cube1", "cube2", "cube3", "table"], "embodied": False},
    {"id": "Q3_04", "instruction": "Put cube2 on cube3, then put cube1 on cube2", "n_blocks": 3,
     "objects": ["cube1", "cube2", "cube3", "table"], "embodied": False},
    {"id": "Q3_05", "instruction": "Arrange: cube3 on table, cube1 on cube3, cube2 on cube1", "n_blocks": 3,
     "objects": ["cube1", "cube2", "cube3", "table"], "embodied": False},
]


def main():
    parser = OnlineQwenGoalParser(
        model="qwen-plus", api_key=API_KEY, api_base=API_BASE,
        temperature=0.0, max_tokens=512, retries=1)

    all_objects = ["cube1", "cube2", "cube3", "table"]
    verifier = CoarseOptionVerifier(valid_objects=all_objects)

    results = []

    for test in QNUM_TESTS:
        tid = test["id"]
        instruction = test["instruction"]
        objects = test["objects"]
        n_blocks = test["n_blocks"]
        do_embodied = test["embodied"]

        entry = {
            "id": tid, "instruction": instruction,
            "n_blocks": n_blocks, "objects": objects,
        }

        print(f"[{tid}] \"{instruction}\"")
        t0 = time.perf_counter()

        # Step 1: Qwen API call
        try:
            pr = parser.parse(instruction, objects=objects,
                             task_type="coarse_stacking", n_blocks=n_blocks)
            entry["parse_success"] = pr.success
            entry["raw_output"] = pr.raw_output
            entry["latency_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            entry["parse_success"] = False
            entry["parse_error"] = str(e)
            entry["latency_s"] = round(time.perf_counter() - t0, 2)
            results.append(entry)
            print(f"  Parse: ❌ {e}")
            continue

        if not pr.success or not pr.goal_json:
            entry["parse_error"] = pr.error or "no goal_json"
            results.append(entry)
            print(f"  Parse: ❌")
            continue

        steps = pr.goal_json.get("steps", [])
        entry["qwen_steps"] = steps
        entry["n_steps"] = len(steps)

        # Step 2: Verification
        report = verifier.check(coarse_steps=steps)
        entry["verification_passed"] = report.passed
        entry["verification_failures"] = [(c, r) for c, r in report.failures]
        entry["verification_warnings"] = [(c, r) for c, r in report.warnings]

        # Check for forbidden ops
        forbidden_ops = {"reach", "grasp", "lift", "reach_place", "release_place", "retreat"}
        has_forbidden = any(
            isinstance(s, dict) and s.get("op", "") in forbidden_ops
            for s in steps)
        entry["has_forbidden_ops"] = has_forbidden

        print(f"  Verify: {'✅' if report.passed else '❌'} "
              f"failures={len(report.failures)} forbidden={has_forbidden}")

        # Step 3: BFS reorder if rejected
        if not report.passed:
            reorder = bfs_reorder_coarse(
                steps, objects,
                rejection_reasons=[f"{c}: {r}" for c, r in report.failures])
            entry["bfs_reorder"] = {
                "triggered": True,
                "success": reorder.success,
                "repaired_steps": reorder.repaired_steps,
                "bfs_plan": reorder.bfs_plan,
                "repaired_verification_passed": reorder.verification_passed,
                "error": reorder.error,
            }
            print(f"  Reorder: {'✅' if reorder.success else '❌'} "
                  f"{reorder.error or ''}")
        else:
            entry["bfs_reorder"] = {"triggered": False}

        # Step 4: Embodied execution (2-block only)
        if do_embodied and (report.passed or entry.get("bfs_reorder", {}).get("success")):
            exec_steps = entry.get("bfs_reorder", {}).get("repaired_steps") or steps
            embodied_results = _run_embodied(exec_steps, objects, tid, n_episodes=5)
            entry["embodied"] = embodied_results
            print(f"  Embodied: {embodied_results['n_success']}/{embodied_results['n_episodes']}")
        else:
            entry["embodied"] = None

        results.append(entry)

    # Save
    out_dir = Path("generated/results/upper_ns")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary stats
    n_parse_ok = sum(1 for r in results if r.get("parse_success"))
    n_verify_ok = sum(1 for r in results if r.get("verification_passed"))
    n_forbidden = sum(1 for r in results if r.get("has_forbidden_ops"))
    n_reorder = sum(1 for r in results
                    if r.get("bfs_reorder", {}).get("triggered")
                    and r.get("bfs_reorder", {}).get("success"))
    n_embodied = sum(1 for r in results if r.get("embodied"))
    n_embodied_ok = sum(1 for r in results
                        if r.get("embodied") and r["embodied"].get("n_success", 0) > 0)

    summary = {
        "n_tests": len(results),
        "n_parse_ok": n_parse_ok,
        "n_verify_ok": n_verify_ok,
        "n_forbidden_ops": n_forbidden,
        "n_reorder_triggered": sum(1 for r in results if r.get("bfs_reorder", {}).get("triggered")),
        "n_reorder_success": n_reorder,
        "n_embodied_tests": n_embodied,
        "n_embodied_ok": n_embodied_ok,
        "note_3block": "3-block embodied execution skipped: AtomicPickSkill/AtomicPlaceSkill fail at physics layer (0/20 oracle). Root cause documented in oracle eval.",
    }

    output = {"summary": summary, "results": results}
    out_path = out_dir / "qwen_coarse_embodied_eval.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved to {out_path}")
    print(f"Parse: {n_parse_ok}/{len(results)} | Verify: {n_verify_ok}/{len(results)} | "
          f"Forbidden: {n_forbidden} | Reorder: {n_reorder} | Embodied: {n_embodied_ok}/{n_embodied}")


def _run_embodied(steps, objects, tid, n_episodes=5):
    """Run embodied execution for 2-block."""
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
    from vlm_nsm.embodied.skills.atomic_pick_skill import AtomicPickSkill
    from vlm_nsm.embodied.skills.atomic_place_skill import AtomicPlaceSkill
    from vlm_nsm.embodied.skills.base_skill import SkillParams
    from vlm_nsm.neuro_symbolic.coarse_planner_bridge import CoarsePlannerBridge
    from vlm_nsm.language.coarse_dsl import CoarseTaskProgram
    from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
    from vlm_nsm.core.predicate_monitor import goal_satisfied

    def _json_safe(obj):
        if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
        elif isinstance(obj, set): return list(obj)
        elif hasattr(obj, 'tolist'): return obj.tolist()
        elif hasattr(obj, 'item'): return obj.item()
        return obj

    cube_names = [o for o in objects if o != "table"]
    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                              cube_names=cube_names, render_mode="none")
    env.env.horizon = 5000

    bridge = CoarsePlannerBridge()
    seq = bridge.execution_sequence(steps, objects)
    pick_skill = AtomicPickSkill()
    place_skill = AtomicPlaceSkill()

    episodes = []
    for ep in range(n_episodes):
        np.random.seed(ep)
        pid = OperationalSpacePID()
        env.reset()
        pid.reset()

        trace = []
        total_steps = 0
        failed = False
        final_failure = None

        for i, exec_step in enumerate(seq):
            skill_name = exec_step["skill"]
            obj = exec_step.get("object", "")
            tgt = exec_step.get("target", "")

            if skill_name == "atomic_pick":
                params = SkillParams(target_object=obj)
                result = pick_skill.execute_with_pid(env, pid, params)
                total_steps += result.steps
                entry = {"step": i, "skill": "atomic_pick", "object": obj,
                         "success": result.success, "steps": result.steps,
                         "failure_type": result.failure_type}
                if not result.success:
                    failed = True; final_failure = result.failure_type
            elif skill_name == "atomic_place":
                params = SkillParams(target_object=obj, target_location=tgt)
                result = place_skill.execute_with_pid(env, pid, params)
                total_steps += result.steps
                entry = {"step": i, "skill": "atomic_place", "object": obj,
                         "target": tgt, "success": result.success,
                         "steps": result.steps,
                         "failure_type": result.failure_type}
                if not result.success:
                    failed = True; final_failure = result.failure_type
            else:
                entry = {"step": i, "skill": skill_name, "success": False,
                         "failure_type": f"unknown_skill_{skill_name}"}
                failed = True

            trace.append(_json_safe(entry))
            if failed:
                break

        # Goal check
        goal_preds = []
        try:
            prog = CoarseTaskProgram.from_json({"steps": steps})
            if prog: goal_preds = prog.to_goal_predicates()
        except Exception: pass

        final_ok = False
        if goal_preds and not failed:
            try:
                poses = env.get_all_object_poses()
                targets = dict(poses)
                targets["table"] = env.get_object_pose("table")
                physical = physical_state_to_predicates(poses, targets,
                    {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05})
                final_ok = goal_satisfied(physical, goal_preds, mode="symbolic")
            except Exception: pass

        success = (not failed) and final_ok
        episodes.append(_json_safe({
            "episode": ep, "seed": ep, "success": success,
            "failure_type": final_failure, "n_total_steps": total_steps,
            "final_goal_ok": final_ok, "trace": trace,
        }))

    n_ok = sum(1 for e in episodes if e["success"])
    return {"n_episodes": n_episodes, "n_success": n_ok,
            "success_rate": n_ok / n_episodes, "episodes": episodes}


if __name__ == "__main__":
    main()
