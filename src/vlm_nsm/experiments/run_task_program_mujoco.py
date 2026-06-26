"""OptionDSL MuJoCo Runner — execute TaskProgram through PID controller.

Usage:
    python -m vlm_nsm.experiments.run_task_program_mujoco \
        --task stacking_2block --subgoals "[[["clear","cube2"]],[["on","cube1","cube2"]]]" \
        --seed 0 --executor linear --output generated/results/upper_ns/task_program_mujoco/
"""

import argparse, json, math, os
from pathlib import Path
import numpy as np

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.skills.base_skill import SkillLibrary, SkillParams
from vlm_nsm.embodied.skills.reach_skill import ReachSkill, ReachPlaceSkill, RetreatSkill
from vlm_nsm.embodied.skills.grasp_skill import GraspSkill, ReleasePlaceSkill
from vlm_nsm.embodied.skills.lift_skill import LiftSkill
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.neuro_symbolic.verifier_bank import standard_verifier_bank

def _check_goal(env, goal_preds):
    try:
        poses=env.get_all_object_poses();targets=dict(poses)
        targets["table"]=env.get_object_pose("table")
        physical=physical_state_to_predicates(poses,targets,
            {"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
        return goal_satisfied(physical,goal_preds,mode="symbolic")
    except: return False

def run_task_program(env, pid, lib, subgoals, cubes):
    """Execute TaskProgram subgoals sequentially through PID."""
    trace = []
    for sg_idx, sg in enumerate(subgoals):
        # Verify subgoal
        bank = standard_verifier_bank(cubes + ["table"])
        report = bank.verify_all(goal_json={"goal_predicates": sg}, task_type="stacking")
        if not report.passed:
            trace.append({"subgoal": sg_idx, "status": "rejected",
                          "reason": str(report.failures)})
            continue

        # Plan: use BFS to achieve this subgoal
        ts = [p[2] for p in sg if p[0]=="on" and len(p)>=3] + \
             [p[1] for p in sg if p[0]=="on" and len(p)>=3]
        if not ts: ts = cubes[::-1]
        planner = PlannerAdapter("bfs")
        plan = planner.solve(cubes=cubes, target_stack=list(dict.fromkeys(ts)))

        if not plan.success:
            trace.append({"subgoal": sg_idx, "status": "plan_failed",
                          "reason": plan.error})
            continue

        # Execute each action — all skills through SkillLibrary (Phase 4.2 validation target)
        for action in plan.actions:
            pick = action.args[0]
            place = action.args[2] if len(action.args) > 2 else "table"
            for skill_name in ["reach", "grasp", "lift", "reach_place", "release_place"]:
                try:
                    skill = lib.get(skill_name)
                except KeyError:
                    continue
                tgt = place if "place" in skill_name else pick
                # Match direct PID approach heights
                approach = 0.15 if "place" in skill_name else 0.10
                params = SkillParams(target_object=pick, target_location=tgt,
                                     approach_height=approach)
                result = skill.execute_with_pid(env, pid, params)
                if not result.success:
                    trace.append({"subgoal": sg_idx, "status": "exec_failed",
                                  "skill": skill_name, "failure_type": result.failure_type})
                    return trace
        trace.append({"subgoal": sg_idx, "status": "executed", "predicates": sg})

    # Final goal check
    all_goals = [p for sg in subgoals for p in sg]
    final_ok = _check_goal(env, all_goals) if all_goals else True
    trace.append({"status": "final_check", "goal_ok": final_ok})
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stacking_2block")
    parser.add_argument("--subgoals", default='[[["on","cube1","cube2"]]]',
                        help="JSON list of subgoal predicate lists")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--executor", default="linear")
    parser.add_argument("--output", default="generated/results/upper_ns/task_program_mujoco_eval/")
    args = parser.parse_args()

    subgoals = json.loads(args.subgoals)
    cubes = ["cube1","cube2"] if "2block" in args.task else ["cube1","cube2","cube3"]

    if len(cubes) >= 3:
        import vlm_nsm.executors.stack_three_cube_env as _  # noqa

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                              cube_names=cubes, render_mode="none")
    env.env.horizon = 10000 if len(cubes)>=3 else 5000
    pid = OperationalSpacePID()
    lib = SkillLibrary()
    for s in [ReachSkill(),GraspSkill(),LiftSkill(),ReachPlaceSkill(),ReleasePlaceSkill(),RetreatSkill()]:
        lib.register(s)

    np.random.seed(args.seed)
    env.reset(); pid.reset()
    trace = run_task_program(env, pid, lib, subgoals, cubes)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"task": args.task, "seed": args.seed, "subgoals": subgoals,
               "trace": trace, "success": trace[-1].get("goal_ok", False) if trace else False}
    with open(out / f"seed{args.seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"TaskProgram result: success={summary['success']}, "
          f"steps={len(trace)} subgoals, saved to {out / f'seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
