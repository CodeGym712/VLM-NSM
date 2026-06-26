"""Analyze robosuite step budget consumption per skill and phase.

Usage:
    MUJOCO_GL=egl python3 -m vlm_nsm.experiments.analyze_robosuite_step_budget --num_episodes 10
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _try_import_robosuite():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper  # noqa: F401
        return True
    except ImportError:
        return False


def analyze_step_budget(num_episodes: int = 10) -> dict:
    """Run stacking evaluation and collect per-phase step counts."""
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.core.data_types import Predicate
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.embodied.skill_step_decomposer import SkillStepDecomposer

    cubes = ["cube1", "cube2"]
    target_stack = cubes[::-1]
    goal_predicates = [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        return {"error": f"Planner failed: {plan_result.error}"}

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
    decomposer = SkillStepDecomposer()

    HORIZON = env.env.horizon if hasattr(env.env, "horizon") else 1000

    per_episode_budget: list[dict] = []
    phase_step_stats: dict[str, list[int]] = {}
    total_steps_list: list[int] = []

    for ep in range(num_episodes):
        env.reset()
        steps_used = 0

        actions = plan_result.actions
        phase_counts: dict[str, int] = {}

        for action_idx, action in enumerate(actions):
            decomposed = decomposer.decompose(action)
            for step_name in decomposed.steps:
                phase_steps = 0

                if step_name == "reach_pick":
                    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _STEP_LIMIT, _REACH_THRESHOLD, _norm_action, _GRIPPER_OPEN, _dist
                    kp, max_delta = 0.05, 0.02
                    target_obj = action.args[0] if action.args else cubes[0]
                    obj_pose = env.get_object_pose(target_obj)
                    target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + 0.10}
                    for _ in range(_STEP_LIMIT):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            break
                        dx = np.clip(kp * (target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_OPEN))
                        except ValueError:
                            break
                        phase_steps += 1
                        steps_used += 1
                        if _dist(eef, target) < _REACH_THRESHOLD:
                            break

                elif step_name == "grasp":
                    from vlm_nsm.embodied.grasp_state_machine import execute_grasp as _exec_grasp
                    target_obj = action.args[0] if action.args else cubes[0]
                    result = _exec_grasp(env, target_obj, skip_approach=True, skip_lift=True)
                    phase_steps = result.get("total_step_count", 0)
                    steps_used += phase_steps

                elif step_name == "lift":
                    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _STEP_LIMIT, _LIFT_HEIGHT, _REACH_THRESHOLD, _norm_action, _GRIPPER_CLOSE, _dist
                    kp, max_delta = 0.05, 0.02
                    target_obj = action.args[0] if action.args else cubes[0]
                    obj_pose = env.get_object_pose(target_obj)
                    lift_target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + _LIFT_HEIGHT + 0.10}
                    for _ in range(_STEP_LIMIT):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            break
                        dx = np.clip(kp * (lift_target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (lift_target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (lift_target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_CLOSE))
                        except ValueError:
                            break
                        phase_steps += 1
                        steps_used += 1
                        if _dist(eef, lift_target) < _REACH_THRESHOLD:
                            break

                elif step_name == "reach_place":
                    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _STEP_LIMIT, _REACH_THRESHOLD, _norm_action, _GRIPPER_CLOSE, _dist
                    kp, max_delta = 0.05, 0.02
                    target_place = action.args[2] if len(action.args) > 2 else cubes[-1]
                    tgt_pose = env.get_target_pose(target_place)
                    place_target = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": tgt_pose["z"] + 0.10}
                    for _ in range(2 * _STEP_LIMIT):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            break
                        dx = np.clip(kp * (place_target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (place_target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (place_target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_CLOSE))
                        except ValueError:
                            break
                        phase_steps += 1
                        steps_used += 1
                        if _dist(eef, place_target) < _REACH_THRESHOLD:
                            break

                elif step_name == "release":
                    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _RELEASE_OPEN_STEPS, _norm_action, _GRIPPER_OPEN
                    for _ in range(_RELEASE_OPEN_STEPS):
                        try:
                            env.step(_norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
                        except ValueError:
                            break
                        phase_steps += 1
                        steps_used += 1

                elif step_name == "retreat":
                    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _STEP_LIMIT, _REACH_THRESHOLD, _norm_action, _GRIPPER_OPEN, _dist
                    kp, max_delta = 0.05, 0.02
                    try:
                        eef = env.get_eef_pose()
                    except ValueError:
                        break
                    retreat_target = {"x": eef["x"], "y": eef["y"], "z": eef["z"] + 0.15}
                    for _ in range(_STEP_LIMIT):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            break
                        dx = np.clip(kp * (retreat_target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (retreat_target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (retreat_target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_OPEN))
                        except ValueError:
                            break
                        phase_steps += 1
                        steps_used += 1
                        if _dist(eef, retreat_target) < _REACH_THRESHOLD:
                            break

                if phase_steps > 0:
                    phase_step_stats.setdefault(step_name, []).append(phase_steps)
                phase_counts[step_name] = phase_steps

        total_steps_list.append(steps_used)
        per_episode_budget.append({
            "episode": ep + 1,
            "total_steps": steps_used,
            "remaining_budget": max(0, HORIZON - steps_used),
            "phase_breakdown": dict(phase_counts),
        })

    env.env.close()

    analysis = {
        "num_episodes": num_episodes,
        "horizon": HORIZON,
        "total_steps": {
            "mean": float(np.mean(total_steps_list)),
            "std": float(np.std(total_steps_list)),
            "min": int(np.min(total_steps_list)) if total_steps_list else 0,
            "max": int(np.max(total_steps_list)) if total_steps_list else 0,
            "p50": float(np.percentile(total_steps_list, 50)) if total_steps_list else 0,
            "p95": float(np.percentile(total_steps_list, 95)) if total_steps_list else 0,
        },
        "remaining_budget": {
            "mean": float(np.mean([max(0, HORIZON - s) for s in total_steps_list])),
            "min": int(np.min([max(0, HORIZON - s) for s in total_steps_list])),
        },
        "phase_stats": {},
        "per_episode": per_episode_budget,
    }

    for phase, steps in phase_step_stats.items():
        analysis["phase_stats"][phase] = {
            "mean": float(np.mean(steps)),
            "std": float(np.std(steps)),
            "min": int(np.min(steps)),
            "max": int(np.max(steps)),
            "p50": float(np.percentile(steps, 50)),
            "p95": float(np.percentile(steps, 95)),
            "count": len(steps),
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze robosuite step budget")
    parser.add_argument("--num_episodes", type=int, default=10)
    args = parser.parse_args()

    if not _try_import_robosuite():
        print("ERROR: Robosuite not available.", file=sys.stderr)
        sys.exit(1)

    print(f"Running step budget analysis: {args.num_episodes} episodes...")
    analysis = analyze_step_budget(num_episodes=args.num_episodes)

    if "error" in analysis:
        print(f"ERROR: {analysis['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("STEP BUDGET ANALYSIS")
    print(f"{'='*60}")
    print(f"Horizon: {analysis['horizon']}")
    ts = analysis["total_steps"]
    print(f"Total steps: mean={ts['mean']:.0f}  p50={ts['p50']:.0f}  p95={ts['p95']:.0f}  "
          f"min={ts['min']}  max={ts['max']}")
    rb = analysis["remaining_budget"]
    print(f"Remaining budget: mean={rb['mean']:.0f}  min={rb['min']}")
    print(f"\nPer-phase step consumption:")
    for phase, stats in sorted(analysis["phase_stats"].items()):
        print(f"  {phase:20s}: mean={stats['mean']:.0f}  p50={stats['p50']:.0f}  "
              f"p95={stats['p95']:.0f}  min={stats['min']}  max={stats['max']}  "
              f"n={stats['count']}")

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / "robosuite_step_budget_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
