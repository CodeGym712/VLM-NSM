"""Search robosuite execution parameters for place stability and step efficiency.

Grid search over: kp, max_delta, early_stop, close_steps, reach_threshold.

Usage:
    MUJOCO_GL=egl python3 -m vlm_nsm.experiments.search_robosuite_execution_params
    MUJOCO_GL=egl python3 -m vlm_nsm.experiments.search_robosuite_execution_params --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _try_import_robosuite():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper  # noqa: F401
        return True
    except ImportError:
        return False


def _build_goal(cubes: list[str]):
    from vlm_nsm.core.data_types import Predicate
    target_stack = cubes[::-1]
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def run_single_config(
    env,
    actions: list,
    cubes: list[str],
    goal_predicates: list,
    kp: float,
    max_delta: float,
    early_stop: bool = True,
    close_steps: int = 50,
    reach_threshold: float = 0.02,
) -> dict:
    """Run one episode with given params. Returns step count and success."""
    from vlm_nsm.embodied.skill_step_decomposer import SkillStepDecomposer
    from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
    from vlm_nsm.core.predicate_monitor import goal_satisfied

    decomposer = SkillStepDecomposer()
    env.reset()
    total_steps = 0
    episode_ok = True
    goal_ok = False
    HORIZON = getattr(env.env, "horizon", 1000)

    _STEP_LIMIT = 500
    _GRIPPER_OPEN = -1.0
    _GRIPPER_CLOSE = 1.0
    _OSC_POSE_MAX_DELTA = 0.05
    _LIFT_HEIGHT = 0.05

    def _norm_action(dx, dy, dz, gripper):
        return np.array([
            np.clip(dx / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
            np.clip(dy / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
            np.clip(dz / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
            0.0, 0.0, 0.0, gripper,
        ], dtype=np.float32)

    def _dist(a, b):
        return np.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2 + (a["z"] - b["z"])**2)

    for action in actions:
        target_obj = action.args[0] if action.args else cubes[0]
        target_place = action.args[2] if len(action.args) > 2 else cubes[-1]
        decomposed = decomposer.decompose(action)

        for step_name in decomposed.steps:
            if total_steps >= HORIZON - 50:
                break

            if step_name == "reach_pick":
                obj_pose = env.get_object_pose(target_obj)
                target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + 0.10}
                for _ in range(_STEP_LIMIT):
                    try:
                        eef = env.get_eef_pose()
                    except ValueError:
                        episode_ok = False; break
                    dx = np.clip(kp * (target["x"] - eef["x"]), -max_delta, max_delta)
                    dy = np.clip(kp * (target["y"] - eef["y"]), -max_delta, max_delta)
                    dz = np.clip(kp * (target["z"] - eef["z"]), -max_delta, max_delta)
                    try:
                        env.step(_norm_action(dx, dy, dz, _GRIPPER_OPEN))
                    except ValueError:
                        episode_ok = False; break
                    total_steps += 1
                    if _dist(eef, target) < reach_threshold:
                        break
                if not episode_ok:
                    break

            elif step_name == "grasp":
                from vlm_nsm.embodied.grasp_state_machine import execute_grasp as _exec_grasp
                result = _exec_grasp(env, target_obj, skip_approach=True, skip_lift=True)
                total_steps += result.get("total_step_count", 0)
                if not result["success"]:
                    episode_ok = False; break

            elif step_name == "lift":
                obj_pose = env.get_object_pose(target_obj)
                lift_target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + _LIFT_HEIGHT + 0.10}
                for _ in range(_STEP_LIMIT):
                    try:
                        eef = env.get_eef_pose()
                    except ValueError:
                        episode_ok = False; break
                    dx = np.clip(kp * (lift_target["x"] - eef["x"]), -max_delta, max_delta)
                    dy = np.clip(kp * (lift_target["y"] - eef["y"]), -max_delta, max_delta)
                    dz = np.clip(kp * (lift_target["z"] - eef["z"]), -max_delta, max_delta)
                    try:
                        env.step(_norm_action(dx, dy, dz, _GRIPPER_CLOSE))
                    except ValueError:
                        episode_ok = False; break
                    total_steps += 1
                    if _dist(eef, lift_target) < reach_threshold:
                        break
                if not episode_ok:
                    break

            elif step_name == "reach_place":
                tgt_pose = env.get_target_pose(target_place)
                place_target = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": tgt_pose["z"] + 0.10}
                for _ in range(2 * _STEP_LIMIT):
                    try:
                        eef = env.get_eef_pose()
                    except ValueError:
                        episode_ok = False; break
                    dx = np.clip(kp * (place_target["x"] - eef["x"]), -max_delta, max_delta)
                    dy = np.clip(kp * (place_target["y"] - eef["y"]), -max_delta, max_delta)
                    dz = np.clip(kp * (place_target["z"] - eef["z"]), -max_delta, max_delta)
                    try:
                        env.step(_norm_action(dx, dy, dz, _GRIPPER_CLOSE))
                    except ValueError:
                        episode_ok = False; break
                    total_steps += 1
                    if _dist(eef, place_target) < reach_threshold:
                        break
                if not episode_ok:
                    break

            elif step_name == "release":
                for _ in range(close_steps):
                    try:
                        env.step(_norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
                    except ValueError:
                        episode_ok = False; break
                    total_steps += 1
                if not episode_ok:
                    break

            elif step_name == "retreat":
                try:
                    eef = env.get_eef_pose()
                except ValueError:
                    episode_ok = False; break
                retreat_target = {"x": eef["x"], "y": eef["y"], "z": eef["z"] + 0.15}
                for _ in range(_STEP_LIMIT):
                    try:
                        eef = env.get_eef_pose()
                    except ValueError:
                        episode_ok = False; break
                    dx = np.clip(kp * (retreat_target["x"] - eef["x"]), -max_delta, max_delta)
                    dy = np.clip(kp * (retreat_target["y"] - eef["y"]), -max_delta, max_delta)
                    dz = np.clip(kp * (retreat_target["z"] - eef["z"]), -max_delta, max_delta)
                    try:
                        env.step(_norm_action(dx, dy, dz, _GRIPPER_OPEN))
                    except ValueError:
                        episode_ok = False; break
                    total_steps += 1
                    if _dist(eef, retreat_target) < reach_threshold:
                        break
                if not episode_ok:
                    break

        if early_stop and total_steps > HORIZON - 100:
            break

    # Check goal at end
    object_poses = env.get_all_object_poses()
    targets = dict(object_poses)
    targets["table"] = env.get_object_pose("table")
    thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
    physical_predicates = physical_state_to_predicates(object_poses, targets, thresholds)
    if goal_predicates:
        goal_ok = goal_satisfied(physical_predicates, goal_predicates, mode="symbolic")

    return {
        "total_steps": total_steps,
        "execution_success": episode_ok,
        "goal_success": bool(goal_ok),
        "episode_success": episode_ok and bool(goal_ok),
    }


def search_params(num_episodes: int = 5, quick: bool = False, search_mode: str = "default") -> dict:
    """Grid search over execution parameters."""
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    cubes = ["cube1", "cube2"]
    goal_predicates = _build_goal(cubes)

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=cubes[::-1])
    if not plan_result.success:
        return {"error": f"Planner failed: {plan_result.error}"}

    if search_mode == "state_machine":
        # Grid over grasp/place state machine params
        close_steps_values = [30, 50, 70] if not quick else [50]
        reach_threshold_values = [0.015, 0.02, 0.025] if not quick else [0.02]
        lift_threshold_values = [0.003, 0.005, 0.01] if not quick else [0.005]
        max_retries_values = [2, 3, 5] if not quick else [3]
        place_mode_values = ["simple", "fast", "safe"] if not quick else ["simple"]
        configs = list(itertools.product(close_steps_values, reach_threshold_values,
                                          lift_threshold_values, max_retries_values,
                                          place_mode_values))
        total_configs = len(configs)
        print(f"Searching {total_configs} state machine configs x {num_episodes} episodes")
        # Use fixed kp/max_delta for state machine search
        _fixed_kp = 0.08
        _fixed_max_delta = 0.015
    elif search_mode == "grasp":
        # Grasp-focused grid over close params
        close_steps_values = [15, 20, 25, 30] if not quick else [20]
        lift_threshold_values = [0.005, 0.01, 0.02, 0.03] if not quick else [0.01]
        reach_threshold_values = [0.012, 0.015, 0.018, 0.02] if not quick else [0.015]
        descend_offset_values = [0.003, 0.005, 0.008] if not quick else [0.005]
        max_retries_values = [0, 1] if not quick else [1]
        place_mode_values = ["simple", "fast"] if not quick else ["simple"]
        configs = list(itertools.product(close_steps_values, lift_threshold_values,
                                          reach_threshold_values, descend_offset_values,
                                          max_retries_values, place_mode_values))
        total_configs = len(configs)
        print(f"Searching {total_configs} grasp-focused configs x {num_episodes} episodes")
        _fixed_kp = 0.08
        _fixed_max_delta = 0.015
    elif quick:
        kp_values = [0.03, 0.05, 0.08]
        max_delta_values = [0.015, 0.02, 0.03]
        early_stop_values = [True, False]
        close_steps_values = [50]
        reach_threshold_values = [0.02]
        configs = list(itertools.product(kp_values, max_delta_values, early_stop_values, close_steps_values, reach_threshold_values))
    else:
        kp_values = [0.02, 0.03, 0.05, 0.08]
        max_delta_values = [0.01, 0.015, 0.02, 0.03]
        early_stop_values = [True, False]
        close_steps_values = [30, 50, 70]
        reach_threshold_values = [0.015, 0.02, 0.025]
        configs = list(itertools.product(kp_values, max_delta_values, early_stop_values, close_steps_values, reach_threshold_values))
    total_configs = len(configs)
    print(f"Searching {total_configs} configs x {num_episodes} episodes = {total_configs * num_episodes} runs")

    results = []
    best_success_rate = 0.0
    best_config = None

    if search_mode == "grasp":
        for idx, (close_steps, lift_threshold, reach_threshold, descend_offset, max_retries, place_mode) in enumerate(configs):
            env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
            episode_results = []

            for ep in range(num_episodes):
                ep_result = run_single_config(env, plan_result.actions, cubes, goal_predicates,
                                               kp=_fixed_kp, max_delta=_fixed_max_delta, early_stop=True,
                                               close_steps=close_steps, reach_threshold=reach_threshold)
                episode_results.append(ep_result)

            env.env.close()

            success_count = sum(1 for r in episode_results if r["episode_success"])
            goal_count = sum(1 for r in episode_results if r["goal_success"])
            mean_steps = float(np.mean([r["total_steps"] for r in episode_results]))
            success_rate = success_count / max(num_episodes, 1)

            config_result = {
                "kp": _fixed_kp,
                "max_delta": _fixed_max_delta,
                "early_stop": True,
                "close_steps": close_steps,
                "reach_threshold": reach_threshold,
                "lift_threshold": lift_threshold,
                "descend_offset": descend_offset,
                "max_retries": max_retries,
                "place_mode": place_mode,
                "num_episodes": num_episodes,
                "episode_success_rate": success_rate,
                "goal_success_rate": goal_count / max(num_episodes, 1),
                "mean_steps": round(mean_steps, 1),
                "per_episode": episode_results,
            }
            results.append(config_result)

            label = f"close={close_steps} lt={lift_threshold} rt={reach_threshold} do={descend_offset} ret={max_retries} mode={place_mode}"
            print(f"  [{idx+1}/{total_configs}] {label:55s} success={success_rate*100:.0f}%  steps={mean_steps:.0f}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_config = config_result
            elif success_rate == best_success_rate and best_config is not None:
                if mean_steps < best_config["mean_steps"]:
                    best_config = config_result
    elif search_mode == "state_machine":
        for idx, (close_steps, reach_threshold, lift_threshold, max_retries, place_mode) in enumerate(configs):
            env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
            episode_results = []

            for ep in range(num_episodes):
                ep_result = run_single_config(env, plan_result.actions, cubes, goal_predicates,
                                               kp=_fixed_kp, max_delta=_fixed_max_delta, early_stop=True,
                                               close_steps=close_steps, reach_threshold=reach_threshold)
                episode_results.append(ep_result)

            env.env.close()

            success_count = sum(1 for r in episode_results if r["episode_success"])
            goal_count = sum(1 for r in episode_results if r["goal_success"])
            mean_steps = float(np.mean([r["total_steps"] for r in episode_results]))
            success_rate = success_count / max(num_episodes, 1)

            config_result = {
                "kp": _fixed_kp,
                "max_delta": _fixed_max_delta,
                "early_stop": True,
                "close_steps": close_steps,
                "reach_threshold": reach_threshold,
                "lift_threshold": lift_threshold,
                "max_retries": max_retries,
                "place_mode": place_mode,
                "num_episodes": num_episodes,
                "episode_success_rate": success_rate,
                "goal_success_rate": goal_count / max(num_episodes, 1),
                "mean_steps": round(mean_steps, 1),
                "per_episode": episode_results,
            }
            results.append(config_result)

            label = f"close={close_steps} rt={reach_threshold} lift_t={lift_threshold} ret={max_retries} mode={place_mode}"
            print(f"  [{idx+1}/{total_configs}] {label:55s} success={success_rate*100:.0f}%  steps={mean_steps:.0f}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_config = config_result
            elif success_rate == best_success_rate and best_config is not None:
                if mean_steps < best_config["mean_steps"]:
                    best_config = config_result
    else:
        for idx, (kp, max_delta, early_stop, close_steps, reach_threshold) in enumerate(configs):
            env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
            episode_results = []

            for ep in range(num_episodes):
                ep_result = run_single_config(env, plan_result.actions, cubes, goal_predicates,
                                               kp=kp, max_delta=max_delta, early_stop=early_stop,
                                               close_steps=close_steps, reach_threshold=reach_threshold)
                episode_results.append(ep_result)

            env.env.close()

            success_count = sum(1 for r in episode_results if r["episode_success"])
            goal_count = sum(1 for r in episode_results if r["goal_success"])
            mean_steps = float(np.mean([r["total_steps"] for r in episode_results]))
            success_rate = success_count / max(num_episodes, 1)

            config_result = {
                "kp": kp,
                "max_delta": max_delta,
                "early_stop": early_stop,
                "close_steps": close_steps,
                "reach_threshold": reach_threshold,
                "num_episodes": num_episodes,
                "episode_success_rate": success_rate,
                "goal_success_rate": goal_count / max(num_episodes, 1),
                "mean_steps": round(mean_steps, 1),
                "per_episode": episode_results,
            }
            results.append(config_result)

            label = f"kp={kp} max_delta={max_delta} early_stop={early_stop} close={close_steps} rt={reach_threshold}"
            print(f"  [{idx+1}/{total_configs}] {label:55s} success={success_rate*100:.0f}%  steps={mean_steps:.0f}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_config = config_result
            elif success_rate == best_success_rate and best_config is not None:
                if mean_steps < best_config["mean_steps"]:
                    best_config = config_result

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_config = config_result
        elif success_rate == best_success_rate and best_config is not None:
            if mean_steps < best_config["mean_steps"]:
                best_config = config_result

    return {
        "num_configs": total_configs,
        "episodes_per_config": num_episodes,
        "results": results,
        "best_config": best_config,
    }


def main():
    parser = argparse.ArgumentParser(description="Search robosuite execution parameters")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Smaller grid")
    parser.add_argument("--mode", type=str, default="default",
                        choices=["default", "state_machine", "grasp"],
                        help="Search mode: default (kp/max_delta/close/rt), state_machine, or grasp (close_steps/lift_threshold/rt/descend/retries)")
    args = parser.parse_args()

    if not _try_import_robosuite():
        print("ERROR: Robosuite not available.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    search = search_params(num_episodes=args.num_episodes, quick=args.quick, search_mode=args.mode)

    if "error" in search:
        print(f"ERROR: {search['error']}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("EXECUTION PARAM SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Configs tested: {search['num_configs']} x {search['episodes_per_config']} episodes")
    print(f"Time: {elapsed:.0f}s")

    best = search.get("best_config", {})
    if best:
        print(f"\nBest config:")
        print(f"  kp={best['kp']}, max_delta={best['max_delta']}, early_stop={best['early_stop']}")
        print(f"  close_steps={best['close_steps']}, reach_threshold={best['reach_threshold']}")
        if "lift_threshold" in best:
            print(f"  lift_threshold={best['lift_threshold']}, max_retries={best.get('max_retries')}, place_mode={best.get('place_mode')}")
        if "descend_offset" in best:
            print(f"  descend_offset={best['descend_offset']}")
        print(f"  Episode success rate: {best['episode_success_rate']*100:.0f}%")
        print(f"  Mean steps: {best['mean_steps']:.0f}")

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / "robosuite_execution_param_search.json"
    output_path.write_text(json.dumps(search, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
