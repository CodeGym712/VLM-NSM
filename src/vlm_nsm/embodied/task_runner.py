"""Generic Task Runner — executes any TaskSpec using PID + Skill Library.

Replaces hardcoded run_embodied_stacking_robosuite.py logic with
task-agnostic execution. New tasks need only a TaskSpec config.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.physical_predicate_extractor import (
    physical_state_to_predicates,
)
from vlm_nsm.core.task_spec import TaskSpec
from vlm_nsm.embodied.skills.base_skill import (
    SkillLibrary, SkillParams, SkillResult,
)
from vlm_nsm.embodied.skills.reach_skill import (
    ReachSkill, ReachPlaceSkill, RetreatSkill,
)
from vlm_nsm.embodied.skills.grasp_skill import GraspSkill, ReleasePlaceSkill
from vlm_nsm.embodied.skills.lift_skill import LiftSkill

_STEP_LIMIT = 1000  # increased for multi-task (Hanoi needs ~500/skill × 42 skills)
_PREDICATE_THRESHOLDS = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}


def _default_skill_library() -> SkillLibrary:
    lib = SkillLibrary()
    lib.register(ReachSkill())
    lib.register(GraspSkill())
    lib.register(LiftSkill())
    lib.register(ReachPlaceSkill())
    lib.register(ReleasePlaceSkill())
    lib.register(RetreatSkill())
    from vlm_nsm.embodied.skills.pull_skill import PullSkill
    lib.register(PullSkill())
    return lib


def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
    return [
        Predicate("on", [obj, support])
        for support, obj in zip(target_stack, target_stack[1:])
    ]


def _check_goal(env, cubes: list[str],
                goal_predicates: list[Predicate] | None) -> bool:
    if not goal_predicates:
        return False
    object_poses = env.get_all_object_poses()
    targets = dict(object_poses)
    try:
        targets["table"] = env.get_object_pose("table")
    except (ValueError, AttributeError):
        pass
    physical_predicates = physical_state_to_predicates(
        object_poses, targets, _PREDICATE_THRESHOLDS)
    return bool(goal_satisfied(physical_predicates, goal_predicates,
                               mode="symbolic"))


def run_task(
    env,
    task: TaskSpec,
    pid_ctrl=None,
    skill_library: SkillLibrary | None = None,
) -> dict:
    """Execute a task specification using PID controller + skill library.

    Args:
        env: RobosuiteEnvWrapper instance.
        task: TaskSpec defining the task.
        pid_ctrl: OperationalSpacePID instance (created if None).
        skill_library: SkillLibrary (default skills used if None).

    Returns:
        Dict with execution metrics.
    """
    if pid_ctrl is None:
        from vlm_nsm.embodied.controllers.operational_space_pid import (
            OperationalSpacePID,
        )
        pid_ctrl = OperationalSpacePID()

    if skill_library is None:
        skill_library = _default_skill_library()

    cubes = [f"cube{i}" for i in range(1, task.n_objects + 1)]
    target_stack = task.target_stack or cubes[::-1]
    goal_predicates = task.goal_predicates or build_stacking_goal(target_stack)

    # Plan (support task-specific planner kwargs)
    planner = PlannerAdapter(task.planner)
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack,
                                **task.planner_kwargs)
    if not plan_result.success:
        return {"success": False, "error": plan_result.error,
                "episode_success_rate": 0.0}

    env.reset()
    env.env.horizon = task.horizon

    # Custom initial state (e.g. Hanoi peg placement)
    if task.init_hook is not None:
        task.init_hook(env, task)
    # Safe retract: move EEF to high Z before starting skills (avoid tower collision)
    eef = env.get_eef_pose()
    retract_z = max(eef["z"] + 0.15, 1.15)  # at least 1.15m (above tallest tower)
    for _ in range(100):
        a = np.zeros(7, dtype=np.float32)
        a[2] = 0.3  # move up
        a[6] = -1.0  # open gripper
        try:
            env.step(a)
        except ValueError:
            break
        if env.get_eef_pose()["z"] >= retract_z:
            break
    total_steps = 0
    episode_success = True
    final_failure: str | None = None
    goal_achieved = False
    place_done = False
    failure_counts: dict[str, int] = {}
    phase_trace: dict[str, int] = {s: 0 for s in task.skills}
    replans_left = task.max_replans

    # Peg map for Hanoi pick_place actions
    _PEG_MAP = {"peg1": (-0.12, -0.08), "peg2": (0.0, -0.08), "peg3": (0.12, -0.08)}

    _actions = list(plan_result.actions)
    _action_idx = 0
    while _action_idx < len(_actions):
        action = _actions[_action_idx]

        # Determine target objects based on action handler
        if task.action_handler == "pick_place":
            # Hanoi: pick(disk, from_peg) / place(disk, to_peg)
            action_name = getattr(action, "name", "")
            disk = action.args[0] if action.args else cubes[0]
            peg = action.args[1] if len(action.args) > 1 else "peg3"
            if action_name == "pick":
                active_skills = ["reach", "grasp", "lift"]
                target_obj = disk
                target_place = disk  # unused for pick
            elif action_name == "place":
                active_skills = ["reach_place", "release_place", "retreat"]
                target_obj = disk
                # Override get_target for place: use peg coordinates
                target_place = peg
            else:
                _action_idx += 1
                continue
        else:
            # Default "move": stacking (obj, source, target)
            active_skills = task.skills
            target_obj = action.args[0] if action.args else cubes[0]
            target_place = action.args[2] if len(action.args) > 2 else cubes[-1]

        if task.early_stop_on_goal and goal_achieved:
            break

        for skill_name in active_skills:
            if place_done and skill_name in ("release_place", "retreat"):
                continue

            remaining = max(0, task.horizon - total_steps)

            if skill_name in ("reach", "grasp", "lift"):
                tgt = target_obj
            elif skill_name in ("reach_place", "release_place"):
                tgt = target_place
            else:
                tgt = target_obj

            gripper_map = {
                "reach": -1.0, "grasp": 1.0, "lift": 1.0,
                "reach_place": 1.0, "release_place": -1.0, "retreat": -1.0,
            }
            phase_id_map = {
                "reach": 0, "grasp": 1, "lift": 2,
                "reach_place": 3, "release_place": 4, "retreat": 5,
            }

            # Resolve peg target to coordinates for Hanoi place
            _tgt = tgt
            _approach = 0.10 if "reach" in skill_name else 0.005
            if isinstance(tgt, str) and tgt.startswith("peg"):
                px, py = _PEG_MAP.get(tgt, (0.0, -0.08))
                _tgt = f"peg:{px},{py}"  # pass via string; skill resolves
                _approach = 0.15

            params = SkillParams(
                target_object=target_obj,
                target_location=_tgt,
                approach_height=_approach,
                gripper_cmd=gripper_map.get(skill_name, -1.0),
                max_steps=min(_STEP_LIMIT, max(remaining, 10)),
                phase_id=phase_id_map.get(skill_name, 0),
            )

            try:
                skill = skill_library.get(skill_name)
            except KeyError:
                continue

            # Capture pre-state predicates (Commit 2: OptionResult)
            pre_preds = []
            try:
                from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
                poses = env.get_all_object_poses()
                targets = dict(poses)
                try: targets["table"] = env.get_object_pose("table")
                except: pass
                physical = physical_state_to_predicates(poses, targets,
                    {"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
                pre_preds = [list(p) if hasattr(p,'__iter__') else [p] for p in physical]
            except Exception:
                pass

            result = skill.execute_with_pid(env, pid_ctrl, params)
            total_steps += result.steps
            phase_trace[skill_name] = phase_trace.get(skill_name, 0) + result.steps

            # Capture post-state and check effects (Commit 2)
            if pre_preds:
                try:
                    poses = env.get_all_object_poses()
                    targets = dict(poses)
                    try: targets["table"] = env.get_object_pose("table")
                    except: pass
                    physical = physical_state_to_predicates(poses, targets,
                        {"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
                    post_preds = [list(p) for p in physical]
                    result.pre_predicates = pre_preds
                    result.post_predicates = post_preds
                    # Check against option contract
                    try:
                        from vlm_nsm.core.option_contract import get_contract
                        contract = get_contract(skill_name)
                        result.expected_effect = contract.expected_effects
                        ok, _ = contract.check_effects(pre_preds, post_preds)
                        result.effect_satisfied = ok
                    except KeyError:
                        pass
                except Exception:
                    pass

            if not result.success:
                episode_success = False
                ft = result.failure_type or f"{skill_name}_failed"
                final_failure = ft
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

                # 3-block replan: intermediate stack collapse detection
                if replans_left > 0 and _action_idx > 0:
                    try:
                        first_action = _actions[0]
                        mid_preds = build_stacking_goal(
                            [first_action.args[2], first_action.args[0]])
                        if not _check_goal(env, cubes, mid_preds):
                            replan = planner.solve(
                                cubes=cubes, target_stack=target_stack)
                            if replan.success:
                                replans_left -= 1
                                _actions = list(replan.actions)
                                _action_idx = 0
                                episode_success = True
                                final_failure = None
                                place_done = False
                                continue  # restart from first action
                    except Exception:
                        pass
                break

            if skill_name == "release_place":
                place_done = True

            try:
                if task.early_stop_on_goal and _check_goal(env, cubes, goal_predicates):
                    goal_achieved = True
                    break
            except Exception:
                pass

        if goal_achieved:
            break
        _action_idx += 1

    # Final goal check
    final_goal_ok = _check_goal(env, cubes, goal_predicates)

    step_success_rate = (
        sum(1 for s in phase_trace.values() if s > 0) / max(len(task.skills), 1)
    )

    return {
        "success": final_goal_ok,
        "episode_success_rate": 1.0 if final_goal_ok else 0.0,
        "goal_success": final_goal_ok,
        "num_steps": total_steps,
        "mean_steps": float(total_steps),
        "failure_type": final_failure,
        "failure_stage_distribution": failure_counts,
        "phase_trace": phase_trace,
        "step_success_rate": step_success_rate,
        "skill_success_rate": 0.0,
    }


def run_task_eval(
    task: TaskSpec,
    num_episodes: int = 50,
    render: bool = False,
    device: str = "cuda:0",
) -> dict:
    """Run multi-episode evaluation of a task.

    Args:
        task: TaskSpec to evaluate.
        num_episodes: Number of episodes.
        render: Whether to render (slower).
        device: GPU device.

    Returns:
        Aggregated metrics across all episodes.
    """
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.embodied.controllers.operational_space_pid import (
        OperationalSpacePID,
    )

    # 3-block: monkey-patch Stack
    if task.n_objects >= 3:
        import vlm_nsm.executors.stack_three_cube_env as _  # noqa: F401

    render_mode = "human" if render else "none"
    env = RobosuiteEnvWrapper(
        env_name=task.env_name,
        robots=task.env_kwargs.get("robots", "Panda"),
        cube_names=task.env_kwargs.get(
            "cube_names",
            [f"cube{i}" for i in range(1, task.n_objects + 1)]),
        render_mode=render_mode,
    )

    pid = OperationalSpacePID()
    lib = _default_skill_library()

    successes = []
    failures: dict[str, int] = {}
    all_steps = []

    for _ in range(num_episodes):
        result = run_task(env, task, pid_ctrl=pid, skill_library=lib)
        successes.append(result["goal_success"])
        all_steps.append(result["num_steps"])
        for ft, count in result.get("failure_stage_distribution", {}).items():
            failures[ft] = failures.get(ft, 0) + count
        pid.reset()

    return {
        "task": task.name,
        "num_episodes": num_episodes,
        "episode_success_rate": sum(successes) / len(successes),
        "mean_steps": float(np.mean(all_steps)),
        "failure_stage_distribution": failures,
    }


# ── CLI Entry Point ────────────────────────────────────────────────────────

def main() -> None:
    """Run a task spec via CLI. Usage: python -m vlm_nsm.embodied.task_runner --task stack_2block"""
    import argparse
    from vlm_nsm.core.task_spec import load_task_spec, get_stacking_2block_spec

    parser = argparse.ArgumentParser(description="Generic Task Runner")
    parser.add_argument("--task", default="stack_2block",
                        help="Task name or path to YAML/JSON task spec")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Lazy imports for heavy dependencies (MuJoCo, robosuite)
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.embodied.controllers.operational_space_pid import (
        OperationalSpacePID, GainSchedule,
    )

    # Load task spec
    if args.task == "stack_2block":
        task = get_stacking_2block_spec()
    else:
        task = load_task_spec(args.task)

    # Create env and PID
    env = RobosuiteEnvWrapper(
        env_name=task.env_name,
        robots=task.env_kwargs.get("robots", "Panda"),
        cube_names=[f"cube{i}" for i in range(1, task.n_objects + 1)],
        render_mode="offscreen",
    )
    pid = OperationalSpacePID(gain_schedule=GainSchedule())

    result = run_task_eval(
        task=task, env=env, pid=pid,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    print(f"Task: {result['task']}")
    print(f"Success: {result['episode_success_rate']:.0%} ({result['num_episodes']}ep)")
    print(f"Mean steps: {result['mean_steps']:.0f}")
    if args.output_dir:
        import json
        from pathlib import Path
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
