"""E2E Pipeline: Language(Qwen/Mock) → Plan(BFS) → PID → Monitor → Replan.

First end-to-end integration of the full neuro-symbolic VLA pipeline.
Connects previously independent components in a single episode run.

Usage:
    # Mock parser (no API key needed) — stacking
    MUJOCO_GL=egl nsm/bin/python -m vlm_nsm.experiments.run_e2e_pid_pipeline \
        --instruction "Put the red cube on the green cube" \
        --task stacking_2block --parser_type mock --num_episodes 5

    # Mock parser — Hanoi
    MUJOCO_GL=egl nsm/bin/python -m vlm_nsm.experiments.run_e2e_pid_pipeline \
        --instruction "Move all cubes from peg1 to peg3" \
        --task hanoi_3disk --parser_type mock --num_episodes 5

    # Real Qwen parser (requires QWEN_API_KEY env var)
    MUJOCO_GL=egl nsm/bin/python -m vlm_nsm.experiments.run_e2e_pid_pipeline \
        --instruction "把红色方块放到绿色方块上面" \
        --task stacking_2block --parser_type qwen --num_episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ── Language → Goal ────────────────────────────────────────────────────────

def parse_instruction(instruction: str, task_type: str, parser_type: str,
                      objects: list[str], n_blocks: int) -> dict[str, Any]:
    """Parse NL instruction into goal predicates.

    Returns:
        {"success": bool, "goal_predicates": list, "error": str|None}
    """
    if parser_type == "qwen":
        import os as _os
        from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser
        parser = OnlineQwenGoalParser(
            model=_os.environ.get("QWEN_MODEL", "qwen-plus"),
            api_base=_os.environ.get(
                "QWEN_API_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=30,
        )
        result = parser.parse(
            instruction=instruction,
            task_type=task_type,
            objects=objects,
            n_blocks=n_blocks,
        )
    else:
        from vlm_nsm.core.goal_parser import MockGoalParser
        parser = MockGoalParser()
        result = parser.parse(
            instruction=instruction,
            task_type=task_type,
            objects=objects,
            n_blocks=n_blocks,
        )
    if not result.success:
        print(f"  Parse error: {result.error}")
    return {
        "success": result.success,
        "goal_predicates": (result.goal_json.get("goal_predicates", [])
                            if result.goal_json else []),
        "error": str(result.error) if result.error else None,
    }


# ── OperatorMonitor diagnosis ──────────────────────────────────────────────

def diagnose_failure(env, task: Any, action_idx: int,
                     failure_type: str | None) -> dict[str, Any]:
    """Run OperatorMonitor to diagnose and classify a step failure.

    Returns diagnosis dict with recovery suggestion.
    """
    try:
        from vlm_nsm.core.physical_predicate_extractor import (
            physical_state_to_predicates,
        )
        from vlm_nsm.neuro_symbolic.operator_monitor import OperatorMonitor
        from vlm_nsm.skills.skill_types import SkillCommand

        monitor = OperatorMonitor()
        all_poses = env.get_all_object_poses()
        current_preds = physical_state_to_predicates(
            all_poses, dict(all_poses),
            thresholds={"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05},
        )

        # Build a rough SkillCommand for the current step
        sc = SkillCommand(
            skill_name="pick_place",
            object_id=f"cube{1 + (action_idx % task.n_objects)}",
            target_object_id=f"cube{1 + ((action_idx + 1) % task.n_objects)}",
            relation="on",
        )

        op_failure, recovery = monitor.classify_failure(
            command=sc,
            low_level_failure=failure_type or "unknown",
        )

        return {
            "diagnosis": str(op_failure) if op_failure else "unclassified",
            "recovery": str(recovery) if recovery else "replan",
            "predicates": [str(p) for p in list(current_preds)[:5]],
        }
    except Exception as e:
        return {"diagnosis": f"monitor_error:{e}", "recovery": "unknown"}


# ── Single episode ─────────────────────────────────────────────────────────

def run_e2e_episode(env, pid, task, parser_type: str,
                    instruction: str) -> dict[str, Any]:
    """Run one complete E2E episode: Language → Plan → PID → Monitor.

    Args:
        env: RobosuiteEnvWrapper, already created and reset.
        pid: OperationalSpacePID.
        task: TaskSpec.
        parser_type: "mock" or "qwen".
        instruction: Natural language instruction string.

    Returns:
        Dict with episode metrics.
    """
    # ── Step 1: Language → Goal ──
    task_type = "hanoi" if "hanoi" in task.name else "stacking"
    objects = [f"cube{i}" for i in range(1, task.n_objects + 1)]

    parse = parse_instruction(
        instruction=instruction,
        task_type=task_type,
        parser_type=parser_type,
        objects=objects,
        n_blocks=task.n_objects,
    )

    if not parse["success"]:
        return {
            "success": False,
            "stage": "parse",
            "error": f"parse_failed: {parse.get('error', 'unknown')}",
            "num_steps": 0,
        }

    print(f"  Parsed: {parse['goal_predicates']}")

    # Override task goal with parsed result
    task.goal_predicates = parse["goal_predicates"]
    task.language_instructions = [instruction]

    # ── Step 2-4: Plan → PID → Monitor (via run_task) ──
    env.env.horizon = task.horizon  # ensure env has enough budget
    from vlm_nsm.embodied.task_runner import run_task

    result = run_task(env, task, pid)

    # ── Step 5: OperatorMonitor on failure ──
    diagnosis = None
    if not result.get("goal_success", result.get("success", False)):
        diagnosis = diagnose_failure(
            env, task,
            action_idx=0,
            failure_type=result.get("failure_type"),
        )

    return {
        "success": result.get("goal_success", result.get("success", False)),
        "stage": "execution",
        "num_steps": result.get("num_steps", 0),
        "failure_type": result.get("failure_type"),
        "parse_goal": parse["goal_predicates"],
        "monitor_diagnosis": diagnosis,
    }


# ── Multi-episode evaluation ───────────────────────────────────────────────

def run_e2e_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run multi-episode E2E evaluation."""

    # Import heavy deps lazily
    import vlm_nsm.executors.stack_three_cube_env as _  # noqa: Hanoi monkey-patch
    from vlm_nsm.core.task_spec import get_stacking_2block_spec
    from vlm_nsm.embodied.controllers.operational_space_pid import (
        OperationalSpacePID, GainSchedule,
    )
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    # ── Build TaskSpec ──
    if args.task == "stacking_2block":
        task = get_stacking_2block_spec()
        task.horizon = 2000  # ensure enough budget for PID episodes
    elif args.task == "hanoi_3disk":
        # Hanoi: use internal_hanoi planner, 3 disks
        from vlm_nsm.core.task_spec import TaskSpec
        task = TaskSpec(
            name="hanoi_3disk",
            env_name="Stack",
            env_kwargs={"robots": "Panda"},
            planner="internal_hanoi",
            n_objects=3,
            skills=["reach", "grasp", "lift", "reach_place", "release_place", "retreat"],
            goal_predicates=[],
            target_stack=["cube3", "cube2", "cube1"],
            horizon=20000,
            controller="pid",
            monitor_enabled=True,
            max_replans=3,
            language_instructions=[args.instruction],
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # ── Create env ──
    env = RobosuiteEnvWrapper(
        env_name=task.env_name,
        robots=task.env_kwargs.get("robots", "Panda"),
        cube_names=[f"cube{i}" for i in range(1, task.n_objects + 1)],
        render_mode="offscreen",
    )

    # ── PID controller ──
    pid = OperationalSpacePID(gain_schedule=GainSchedule())

    # ── TaskComposer experimental path ──
    if getattr(args, 'composer', False):
        print("[composer] TaskComposer experimental mode enabled")
        from vlm_nsm.core.predicate_scene_graph import build_predicate_scene_graph
        from vlm_nsm.neuro_symbolic.verifier_bank import standard_verifier_bank
        from vlm_nsm.language.task_composer import TaskComposer
        from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser

        objects = [f"cube{i}" for i in range(1, task.n_objects + 1)] + ["table"]
        verifier_bank = standard_verifier_bank(objects)
        qwen = OnlineQwenGoalParser(model="qwen-plus",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
        composer = TaskComposer(qwen, verifier_bank)

        # Build scene graph from env
        env.reset()
        scene_graph = build_predicate_scene_graph(
            env, [f"cube{i}" for i in range(1, task.n_objects + 1)],
            task_name=task.name,
        )
        print(f"[composer] Scene graph built: {len(scene_graph.objects)} objects, "
              f"{len(scene_graph.predicates)} predicates")

        # Compose task program
        result = composer.compose(
            instruction=args.instruction,
            scene_graph=scene_graph,
            objects=objects,
        )
        print(f"[composer] Mode: {result.mode}")
        if result.mode == "full":
            print(f"[composer] TaskProgram: {len(result.task_program.steps)} steps, "
                  f"verified: {result.verification_report.passed}")
            task.language_instructions = [args.instruction]
            # Add composer metadata to task
            task.execution_config["composer_program"] = result.task_program.to_dict() \
                if hasattr(result.task_program, 'to_dict') else {}
            task.execution_config["composer_mode"] = "full"
        elif result.mode == "goal_only":
            print("[composer] Falling back to goal-only mode")
            task.execution_config["composer_mode"] = "goal_only"
        else:
            print(f"[composer] Failed: {result.error}")

    # ── Run episodes ──
    results = []
    successes = 0
    for ep in range(args.num_episodes):
        ep_result = run_e2e_episode(
            env=env, pid=pid, task=task,
            parser_type=args.parser_type,
            instruction=args.instruction,
        )
        results.append(ep_result)
        if ep_result["success"]:
            successes += 1

        status = "OK" if ep_result["success"] else "FAIL"
        diag = ""
        if ep_result.get("monitor_diagnosis"):
            diag = f" | diagnosis={ep_result['monitor_diagnosis']['diagnosis']}"
        print(f"  [{ep+1}/{args.num_episodes}] {status} "
              f"steps={ep_result['num_steps']} stage={ep_result['stage']}{diag}")

        pid.reset()
        env.reset()

    # ── Summary ──
    success_rate = successes / args.num_episodes if args.num_episodes > 0 else 0
    summary = {
        "task": args.task,
        "instruction": args.instruction,
        "parser_type": args.parser_type,
        "num_episodes": args.num_episodes,
        "success_rate": success_rate,
        "success_count": successes,
        "mean_steps": float(np.mean([r["num_steps"] for r in results])),
        "parse_failures": sum(1 for r in results if r["stage"] == "parse"),
        "execution_failures": sum(
            1 for r in results if r["stage"] == "execution" and not r["success"]
        ),
    }

    # ── Save ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "episodes.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "monitor_diagnosis"}
                     for r in results], indent=2, ensure_ascii=False))

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E2E Neuro-Symbolic Pipeline: Language → Plan → PID → Monitor")

    parser.add_argument("--instruction", required=True,
                        help="Natural language instruction")
    parser.add_argument("--task", default="stacking_2block",
                        choices=["stacking_2block", "hanoi_3disk"])
    parser.add_argument("--parser_type", default="mock",
                        choices=["mock", "qwen"])
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--output_dir", default="generated/results/e2e_pid_pipeline")
    parser.add_argument("--composer", action="store_true",
                        help="Use TaskComposer experimental path (DSL+Verifier) "
                             "instead of goal-only Qwen→BFS path")
    args = parser.parse_args()

    print(f"E2E Pipeline: '{args.instruction}' → {args.task}")
    print(f"Parser: {args.parser_type}, Episodes: {args.num_episodes}")
    print()

    summary = run_e2e_eval(args)

    print(f"\n{'='*50}")
    print(f"E2E Pipeline Complete:")
    print(f"  Task:      {summary['task']}")
    print(f"  Success:   {summary['success_count']}/{summary['num_episodes']} "
          f"({summary['success_rate']:.0%})")
    print(f"  Steps:     {summary['mean_steps']:.0f} (mean)")
    print(f"  Parse OK:  {summary['num_episodes'] - summary['parse_failures']}"
          f"/{summary['num_episodes']}")
    print(f"  Saved to:  {args.output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
