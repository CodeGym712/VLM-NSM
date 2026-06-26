"""Collect robosuite demonstration data for downstream training.

Saves per-episode npz files categorized by success type:
  - strict_success: clean execution + goal achieved, no timeout
  - semantic_only_success: goal achieved but not clean execution
  - all_semantic_success: all episodes where goal was achieved (copy of strict+semantic_only)
  - failure_debug: goal not achieved

Usage:
    MUJOCO_GL=egl python3 -m vlm_nsm.experiments.collect_robosuite_demos \\
        --num_episodes 50 --output_dir generated/demos/robosuite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _try_import_robosuite():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper  # noqa: F401
        return True
    except ImportError:
        return False


def _capture_frame(env, cubes: list[str]) -> dict:
    """Capture full state snapshot from the environment."""
    frame = {}
    frame["object_poses"] = env.get_all_object_poses()
    frame["eef_pose"] = env.get_eef_pose()
    gripper = env.get_gripper_state()
    frame["gripper_pos"] = gripper["pos"]
    frame["gripper_open"] = gripper["open"]
    frame["gripper_has_object"] = gripper.get("has_object", False)
    return frame


def collect_demos(
    num_episodes: int = 50,
    output_dir: str = "generated/demos/robosuite",
    early_stop_on_goal: bool = True,
    seed_offset: int = 0,
    kp: float = 0.05,
    max_delta: float = 0.02,
    # Stage 7A-10/11: forwarded to _run_robosuite_episode
    reach_threshold: float | None = None,
    lift_height: float | None = None,
    release_open_steps: int | None = None,
    skip_retreat_if_goal_satisfied: bool = False,
    terminate_after_safe_release: bool = False,
    grasp_close_steps: int | None = None,
    grasp_max_retries: int | None = None,
    place_max_retries: int | None = None,
    place_mode: str = "simple",
    holding_check_enabled: bool = False,
    close_steps_min: int | None = None,
    close_steps_max: int | None = None,
    close_stabilize_steps: int | None = None,
    lift_early_exit_enabled: bool = False,
    release_steps_min: int | None = None,
    release_steps_max: int | None = None,
    post_release_stable_steps: int | None = None,
    # Stage 7B-1: optional demonstration recorder for BC training
    recorder=None,
    # Stage 7A-12: config-specified close mode
    close_mode: str = "fixed",
    close_steps_fast: int | None = None,
    fallback_extend_close_if_not_holding: bool = False,
    # Stage 7A-13: collect until strict success count reached
    collect_until_strict: int | None = None,
) -> dict:
    """Run episodes and save demonstration data.

    Args:
        num_episodes: Number of episodes to collect.
        output_dir: Output directory for npz files.
        early_stop_on_goal: Whether to early-stop when goal achieved.
        seed_offset: Random seed offset for reproducibility.
        kp: Proportional gain for position control.
        max_delta: Max position delta per step.
        collect_until_strict: If set, keep collecting until this many strict
            successes are obtained (overrides num_episodes as a soft max).
        **kwargs: Additional parameters forwarded to _run_robosuite_episode.

    Returns:
        Summary dict with collection stats.
    """
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.core.data_types import Predicate
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
        _run_robosuite_episode, build_stacking_goal,
    )

    cubes = ["cube1", "cube2"]
    target_stack = cubes[::-1]
    goal_predicates = build_stacking_goal(target_stack)

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        return {"error": f"Planner failed: {plan_result.error}"}

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
    horizon = env.env.horizon if hasattr(env.env, "horizon") else 1000

    out_path = ensure_dir(PROJECT_ROOT / output_dir)
    strict_count = 0
    semantic_only_count = 0
    failure_count = 0
    per_episode_summary: list[dict] = []

    # Stage 7A-13: collect until strict success count reached
    collect_target = collect_until_strict
    max_attempts = max(num_episodes, collect_target * 3) if collect_target else num_episodes
    total_attempts = 0

    for ep in range(max_attempts):
        # Early exit if we've reached target
        if collect_target is not None and strict_count >= collect_target:
            break

        # Set random seed for reproducibility
        if hasattr(env.env, "seed") and callable(env.env.seed):
            env.env.seed(seed_offset + ep)

        # Record initial frame
        env.reset()
        initial_frame = _capture_frame(env, cubes)

        # Run episode with frame capture at key steps
        result = _run_robosuite_episode(
            env, plan_result.actions, cubes, goal_predicates,
            horizon=horizon, early_stop_on_goal=early_stop_on_goal,
            use_place_state_machine=True,
            kp=kp, max_delta=max_delta,
            reach_threshold=reach_threshold,
            lift_height=lift_height,
            release_open_steps=release_open_steps,
            skip_retreat_if_goal_satisfied=skip_retreat_if_goal_satisfied,
            terminate_after_safe_release=terminate_after_safe_release,
            grasp_close_steps=grasp_close_steps,
            grasp_max_retries=grasp_max_retries,
            place_max_retries=place_max_retries,
            place_mode=place_mode,
            holding_check_enabled=holding_check_enabled,
            close_steps_min=close_steps_min,
            close_steps_max=close_steps_max,
            close_stabilize_steps=close_stabilize_steps,
            lift_early_exit_enabled=lift_early_exit_enabled,
            release_steps_min=release_steps_min,
            release_steps_max=release_steps_max,
            post_release_stable_steps=post_release_stable_steps,
            recorder=recorder,
            close_mode=close_mode,
            close_steps_fast=close_steps_fast,
            fallback_extend_close_if_not_holding=fallback_extend_close_if_not_holding,
        )

        # Capture final frame
        final_frame = _capture_frame(env, cubes)

        # Categorize
        strict_ok = result.get("strict_physical_success", False)
        semantic_ok = result.get("semantic_goal_success", False)
        num_steps = result.get("num_steps", 0)
        failure_type = result.get("failure_type", "success") or "success"

        record = {
            "episode": ep + 1,
            "strict_success": strict_ok,
            "semantic_success": semantic_ok,
            "failure_type": failure_type,
            "num_steps": num_steps,
            "timeout_type": result.get("timeout_type"),
            "early_stopped": result.get("early_stopped", False),
            "initial_cube1_z": initial_frame["object_poses"].get("cube1", {}).get("z", 0),
            "initial_cube2_z": initial_frame["object_poses"].get("cube2", {}).get("z", 0),
            "final_cube1_z": final_frame["object_poses"].get("cube1", {}).get("z", 0),
            "final_cube2_z": final_frame["object_poses"].get("cube2", {}).get("z", 0),
        }
        per_episode_summary.append(record)

        # Save npz with state data
        npz_data = {
            "initial_object_poses": np.array([list(v.values()) for v in initial_frame["object_poses"].values()]),
            "initial_eef": np.array(list(initial_frame["eef_pose"].values())),
            "initial_gripper_open": np.array([initial_frame["gripper_open"]]),
            "final_object_poses": np.array([list(v.values()) for v in final_frame["object_poses"].values()]),
            "final_eef": np.array(list(final_frame["eef_pose"].values())),
            "final_gripper_open": np.array([final_frame["gripper_open"]]),
            "strict_success": np.array([strict_ok]),
            "semantic_success": np.array([semantic_ok]),
            "failure_type": np.array([ord(c) for c in failure_type[:8].ljust(8)]),
            "num_steps": np.array([num_steps]),
        }

        # Stage 7B-2R: enrich per-episode npz with full trajectory data
        # (obs, action, phase_id) from the demonstration recorder
        if recorder is not None:
            traj_data = recorder.pop_last_episode_data()
            if traj_data is not None:
                # Merge trajectory keys (obs, action, phase_id, etc.)
                # without overwriting existing diagnostic fields
                for k, v in traj_data.items():
                    if k not in npz_data:
                        npz_data[k] = v

        # Determine primary category directory
        if strict_ok:
            cat_dir = ensure_dir(out_path / "strict_success")
            strict_count += 1
        elif semantic_ok:
            cat_dir = ensure_dir(out_path / "semantic_only_success")
            semantic_only_count += 1
        else:
            cat_dir = ensure_dir(out_path / "failure_debug")
            failure_count += 1

        npz_path = cat_dir / f"episode_{ep+1:04d}.npz"
        np.savez_compressed(npz_path, **npz_data)

        total_attempts += 1

    env.env.close()

    # Create all_semantic_success by copying from strict_success and semantic_only_success
    all_semantic_count = 0
    for src_dir in [out_path / "strict_success", out_path / "semantic_only_success"]:
        if src_dir.exists():
            dst_dir = ensure_dir(out_path / "all_semantic_success")
            for f in src_dir.iterdir():
                if f.suffix == ".npz":
                    dst_path = dst_dir / f.name
                    if not dst_path.exists():
                        import shutil
                        shutil.copy2(f, dst_path)
                    all_semantic_count += 1

    summary = {
        "output_dir": str(out_path),
        "num_episodes": num_episodes,
        "collect_until_strict": collect_target,
        "total_attempts": total_attempts,
        "strict_success_count": strict_count,
        "semantic_only_success_count": semantic_only_count,
        "all_semantic_success_count": all_semantic_count,
        "failure_count": failure_count,
        "strict_success_rate": strict_count / max(total_attempts, 1),
        "semantic_success_rate": (strict_count + semantic_only_count) / max(total_attempts, 1),
        "efficiency_ratio": strict_count / max(total_attempts, 1) if collect_target else None,
        "per_episode": per_episode_summary,
    }

    # Save summary JSON
    summary_path = out_path / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Collection summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect robosuite demonstration data")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="generated/demos/robosuite")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file for execution parameters")
    parser.add_argument("--kp", type=float, default=None,
                        help="Proportional gain (overrides config/module default)")
    parser.add_argument("--max_delta", type=float, default=None,
                        help="Max position delta per step (overrides config/module default)")
    parser.add_argument("--record", action="store_true",
                        help="Record full trajectory data for BC training")
    parser.add_argument("--record-output", type=str, default=None,
                        help="Output npz path for recorded trajectories")
    parser.add_argument("--collect-until-strict", type=int, default=None,
                        help="Keep collecting until this many strict successes (overrides --num_episodes)")
    args = parser.parse_args()

    if not _try_import_robosuite():
        print("ERROR: Robosuite not available.", file=sys.stderr)
        sys.exit(1)

    # Load config
    kp = 0.05
    max_delta = 0.02
    early_stop_on_goal = not args.no_early_stop
    # Stage 7A-10/11 params (defaults match module defaults)
    reach_threshold = None
    lift_height = None
    release_open_steps = None
    skip_retreat_if_goal_satisfied = False
    terminate_after_safe_release = False
    grasp_close_steps = None
    grasp_max_retries = None
    place_max_retries = None
    place_mode = "simple"
    holding_check_enabled = False
    close_steps_min = None
    close_steps_max = None
    close_stabilize_steps = None
    lift_early_exit_enabled = False
    release_steps_min = None
    release_steps_max = None
    post_release_stable_steps = None
    # Stage 7A-12: config-specified close mode params
    close_mode = "fixed"
    close_steps_fast = None
    fallback_extend_close_if_not_holding = False

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            import json as _json
            cfg = _json.loads(config_path.read_text())
            print(f"[collect] loaded config from {config_path}")
            kp = cfg.get("kp", kp)
            max_delta = cfg.get("max_delta", max_delta)
            import vlm_nsm.experiments.run_embodied_stacking_robosuite as _robosuite_mod
            if "reach_threshold" in cfg:
                _robosuite_mod._REACH_THRESHOLD = cfg["reach_threshold"]
                reach_threshold = cfg["reach_threshold"]
            if "lift_height" in cfg:
                _robosuite_mod._LIFT_HEIGHT = cfg["lift_height"]
                lift_height = cfg["lift_height"]
            if "release_open_steps" in cfg:
                _robosuite_mod._RELEASE_OPEN_STEPS = cfg["release_open_steps"]
                release_open_steps = cfg["release_open_steps"]
            early_stop_on_goal = cfg.get("early_stop_on_goal", early_stop_on_goal)
            skip_retreat_if_goal_satisfied = cfg.get("skip_retreat_if_goal_satisfied", skip_retreat_if_goal_satisfied)
            terminate_after_safe_release = cfg.get("terminate_after_safe_release", terminate_after_safe_release)
            grasp_close_steps = cfg.get("grasp_close_steps", grasp_close_steps)
            grasp_max_retries = cfg.get("grasp_max_retries", grasp_max_retries)
            place_max_retries = cfg.get("place_max_retries", place_max_retries)
            place_mode = cfg.get("place_mode", place_mode)
            holding_check_enabled = cfg.get("holding_check_enabled", holding_check_enabled)
            close_steps_min = cfg.get("close_steps_min", close_steps_min)
            close_steps_max = cfg.get("close_steps_max", close_steps_max)
            close_stabilize_steps = cfg.get("close_stabilize_steps", close_stabilize_steps)
            lift_early_exit_enabled = cfg.get("lift_early_exit_enabled", lift_early_exit_enabled)
            release_steps_min = cfg.get("release_steps_min", release_steps_min)
            release_steps_max = cfg.get("release_steps_max", release_steps_max)
            post_release_stable_steps = cfg.get("post_release_stable_steps", post_release_stable_steps)
            # Stage 7A-12: close mode from config
            close_mode = cfg.get("close_mode", close_mode)
            close_steps_fast = cfg.get("close_steps_fast", close_steps_fast)
            fallback_extend_close_if_not_holding = cfg.get("fallback_extend_close_if_not_holding",
                                                           fallback_extend_close_if_not_holding)

    if args.kp is not None:
        kp = args.kp
    if args.max_delta is not None:
        max_delta = args.max_delta

    # Stage 7B-1: create demonstration recorder for BC training data
    _recorder = None
    _record_output_path = None
    if args.record:
        from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder
        from vlm_nsm.learning.robosuite_schema import OBS_DIM, ACTION_DIM
        record_dir = ensure_dir(PROJECT_ROOT / "generated" / "demos" / "stage_7B-1")
        _recorder = DemonstrationRecorder(
            output_dir=record_dir,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
        )
        _record_output_path = str(record_dir / "robosuite_stage7b1_trajectories.npz")
        print(f"[BC] Recording full trajectories to {_record_output_path}")

    print(f"Collecting {args.num_episodes} robosuite demos (kp={kp}, max_delta={max_delta}, grasp_close_steps={grasp_close_steps})...")
    summary = collect_demos(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        early_stop_on_goal=early_stop_on_goal,
        seed_offset=args.seed_offset,
        kp=kp, max_delta=max_delta,
        reach_threshold=reach_threshold,
        lift_height=lift_height,
        release_open_steps=release_open_steps,
        skip_retreat_if_goal_satisfied=skip_retreat_if_goal_satisfied,
        terminate_after_safe_release=terminate_after_safe_release,
        grasp_close_steps=grasp_close_steps,
        grasp_max_retries=grasp_max_retries,
        place_max_retries=place_max_retries,
        place_mode=place_mode,
        holding_check_enabled=holding_check_enabled,
        close_steps_min=close_steps_min,
        close_steps_max=close_steps_max,
        close_stabilize_steps=close_stabilize_steps,
        lift_early_exit_enabled=lift_early_exit_enabled,
        release_steps_min=release_steps_min,
        release_steps_max=release_steps_max,
        post_release_stable_steps=post_release_stable_steps,
        recorder=_recorder,
        close_mode=close_mode,
        close_steps_fast=close_steps_fast,
        fallback_extend_close_if_not_holding=fallback_extend_close_if_not_holding,
        collect_until_strict=args.collect_until_strict,
    )

    # Stage 7B-2R: per-episode trajectory data is now saved via pop_last_episode_data()
    # inside collect_demos(). Only save combined npz if there's residual data.
    if _recorder is not None and _record_output_path is not None and _recorder.total_steps > 0:
        npz_path, meta_path = _recorder.save(
            filename=Path(_record_output_path).name,
            task="stacking",
            n_blocks=2,
            env_type="robosuite",
            controller_type="proportional",
            observation_type="raw",
            step_names=["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"],
        )
        print(f"[BC] Trajectories saved to {npz_path}")
        print(f"[BC] Metadata saved to {meta_path}")
        summary["recorded_trajectories_path"] = str(npz_path)

    if "error" in summary:
        print(f"ERROR: {summary['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DEMO COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {summary['num_episodes']}")
    print(f"Strict success:  {summary['strict_success_count']} ({summary['strict_success_rate']*100:.1f}%)")
    sem_only = summary.get("semantic_only_success_count", 0)
    all_sem = summary.get("all_semantic_success_count", 0)
    print(f"Semantic only:   {sem_only}")
    print(f"All semantic:    {all_sem}")
    print(f"Failures:        {summary['failure_count']}")
    print(f"Output dir:      {summary['output_dir']}")


if __name__ == "__main__":
    main()
