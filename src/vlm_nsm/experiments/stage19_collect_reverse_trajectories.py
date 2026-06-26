"""Stage 19-D3-2: Collect per-step reverse stacking trajectories.

Re-runs successful green_on_red seeds from D2-3 with DemonstrationRecorder
to save full per-step obs/action trajectories, matching forward demo format.

CLI:
  MUJOCO_GL=egl python -m vlm_nsm.experiments.stage19_collect_reverse_trajectories \
    --episodes generated/results/stage_19/stage19_d2_reverse_collection_episodes.jsonl \
    --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
    --output_dir generated/datasets/stage_19/reverse_stack_demos_v0_1_trajectories \
    --target_success_demos 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder
from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor
from vlm_nsm.learning.robosuite_schema import ACTION_DIM, OBS_DIM
from vlm_nsm.skills.skill_types import SkillCommand


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p


def load_successful_episodes(episodes_path: Path) -> list[dict]:
    """Load successful reverse episodes, sorted by seed."""
    episodes: list[dict] = []
    with open(episodes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep.get("task_success") is True:
                episodes.append(ep)
    episodes.sort(key=lambda e: e.get("seed", 0))
    return episodes


def load_execution_config(config_path: str | None) -> dict[str, Any]:
    path = _resolve_path(config_path) if config_path else None
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def episode_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "kp", "max_delta", "use_place_state_machine", "horizon_aware",
        "place_retry", "grasp_close_steps", "grasp_max_retries",
        "place_max_retries", "place_mode", "skip_retreat_if_goal_satisfied",
        "holding_check_enabled", "lift_early_exit_enabled",
        "close_steps_min", "close_steps_max", "close_stabilize_steps",
        "release_steps_min", "release_steps_max", "post_release_stable_steps",
        "close_mode", "close_steps_fast",
        "fallback_extend_close_if_not_holding", "early_stop_on_goal", "horizon",
    ]
    return {key: config[key] for key in allowed if key in config}


def build_reverse_command() -> SkillCommand:
    return SkillCommand(
        skill_name="pick_place",
        object_id="cube2",
        target_object_id="cube1",
        source="table",
        relation="on",
        metadata={"task": "two_block_stack", "direction": "reverse", "teacher": "scripted_only"},
    )


def capture_scene(env: Any, cubes: list[str]) -> dict[str, Any]:
    scene: dict[str, Any] = {}
    try:
        scene["object_poses"] = env.get_all_object_poses()
    except Exception:
        scene["object_poses"] = {}
    try:
        scene["eef_pose"] = env.get_eef_pose()
    except Exception:
        scene["eef_pose"] = {}
    try:
        scene["gripper_state"] = env.get_gripper_state()
    except Exception:
        scene["gripper_state"] = {}
    scene["cubes"] = list(cubes)
    return scene


def add_skill_fields(
    traj_data: dict[str, np.ndarray],
    command: SkillCommand,
    episode_index: int,
    success: bool,
    direction: str = "reverse",
) -> dict[str, np.ndarray]:
    output = dict(traj_data)
    steps = len(output.get("action", output.get("obs", [])))
    output["skill_name"] = np.array([command.skill_name] * steps)
    output["skill_object_id"] = np.array([command.object_id] * steps)
    output["skill_target_object_id"] = np.array([command.target_object_id or ""] * steps)
    output["skill_relation"] = np.array([command.relation or ""] * steps)
    output["skill_source"] = np.array([command.source or ""] * steps)
    output["skill_demo_episode_index"] = np.full((steps,), episode_index, dtype=np.int32)
    output["skill_demo_success"] = np.full((steps,), bool(success), dtype=bool)
    return output


def save_trajectory_npz(
    path: Path,
    traj_data: dict[str, np.ndarray],
    command: SkillCommand,
    episode_index: int,
    success: bool,
) -> Path | None:
    if traj_data is None:
        return None
    labelled = add_skill_fields(traj_data, command, episode_index, success)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **labelled)
    return path


def build_manifest_record(
    episode_id: str,
    command: SkillCommand,
    traj_path: str | None,
    seed: int,
    success: bool,
    num_steps: int,
    failure_type: str | None,
    initial_scene: dict | None = None,
    final_scene: dict | None = None,
) -> dict[str, Any]:
    return _json_safe({
        "episode_id": episode_id,
        "skill_name": command.skill_name,
        "object_id": command.object_id,
        "target_object_id": command.target_object_id,
        "relation": command.relation,
        "source": command.source,
        "direction": "cube2_to_cube1",
        "goal_id": "green_on_red",
        "success": success,
        "num_steps": num_steps,
        "failure_type": failure_type,
        "trajectory_path": traj_path,
        "seed": seed,
        "teacher_mode": "scripted_only",
        "initial_scene": initial_scene or {},
        "final_scene": final_scene or {},
    })


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def collect_reverse_trajectories(
    success_seeds: list[int],
    output_dir: Path,
    config_path: str,
    *,
    target_count: int = 50,
) -> dict[str, Any]:
    """Collect per-step reverse trajectories for the given seeds."""
    config_data = load_execution_config(config_path)
    ep_kwargs = episode_kwargs_from_config(config_data)
    ep_kwargs.setdefault("horizon", 1000)
    ep_kwargs.setdefault("early_stop_on_goal", True)

    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    cubes = ["cube1", "cube2"]
    command = build_reverse_command()
    executor = ScriptedSkillExecutor(ep_kwargs)

    manifest_records: list[dict[str, Any]] = []
    collected = 0
    start_time = time.time()

    seeds_to_run = success_seeds[:target_count]
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        cube_names=cubes,
        render_mode="none",
    )

    try:
        for ep_idx, seed in enumerate(seeds_to_run):
            print(f"[{ep_idx + 1}/{len(seeds_to_run)}] seed={seed}", end=" ", flush=True)

            try:
                env.env.seed(seed)
            except Exception:
                pass
            env.reset()

            recorder = DemonstrationRecorder(
                output_dir=trajectory_dir,
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
            )
            initial_scene = capture_scene(env, cubes)

            context = {
                "cubes": cubes,
                "goal_predicates": [["on", "cube2", "cube1"]],
                "recorder": recorder,
            }
            result = executor.execute(command, env, context=context)

            traj_data = recorder.pop_last_episode_data()
            episode_id = f"rev_ep_{seed:06d}"

            traj_path: str | None = None
            if traj_data is not None and result.success:
                rel_path = f"trajectories/{episode_id}.npz"
                abs_path = output_dir / rel_path
                save_trajectory_npz(abs_path, traj_data, command, collected, result.success)
                traj_path = rel_path
                collected += 1

            final_scene = capture_scene(env, cubes)
            record = build_manifest_record(
                episode_id=episode_id,
                command=command,
                traj_path=traj_path,
                seed=seed,
                success=result.success,
                num_steps=result.steps,
                failure_type=result.failure_type,
                initial_scene=initial_scene,
                final_scene=final_scene,
            )
            manifest_records.append(record)

            status = "OK" if result.success else f"FAIL({result.failure_type})"
            print(f"success={result.success} steps={result.steps} [{status}]")

            if collected >= target_count:
                print(f"Reached target of {target_count} successful trajectories.")
                seeds_to_run = seeds_to_run[:ep_idx + 1]
                break

    finally:
        try:
            env.env.close()
        except Exception:
            pass

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write summary
    successes = sum(1 for r in manifest_records if r["success"])
    total = len(manifest_records)
    meta = {
        "dataset_name": "reverse_stack_demos_v0_1",
        "dataset_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "direction": "cube2_to_cube1",
        "goal_id": "green_on_red",
        "teacher_mode": "scripted_only",
        "num_episodes": total,
        "success_count": successes,
        "success_rate": successes / max(total, 1),
        "trajectories_saved": collected,
        "manifest_path": "manifest.jsonl",
        "trajectory_dir": "trajectories",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "config_path": config_path,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    meta_path = output_dir / "summary.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nReverse trajectory collection complete:")
    print(f"  Episodes: {total}, Success: {successes}")
    print(f"  Trajectories saved: {collected}")
    print(f"  Output: {output_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="D3-2: Collect reverse stacking trajectories")
    parser.add_argument("--episodes", required=True,
                        help="Path to D2-3 reverse collection episodes JSONL")
    parser.add_argument("--execution_config", required=True,
                        help="Path to execution config JSON")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for trajectories and manifest")
    parser.add_argument("--target_success_demos", type=int, default=50,
                        help="Target number of successful trajectories (default: 50)")
    args = parser.parse_args()

    episodes_path = Path(args.episodes)
    output_dir = Path(args.output_dir)

    if not episodes_path.exists():
        print(f"ERROR: Episodes file not found: {episodes_path}")
        return

    print(f"Loading successful episodes from: {episodes_path}")
    episodes = load_successful_episodes(episodes_path)
    success_seeds = [ep["seed"] for ep in episodes]
    print(f"  Found {len(success_seeds)} successful episodes")

    if len(success_seeds) < args.target_success_demos:
        print(f"  WARNING: Only {len(success_seeds)} successes available (target: {args.target_success_demos})")
        print(f"  Will collect up to {len(success_seeds)} trajectories.")

    collect_reverse_trajectories(
        success_seeds=success_seeds,
        output_dir=output_dir,
        config_path=args.execution_config,
        target_count=min(args.target_success_demos, len(success_seeds)),
    )


if __name__ == "__main__":
    main()
