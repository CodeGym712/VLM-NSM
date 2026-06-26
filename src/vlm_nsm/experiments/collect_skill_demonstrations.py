"""Collect skill-labeled demonstrations with the skill executor interface.

This script uses the existing controller stack as a teacher and writes:

- ``manifest.jsonl``: one skill-level record per episode
- ``trajectories/*.npz``: recorded observation/action trajectories
- ``dataset_meta.json``: aggregate metadata

It does not call DashScope and does not modify controller checkpoints/configs.
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

from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder
from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor
from vlm_nsm.learning.robosuite_schema import ACTION_DIM, OBS_DIM
from vlm_nsm.skills.skill_types import SkillCommand, SkillExecutionResult


DEFAULT_CONFIG = "generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json"
DEFAULT_CHECKPOINT = "generated/results/stage_7B-4/dagger_reach_bc/best.pt"
DEFAULT_ACTION_NORM = "generated/results/stage_7B-1/action_norm_stats.json"

PHASE_TO_SKILL_ID = {
    "reach_pick": 0,
    "grasp": 1,
    "lift": 2,
    "reach_place": 3,
    "release": 4,
    "retreat": 5,
}


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_execution_config(config_path: str | None) -> dict[str, Any]:
    """Load controller execution config, returning an empty dict if omitted."""
    path = _resolve_project_path(config_path)
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def episode_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Filter execution config keys accepted by the Robosuite episode runner."""
    allowed = [
        "kp",
        "max_delta",
        "use_place_state_machine",
        "horizon_aware",
        "place_retry",
        "grasp_close_steps",
        "grasp_max_retries",
        "place_max_retries",
        "place_mode",
        "skip_retreat_if_goal_satisfied",
        "holding_check_enabled",
        "lift_early_exit_enabled",
        "close_steps_min",
        "close_steps_max",
        "close_stabilize_steps",
        "release_steps_min",
        "release_steps_max",
        "post_release_stable_steps",
        "close_mode",
        "close_steps_fast",
        "fallback_extend_close_if_not_holding",
        "early_stop_on_goal",
        "horizon",
    ]
    return {key: config[key] for key in allowed if key in config}


def default_skill_command(task: str = "two_block_stack") -> SkillCommand:
    if task != "two_block_stack":
        raise ValueError(f"Unsupported task for initial skill demos: {task}")
    return SkillCommand(
        skill_name="pick_place",
        object_id="cube1",
        target_object_id="cube2",
        source="table",
        relation="on",
        metadata={"task": task, "teacher": "scripted_skill_executor"},
    )


def capture_scene(env: Any, cubes: list[str]) -> dict[str, Any]:
    """Capture lightweight scene state from a RobosuiteEnvWrapper-like env."""
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


def add_skill_fields_to_trajectory(
    traj_data: dict[str, np.ndarray],
    command: SkillCommand,
    episode_index: int,
    success: bool,
) -> dict[str, np.ndarray]:
    """Add skill-level labels to a DemonstrationRecorder trajectory dict."""
    output = dict(traj_data)
    steps = _trajectory_length(output)
    output["skill_name"] = np.array([command.skill_name] * steps)
    output["skill_object_id"] = np.array([command.object_id] * steps)
    output["skill_target_object_id"] = np.array([command.target_object_id or ""] * steps)
    output["skill_relation"] = np.array([command.relation or ""] * steps)
    output["skill_source"] = np.array([command.source or ""] * steps)
    output["skill_demo_episode_index"] = np.full((steps,), episode_index, dtype=np.int32)
    output["skill_demo_success"] = np.full((steps,), bool(success), dtype=bool)
    return output


def _trajectory_length(traj_data: dict[str, np.ndarray]) -> int:
    for key in ("action", "obs", "phase_id", "step_id"):
        value = traj_data.get(key)
        if value is not None:
            return int(len(value))
    return 0


def save_trajectory(
    path: Path,
    traj_data: dict[str, np.ndarray] | None,
    command: SkillCommand,
    episode_index: int,
    success: bool,
) -> Path | None:
    """Save one trajectory if data exists; return the path or None."""
    if traj_data is None:
        return None
    labelled = add_skill_fields_to_trajectory(traj_data, command, episode_index, success)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **labelled)
    return path


def build_manifest_record(
    *,
    episode_id: str,
    command: SkillCommand,
    result: SkillExecutionResult,
    trajectory_path: str | None,
    seed: int,
    initial_scene: dict[str, Any] | None = None,
    final_scene: dict[str, Any] | None = None,
    teacher_mode: str = "hybrid_reach_pick",
) -> dict[str, Any]:
    """Build a manifest record for one skill demonstration."""
    return _json_safe({
        "episode_id": episode_id,
        "skill_name": command.skill_name,
        "object_id": command.object_id,
        "target_object_id": command.target_object_id,
        "relation": command.relation,
        "source": command.source,
        "success": result.success,
        "num_steps": result.steps,
        "failure_type": result.failure_type,
        "trajectory_path": trajectory_path,
        "seed": seed,
        "teacher_mode": teacher_mode,
        "initial_scene": initial_scene or {},
        "final_scene": final_scene or {},
        "metrics": {
            key: value
            for key, value in result.metrics.items()
            if key != "raw_result"
        },
    })


def _json_safe(value: Any) -> Any:
    """Convert numpy values and nested structures into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_dataset_meta(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    task: str,
    teacher_mode: str,
    seed: int,
    config_path: str | None,
    checkpoint_path: str | None,
    record_rgb: bool = False,
) -> dict[str, Any]:
    successes = sum(1 for r in records if r["success"])
    total = len(records)
    meta = {
        "dataset_name": "skill_demos_v0_1",
        "dataset_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "teacher_mode": teacher_mode,
        "num_episodes": total,
        "success_count": successes,
        "success_rate": successes / max(total, 1),
        "seed": seed,
        "manifest_path": "manifest.jsonl",
        "trajectory_dir": "trajectories",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path if teacher_mode != "scripted_only" else None,
        "record_rgb": record_rgb,
        "notes": "Skill-labeled demos collected through ScriptedSkillExecutor.",
    }
    (output_dir / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def _load_hybrid_policy(
    checkpoint_path: str,
    action_norm_path: str,
    device: str,
    bc_phases: list[str],
):
    """Load DAgger BC reach policy and return policy context fields."""
    import torch

    from vlm_nsm.learning.eval_bc import load_checkpoint
    from vlm_nsm.learning.eval_bc_hybrid import _get_bc_action

    torch_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model, meta = load_checkpoint(checkpoint_path, torch_device)
    model.eval()

    action_mean = None
    action_std = None
    norm_path = _resolve_project_path(action_norm_path)
    if norm_path and norm_path.exists():
        stats = json.loads(norm_path.read_text())
        action_mean = np.array(stats["action_mean"], dtype=np.float32)
        action_std = np.array(stats["action_std"], dtype=np.float32)

    condition_on_phase = bool(meta.get("condition_on_phase", False))
    if not condition_on_phase and int(meta.get("obs_dim", OBS_DIM)) > OBS_DIM:
        condition_on_phase = True

    def policy_fn(obs, phase_id):
        obs_in = obs
        if condition_on_phase:
            one_hot = np.zeros(6, dtype=np.float32)
            one_hot[phase_id] = 1.0
            obs_in = np.concatenate([obs, one_hot])
        return _get_bc_action(model, obs_in, torch_device, action_mean, action_std, [3, 4, 5])

    return {
        "policy_override": policy_fn,
        "override_phases": {PHASE_TO_SKILL_ID[p] for p in bc_phases},
    }


def collect_skill_demonstrations(
    *,
    output_dir: str,
    num_episodes: int,
    task: str = "two_block_stack",
    executor: str = "scripted",
    seed: int = 0,
    save_trajectories: bool = True,
    teacher_mode: str = "hybrid_reach_pick",
    checkpoint: str = DEFAULT_CHECKPOINT,
    config: str = DEFAULT_CONFIG,
    action_norm: str = DEFAULT_ACTION_NORM,
    device: str = "cuda",
    bc_phases: list[str] | None = None,
    max_steps: int | None = None,
    record_rgb: bool = False,
) -> dict[str, Any]:
    """Collect skill-labeled Robosuite demonstrations."""
    if executor != "scripted":
        raise ValueError(f"Unsupported executor: {executor}")
    if task != "two_block_stack":
        raise ValueError(f"Unsupported task: {task}")
    if record_rgb:
        raise NotImplementedError("record_rgb is reserved for a later RGB-conditioned policy stage")

    random.seed(seed)
    np.random.seed(seed)
    bc_phases = bc_phases or ["reach_pick"]

    output_path = _resolve_project_path(output_dir)
    assert output_path is not None
    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_path / "trajectories"
    if save_trajectories:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    config_data = load_execution_config(config)
    episode_kwargs = episode_kwargs_from_config(config_data)
    episode_kwargs.setdefault("horizon", 1000)
    if max_steps is not None:
        episode_kwargs["horizon"] = int(max_steps)
    episode_kwargs.setdefault("early_stop_on_goal", True)

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    cubes = ["cube1", "cube2"]
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        cube_names=cubes,
        render_mode="none",
    )

    policy_context: dict[str, Any] = {}
    if teacher_mode == "hybrid_reach_pick":
        policy_context = _load_hybrid_policy(checkpoint, action_norm, device, bc_phases)
    elif teacher_mode != "scripted_only":
        raise ValueError(f"Unsupported teacher_mode: {teacher_mode}")

    executor_obj = ScriptedSkillExecutor(episode_kwargs)
    command = default_skill_command(task)
    manifest_records: list[dict[str, Any]] = []
    start_time = time.time()

    try:
        for ep_idx in range(num_episodes):
            ep_seed = seed + ep_idx
            if hasattr(env.env, "seed") and callable(env.env.seed):
                env.env.seed(ep_seed)
            env.reset()

            recorder = (
                DemonstrationRecorder(
                    output_dir=trajectory_dir,
                    obs_dim=OBS_DIM,
                    action_dim=ACTION_DIM,
                )
                if save_trajectories
                else None
            )
            initial_scene = capture_scene(env, cubes)

            context = {
                "cubes": cubes,
                "goal_predicates": [["on", command.object_id, command.target_object_id]],
                "recorder": recorder,
                **policy_context,
            }
            result = executor_obj.execute(command, env, context=context)

            traj_data = recorder.pop_last_episode_data() if recorder is not None else None
            traj_path = None
            if save_trajectories:
                rel_path = Path("trajectories") / f"ep_{ep_idx + 1:06d}.npz"
                saved_path = save_trajectory(
                    output_path / rel_path,
                    traj_data,
                    command,
                    ep_idx,
                    result.success,
                )
                traj_path = str(rel_path) if saved_path is not None else None

            final_scene = capture_scene(env, cubes)
            episode_id = f"ep_{ep_idx + 1:06d}"
            record = build_manifest_record(
                episode_id=episode_id,
                command=command,
                result=result,
                trajectory_path=traj_path,
                seed=ep_seed,
                initial_scene=initial_scene,
                final_scene=final_scene,
                teacher_mode=teacher_mode,
            )
            manifest_records.append(record)
            print(
                f"[{ep_idx + 1}/{num_episodes}] {episode_id} "
                f"success={result.success} steps={result.steps} failure={result.failure_type}"
            )
    finally:
        env.env.close()

    write_manifest(manifest_records, output_path / "manifest.jsonl")
    meta = write_dataset_meta(
        output_path,
        manifest_records,
        task=task,
        teacher_mode=teacher_mode,
        seed=seed,
        config_path=config,
        checkpoint_path=checkpoint,
        record_rgb=record_rgb,
    )
    meta["elapsed_seconds"] = round(time.time() - start_time, 2)
    (output_path / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSkill demo collection complete: {meta['success_count']}/{meta['num_episodes']} successful")
    print(f"Output: {output_path}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Collect skill-labeled demonstrations")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--task", default="two_block_stack", choices=["two_block_stack"])
    parser.add_argument("--executor", default="scripted", choices=["scripted"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_trajectories", action="store_true", default=False)
    parser.add_argument("--teacher_mode", default="hybrid_reach_pick",
                        choices=["hybrid_reach_pick", "scripted_only"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--action_norm", default=DEFAULT_ACTION_NORM)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bc_phases", nargs="+", default=["reach_pick"],
                        choices=list(PHASE_TO_SKILL_ID))
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override episode horizon for each collected skill")
    parser.add_argument("--record_rgb", action="store_true", default=False,
                        help="Reserved for future RGB-conditioned skill datasets")
    args = parser.parse_args()

    collect_skill_demonstrations(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        task=args.task,
        executor=args.executor,
        seed=args.seed,
        save_trajectories=args.save_trajectories,
        teacher_mode=args.teacher_mode,
        checkpoint=args.checkpoint,
        config=args.config,
        action_norm=args.action_norm,
        device=args.device,
        bc_phases=args.bc_phases,
        max_steps=args.max_steps,
        record_rgb=args.record_rgb,
    )


if __name__ == "__main__":
    main()
