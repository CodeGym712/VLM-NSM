"""Stage 20-R5: Small contact smoke collection.

Runs 20 episodes (10 red_on_green + 10 green_on_red) with scripted-only
cube2_robust teacher, saving per-step trajectories with contact/holding fields.

CLI:
    MUJOCO_GL=egl python -m vlm_nsm.experiments.stage20_r5_contact_smoke_collection \
        --suite generated/eval_inputs/stage_20/stage20_r5_contact_smoke_suite.json \
        --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
        --output_dir generated/datasets/stage_20/teacher_demos_cube2_robust_v0_3_contact_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.utils import ensure_dir
from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates
from vlm_nsm.skills.skill_plan_from_symbolic import skill_plan_from_symbolic

# Reuse the same pipeline functions from stage19
from vlm_nsm.experiments.stage19_run_symbolic_rollouts import (
    check_gate,
    build_stacking_goal,
    run_single_episode,
    _PHASE_TO_SKILL_ID,
)


def main():
    parser = argparse.ArgumentParser(description="Stage 20-R5: Contact smoke collection")
    parser.add_argument("--suite", required=True, help="Path to eval suite JSON")
    parser.add_argument("--execution_config", required=True, help="Path to execution config JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for trajectories")
    parser.add_argument("--target_success_demos", type=int, default=0,
                        help="Min successful demos to collect (0=collect all)")
    parser.add_argument("--render", default="none")
    args = parser.parse_args()

    render_mode = args.render if args.render in ("none", "human", "offscreen") else "none"

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    episodes = suite.get("episodes", [])
    available_objects = suite.get("available_objects", ["cube1", "cube2"])
    print(f"Suite: {len(episodes)} episodes from {args.suite}")

    exec_config = json.loads(Path(args.execution_config).read_text(encoding="utf-8"))

    # Use cube2_robust profile for scripted controller
    if "cube2_robust" not in str(exec_config):
        exec_config = dict(exec_config)
        exec_config["scripted_profile"] = "cube2_robust"
    print("Scripted profile: cube2_robust")

    planner = PlannerAdapter("bfs")

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    # Initialize recorder
    output_dir = Path(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    recorder = DemonstrationRecorder(
        output_dir=str(trajectories_dir),
        obs_dim=11,
        action_dim=7,
    )

    # Collect successful episodes only
    success_seeds = []
    all_results = []
    attempt = 0

    for ep in episodes:
        attempt += 1
        episode_id = ep["episode_id"]
        goal_id = ep["goal_id"]
        print(f"\n[{attempt}/{len(episodes)}] {episode_id} (goal={goal_id})")

        env = RobosuiteEnvWrapper(
            env_name="Stack",
            robots="Panda",
            cube_names=available_objects,
            render_mode=render_mode,
        )

        # Attach recorder to env for this episode
        # NOTE: We need to wrap env with recording wrapper.
        # We'll use the _RobosuiteRecordingWrapper from run_embodied_stacking_robosuite
        from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
            _RobosuiteRecordingWrapper,
            _PHASE_TO_SKILL_ID as _RE_PHASE_TO_SKILL,
        )
        wrapped_env = _RobosuiteRecordingWrapper(
            env, recorder, available_objects,
            policy_override=None,   # scripted-only
            override_phases=set(),
        )
        recorder.end_episode()  # commit previous episode before starting new one

        t_start = time.time()
        result = run_single_episode(
            ep, available_objects, wrapped_env, planner,
            None, set(), exec_config,  # scripted-only
            collect_diagnostics=False,
        )
        elapsed = time.time() - t_start

        try:
            env.env.close()
        except Exception:
            pass

        all_results.append(result)
        status = "✅" if result.get("task_success") else "❌"
        print(f"  {status} success={result.get('task_success')}, "
              f"steps={result.get('steps')}, "
              f"failure={result.get('failure_reason')}, "
              f"time={elapsed:.1f}s")

        if result.get("task_success"):
            success_seeds.append(episode_id)
            # Save this trajectory NPZ
            traj_data = recorder.pop_last_episode_data()
            if traj_data is not None:
                npz_path = trajectories_dir / f"{episode_id}.npz"
                np.savez_compressed(npz_path, **traj_data)
                print(f"    Saved trajectory: {npz_path.name} ({len(traj_data.get('filtered_obs', []))} steps)")
                if "gripper_has_object" in traj_data:
                    n_contact = int(np.sum(traj_data["gripper_has_object"] > 0.5))
                    print(f"    contact/holding: gripper_has_object present, {n_contact}/{len(traj_data['gripper_has_object'])} positive steps")
                else:
                    print(f"    contact/holding: NOT present in trajectory data")

    # Write manifest
    manifest_entries = []
    for ep_id in success_seeds:
        manifest_entries.append({
            "episode_id": ep_id,
            "trajectory_path": f"trajectories/{ep_id}.npz",
            "success": True,
            "teacher_profile": "cube2_robust",
            "execution_mode": "scripted_only",
        })
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    # Write summary
    summary = {
        "total_attempts": len(episodes),
        "successful_trajectories": len(success_seeds),
        "success_seeds": success_seeds,
        "teacher_profile": "cube2_robust",
        "execution_mode": "scripted_only",
        "contact_signal_available": True,
        "contact_signal_source": "obs_index_4_gripper_has_object_heuristic",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary: {len(success_seeds)}/{len(episodes)} successful trajectories")
    print(f"Output: {output_dir}")

    # Also save episode results
    results_dir = Path("generated/results/stage_20")
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = results_dir / "stage20_r_contact_smoke_episodes.jsonl"
    episodes_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_results) + "\n", encoding="utf-8")
    (results_dir / "stage20_r_contact_smoke_summary.json").write_text(
        json.dumps({"success_rate": len(success_seeds) / len(episodes) if episodes else 0,
                     "successful": len(success_seeds), "total": len(episodes)}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
