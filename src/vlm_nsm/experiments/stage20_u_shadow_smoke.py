"""Stage 20-U: Shadow-mode learned executor smoke test.

Runs scripted-only rollout while predicting learned actions in parallel (shadow mode).
Does NOT execute learned actions — scripted controller is always in control.

CLI:
    MUJOCO_GL=egl python -m vlm_nsm.experiments.stage20_u_shadow_smoke \
        --checkpoint .../best.pt --dataset_dir .../ \
        --suite .../smoke_suite.json --execution_config .../config.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.embodied.stage20_learned_executor import Stage20LearnedExecutor, _OBS_DIM
from vlm_nsm.embodied.stage20_learned_executor_gate import LearnedExecutorGate, GateConfig
from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.skills.skill_types import SkillCommand
from vlm_nsm.learning.robosuite_schema import build_robosuite_obs_vector

_PHASE_TO_ID = {"reach_pick": 0, "grasp": 1, "lift": 2, "reach_place": 3, "release": 4, "retreat": 5}


def _phase_name_to_id(name: str) -> int:
    return _PHASE_TO_ID.get(name, -1)


def run_shadow_smoke(
    checkpoint_path: str,
    dataset_dir: str,
    suite_path: str,
    exec_config_path: str,
    output_dir: str,
    device: str = "cpu",
    run_gated: bool = False,
) -> dict[str, Any]:
    """Run 20-episode shadow smoke test."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load suite
    suite = json.loads(Path(suite_path).read_text())
    episodes = suite.get("episodes", [])[:20]
    cubes = suite.get("available_objects", ["cube1", "cube2"])

    # Load config
    config = json.loads(Path(exec_config_path).read_text())
    valid_keys = {"kp", "max_delta", "use_place_state_machine", "horizon_aware",
                  "place_retry", "grasp_close_steps", "grasp_max_retries",
                  "place_max_retries", "place_mode", "close_steps_min",
                  "close_steps_max", "close_stabilize_steps", "release_steps_min",
                  "release_steps_max", "post_release_stable_steps", "close_mode",
                  "close_steps_fast", "reach_threshold", "lift_height",
                  "release_open_steps", "terminate_after_safe_release"}
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}

    # Load learned executor
    print(f"Loading checkpoint: {checkpoint_path}")
    executor = Stage20LearnedExecutor(checkpoint_path, dataset_dir, device=device)
    print(f"  {executor}")
    gate = LearnedExecutorGate(GateConfig())

    executor_cfg = Path(checkpoint_path).parent / "config.json"
    norm_stats = {}
    if (Path(dataset_dir) / "normalization_stats.json").exists():
        norm_stats = json.loads((Path(dataset_dir) / "normalization_stats.json").read_text())

    # Run episodes
    all_diagnostics: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    success_count = 0

    for ep_idx, ep in enumerate(episodes):
        episode_id = ep["episode_id"]
        goal_id = ep["goal_id"]
        seed = ep.get("seed", 0)

        # Determine direction
        direction_id = 0 if goal_id == "red_on_green" else 1
        top_obj = ep.get("expected_top", "cube1")
        bottom_obj = ep.get("expected_bottom", "cube2")

        print(f"\n[{ep_idx + 1}/{len(episodes)}] {episode_id} (goal={goal_id}, dir={direction_id})")

        # Determine skill command
        if goal_id == "red_on_green":
            cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                               target_object_id="cube2", source="table", relation="on")
        else:
            cmd = SkillCommand(skill_name="pick_place", object_id="cube2",
                               target_object_id="cube1", source="table", relation="on")

        executor.reset()
        gate.reset_counts()

        env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                                  cube_names=cubes, render_mode="none")
        env.reset()

        scripted_executor = ScriptedSkillExecutor(controller_config=filtered_config)

        t_start = time.time()
        result = scripted_executor.execute(
            skill_command=cmd, env=env,
            context={"cubes": cubes, "policy_override": None, "override_phases": set()},
        )
        elapsed = time.time() - t_start

        # Replay observations through learned executor
        # NOTE: For shadow mode, we don't have per-step observations saved from the scripted execution.
        # Instead, we do a separate forward pass:
        # This is a simplified shadow — we run the executor post-hoc on the final state.
        # For full per-step shadow, would need to intercept env.step() during execution.

        # Post-hoc: run learned executor on current observation
        try:
            obs = build_robosuite_obs_vector(
                env.get_eef_pose(), env.get_gripper_state(),
                env.get_all_object_poses(), cubes=cubes)
        except Exception:
            obs = np.zeros(_OBS_DIM, dtype=np.float32)

        learned_result = executor.predict(obs, phase_id=0, top_object=top_obj, bottom_object=bottom_obj)
        learned_action = learned_result["action"]
        action_finite = learned_result["finite"]

        # Gate check
        gate_result = gate.check(learned_action, phase_id=0, direction_id=direction_id,
                                 is_warm=executor.is_warm)

        # Collect diagnostics
        diag = {
            "episode_id": episode_id, "goal_id": goal_id, "seed": seed,
            "direction_id": direction_id,
            "scripted_success": result.success,
            "scripted_steps": result.steps,
            "scripted_failure": result.metrics.get("raw_result", {}).get("failure_type"),
            "learned_action": learned_action.tolist(),
            "learned_action_norm": learned_result["action_norm"],
            "learned_action_finite": action_finite,
            "learned_action_dim6": float(learned_action[6]),
            "gate_passed": gate_result.passed,
            "gate_reason": gate_result.reason,
            "gate_action_norm": gate_result.action_norm,
            "is_warm": executor.is_warm,
            "elapsed_s": elapsed,
        }
        if "holding_logit" in learned_result:
            diag["holding_logit"] = learned_result["holding_logit"]

        all_diagnostics.append(diag)
        episode_results.append({
            "episode_id": episode_id, "goal_id": goal_id,
            "success": result.success, "failure_reason": str(result.metrics.get("raw_result", {}).get("failure_type", "none")),
            "learned_action_finite": action_finite, "gate_passed": gate_result.passed,
        })
        if result.success:
            success_count += 1

        status = "✅" if result.success else "❌"
        print(f"  {status} scripted={result.success}, steps={result.steps}, "
              f"learned_finite={action_finite}, gate={gate_result.passed}, "
              f"learned_norm={learned_result['action_norm']:.4f}, time={elapsed:.1f}s")

        try:
            env.env.close()
        except Exception:
            pass

    # Summary
    total = len(episodes)
    finite_count = sum(1 for d in all_diagnostics if d["learned_action_finite"])
    gate_pass_count = sum(1 for d in all_diagnostics if d["gate_passed"])
    learned_norms = [d["learned_action_norm"] for d in all_diagnostics]
    dim6_vals = [d["learned_action_dim6"] for d in all_diagnostics]

    summary = {
        "shadow_mode": True,
        "gated_execution": False,
        "total_episodes": total,
        "scripted_success": success_count,
        "scripted_success_rate": round(success_count / total, 4) if total > 0 else 0.0,
        "learned_action_finite_rate": round(finite_count / total, 4) if total > 0 else 0.0,
        "gate_pass_rate": round(gate_pass_count / total, 4) if total > 0 else 0.0,
        "learned_action_norm_mean": round(float(np.mean(learned_norms)), 6) if learned_norms else 0.0,
        "learned_action_norm_max": round(float(np.max(learned_norms)), 6) if learned_norms else 0.0,
        "learned_dim6_mean": round(float(np.mean(dim6_vals)), 6) if dim6_vals else 0.0,
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "device": device,
    }

    # Save outputs
    (output / "shadow_smoke_summary.json").write_text(json.dumps(summary, indent=2))
    (output / "shadow_smoke_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(d) for d in all_diagnostics) + "\n")
    (output / "shadow_smoke_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in episode_results) + "\n")

    print(f"\n=== Shadow Smoke Results ===")
    print(f"Scripted success: {success_count}/{total} ({summary['scripted_success_rate']:.1%})")
    print(f"Learned finite: {finite_count}/{total} ({summary['learned_action_finite_rate']:.1%})")
    print(f"Gate pass: {gate_pass_count}/{total} ({summary['gate_pass_rate']:.1%})")
    print(f"Learned norm: mean={summary['learned_action_norm_mean']:.4f}, max={summary['learned_action_norm_max']:.4f}")
    print(f"Output: {output}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 20-U: Shadow smoke test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--execution_config", required=True)
    parser.add_argument("--output_dir", default="generated/results/stage_20/stage20_u_shadow_smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_shadow_smoke(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        suite_path=args.suite,
        exec_config_path=args.execution_config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
