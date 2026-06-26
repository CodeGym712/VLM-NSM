"""Stage 20-V: Per-step shadow integration.

Runs scripted-only rollout with learned executor predicting in parallel at EVERY step.
Maintains warm history buffer. Records per-step diagnostics.

CLI:
    MUJOCO_GL=egl python -m vlm_nsm.experiments.stage20_v_per_step_shadow_rollout \
        --checkpoint .../best.pt --dataset_dir .../ \
        --suite .../suite.json --execution_config .../config.json \
        --output_dir generated/results/stage_20/stage20_v_per_step_shadow_smoke
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
from vlm_nsm.learning.robosuite_schema import build_robosuite_obs_vector
from vlm_nsm.skills.skill_types import SkillCommand

_PHASE_TO_ID = {"reach_pick": 0, "grasp": 1, "lift": 2, "reach_place": 3, "release": 4, "retreat": 5}


class ShadowEnvWrapper:
    """Wraps a Robosuite env to intercept step() for learned executor shadow prediction.

    Builds observations at each step, runs learned executor, and records diagnostics.
    Does NOT modify the executed action — scripted controller is always in control.
    """

    def __init__(self, env: RobosuiteEnvWrapper, executor: Stage20LearnedExecutor,
                 gate: LearnedExecutorGate, cubes: list[str],
                 direction_id: int = -1, top_object: str = "cube1",
                 bottom_object: str = "cube2"):
        self._env = env
        self._executor = executor
        self._gate = gate
        self._cubes = cubes
        self._direction_id = direction_id
        self._top_object = top_object
        self._bottom_object = bottom_object
        self._current_phase: str = "reach_pick"
        self._step_count = 0

        # Per-step diagnostics
        self.diagnostics: list[dict[str, Any]] = []
        # Episode-level stats
        self.scripted_steps = 0
        self.learned_predictions = 0

    def set_phase(self, phase_name: str):
        self._current_phase = phase_name

    def step(self, action: np.ndarray):
        """Intercept step: build obs, run learned executor, then execute scripted action."""
        # Build observation from current state
        try:
            obs = build_robosuite_obs_vector(
                self._env.get_eef_pose(),
                self._env.get_gripper_state(),
                self._env.get_all_object_poses(),
                cubes=self._cubes,
            )
        except Exception:
            obs = np.zeros(_OBS_DIM, dtype=np.float32)

        phase_id = _PHASE_TO_ID.get(self._current_phase, -1)

        # Learned executor prediction
        learned_result = self._executor.predict(
            obs, phase_id=phase_id,
            top_object=self._top_object,
            bottom_object=self._bottom_object,
        )
        learned_action = learned_result["action"]
        self.learned_predictions += 1

        # Gate check (would-trigger mode)
        gate_result = self._gate.check(
            learned_action, phase_id=phase_id,
            direction_id=self._direction_id,
            is_warm=self._executor.is_warm,
        )

        # Execute the actual (scripted) action
        result = self._env.step(action)
        self.scripted_steps += 1

        # Record diagnostics
        scripted_arr = np.asarray(action, dtype=np.float32).ravel()
        action_error = np.linalg.norm(learned_action - scripted_arr)
        dim6_error = abs(float(learned_action[6]) - float(scripted_arr[6]))

        diag = {
            "step": self._step_count,
            "phase_name": self._current_phase,
            "phase_id": phase_id,
            "direction_id": self._direction_id,
            "scripted_action": scripted_arr.tolist(),
            "learned_action": learned_action.tolist(),
            "action_error": float(action_error),
            "dim6_error": float(dim6_error),
            "learned_action_norm": learned_result["action_norm"],
            "action_finite": learned_result["finite"],
            "history_warm": self._executor.is_warm,
            "gate_pass": gate_result.passed,
            "gate_reason": gate_result.reason,
            "gate_action_norm": gate_result.action_norm,
        }
        if "holding_logit" in learned_result:
            diag["holding_logit"] = learned_result["holding_logit"]
            diag["holding_prob"] = float(1.0 / (1.0 + np.exp(-learned_result["holding_logit"])))
        # obs4 holding signal
        if len(obs) > 4:
            diag["obs4_holding_signal"] = float(obs[4])

        self.diagnostics.append(diag)
        self._step_count += 1

        return result

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_episode_summary(self) -> dict[str, Any]:
        """Compute per-episode summary from accumulated diagnostics."""
        if not self.diagnostics:
            return {}
        norms = [d["learned_action_norm"] for d in self.diagnostics]
        errors = [d["action_error"] for d in self.diagnostics]
        dim6_errors = [d["dim6_error"] for d in self.diagnostics]
        finite = [d["action_finite"] for d in self.diagnostics]
        warm_diags = [d for d in self.diagnostics if d["history_warm"]]
        gate_pass_after_warm = [d["gate_pass"] for d in warm_diags] if warm_diags else []

        # direction1/phase1 stats
        d1p1 = [d for d in self.diagnostics if d["direction_id"] == 1 and d["phase_id"] == 1]

        summary = {
            "total_steps": len(self.diagnostics),
            "learned_predictions": self.learned_predictions,
            "learned_action_finite_rate": float(np.mean(finite)) if finite else 0.0,
            "learned_action_norm_mean": float(np.mean(norms)) if norms else 0.0,
            "learned_action_norm_max": float(np.max(norms)) if norms else 0.0,
            "action_error_mean": float(np.mean(errors)) if errors else 0.0,
            "action_error_p95": float(np.percentile(errors, 95)) if errors else 0.0,
            "dim6_error_mean": float(np.mean(dim6_errors)) if dim6_errors else 0.0,
            "dim6_error_p95": float(np.percentile(dim6_errors, 95)) if dim6_errors else 0.0,
            "gate_pass_rate_after_warmup": float(np.mean(gate_pass_after_warm)) if gate_pass_after_warm else 0.0,
            "history_warm_rate": float(np.mean([d["history_warm"] for d in self.diagnostics])),
            "direction1_phase1_count": len(d1p1),
            "direction1_phase1_dim6_error_mean": float(np.mean([d["dim6_error"] for d in d1p1])) if d1p1 else 0.0,
            "direction1_phase1_action_error_mean": float(np.mean([d["action_error"] for d in d1p1])) if d1p1 else 0.0,
        }
        # Gate reason counts
        gate_reasons = Counter(d["gate_reason"] for d in self.diagnostics)
        summary["gate_reason_counts"] = dict(gate_reasons)

        # Holding prediction accuracy (online, if obs4 signal available)
        obs4_diags = [d for d in self.diagnostics if "obs4_holding_signal" in d and "holding_prob" in d]
        if obs4_diags:
            holding_correct = sum(
                1 for d in obs4_diags
                if (d["holding_prob"] >= 0.5) == (d["obs4_holding_signal"] > 0.5)
            )
            summary["holding_prediction_accuracy_online"] = float(holding_correct / len(obs4_diags))

        return summary


def run_per_step_shadow(
    checkpoint_path: str, dataset_dir: str,
    suite_path: str, exec_config_path: str,
    output_dir: str, device: str = "cpu",
) -> dict[str, Any]:
    """Run per-step shadow smoke."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    suite = json.loads(Path(suite_path).read_text())
    episodes = suite.get("episodes", [])
    cubes = suite.get("available_objects", ["cube1", "cube2"])

    config = json.loads(Path(exec_config_path).read_text())
    valid_keys = {"kp", "max_delta", "use_place_state_machine", "horizon_aware",
                  "place_retry", "grasp_close_steps", "grasp_max_retries",
                  "place_max_retries", "place_mode", "close_steps_min",
                  "close_steps_max", "close_stabilize_steps", "release_steps_min",
                  "release_steps_max", "post_release_stable_steps", "close_mode",
                  "close_steps_fast", "reach_threshold", "lift_height",
                  "release_open_steps", "terminate_after_safe_release"}
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}

    # Load learned executor once
    print(f"Loading checkpoint: {checkpoint_path}")
    executor = Stage20LearnedExecutor(checkpoint_path, dataset_dir, device=device)
    print(f"  {executor}")

    all_episode_results = []
    all_diagnostics = []
    success_count = 0

    for ep_idx, ep in enumerate(episodes):
        episode_id = ep["episode_id"]
        goal_id = ep["goal_id"]
        seed = ep.get("seed", 0)
        direction_id = 0 if goal_id == "red_on_green" else 1
        top_obj = ep.get("expected_top", "cube1")
        bottom_obj = ep.get("expected_bottom", "cube2")

        print(f"\n[{ep_idx + 1}/{len(episodes)}] {episode_id} (dir={direction_id})")

        gate = LearnedExecutorGate(GateConfig())
        executor.reset()

        env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                                  cube_names=cubes, render_mode="none")
        env.reset()

        # Wrap with shadow recorder
        shadow_env = ShadowEnvWrapper(
            env, executor, gate, cubes,
            direction_id=direction_id,
            top_object=top_obj, bottom_object=bottom_obj,
        )

        # Determine skill command
        if goal_id == "red_on_green":
            cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                               target_object_id="cube2", source="table", relation="on")
        else:
            cmd = SkillCommand(skill_name="pick_place", object_id="cube2",
                               target_object_id="cube1", source="table", relation="on")

        scripted_executor = ScriptedSkillExecutor(controller_config=filtered_config)

        t_start = time.time()
        result = scripted_executor.execute(
            skill_command=cmd, env=shadow_env,
            context={"cubes": cubes, "policy_override": None, "override_phases": set()},
        )
        elapsed = time.time() - t_start

        # Episode summary
        ep_summary = shadow_env.get_episode_summary()
        ep_summary["episode_id"] = episode_id
        ep_summary["goal_id"] = goal_id
        ep_summary["direction_id"] = direction_id
        ep_summary["scripted_success"] = result.success
        ep_summary["scripted_steps"] = result.steps
        ep_summary["elapsed_s"] = elapsed

        all_episode_results.append(ep_summary)

        # Tag diagnostics with episode info
        for d in shadow_env.diagnostics:
            d["episode_id"] = episode_id
            d["goal_id"] = goal_id
        all_diagnostics.extend(shadow_env.diagnostics)

        if result.success:
            success_count += 1

        status = "✅" if result.success else "❌"
        print(f"  {status} steps={result.steps}, "
              f"finite={ep_summary.get('learned_action_finite_rate', 0):.1%}, "
              f"warm={ep_summary.get('history_warm_rate', 0):.1%}, "
              f"gate_after_warm={ep_summary.get('gate_pass_rate_after_warmup', 0):.1%}, "
              f"time={elapsed:.1f}s")

        try:
            env.env.close()
        except Exception:
            pass

    # Final summary
    total = len(episodes)
    summary = {
        "total_episodes": total,
        "scripted_success": success_count,
        "scripted_success_rate": round(success_count / total, 4) if total > 0 else 0.0,
        "total_shadow_steps": len(all_diagnostics),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "mode": "per_step_shadow",
    }
    # Aggregate across episodes
    if all_episode_results:
        for key in ["learned_action_finite_rate", "history_warm_rate",
                     "gate_pass_rate_after_warmup", "action_error_mean",
                     "dim6_error_mean", "holding_prediction_accuracy_online"]:
            vals = [e.get(key) for e in all_episode_results if e.get(key) is not None]
            if vals:
                summary[f"mean_{key}"] = round(float(np.mean(vals)), 6)

    # Save
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in all_episode_results) + "\n")
    (output / "diagnostics.jsonl").write_text(
        "\n".join(json.dumps(d) for d in all_diagnostics) + "\n")

    print(f"\n=== Per-Step Shadow Results ===")
    print(f"Scripted: {success_count}/{total} ({summary['scripted_success_rate']:.1%})")
    print(f"Total shadow steps: {summary['total_shadow_steps']}")
    print(f"Mean finite rate: {summary.get('mean_learned_action_finite_rate', 'N/A')}")
    print(f"Mean gate after warmup: {summary.get('mean_gate_pass_rate_after_warmup', 'N/A')}")
    print(f"Output: {output}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 20-V: Per-step shadow rollout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--execution_config", required=True)
    parser.add_argument("--output_dir", default="generated/results/stage_20/stage20_v_per_step_shadow_smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_per_step_shadow(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        suite_path=args.suite,
        exec_config_path=args.execution_config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
