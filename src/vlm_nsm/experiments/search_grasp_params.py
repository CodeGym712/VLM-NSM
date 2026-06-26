"""Grid search over grasp parameters to find optimal configuration.

Usage:
    python -m vlm_nsm.experiments.search_grasp_params
    python -m vlm_nsm.experiments.search_grasp_params --quick
    python -m vlm_nsm.experiments.search_grasp_params --episodes 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _try_import_robosuite():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper  # noqa: F401
        return True
    except ImportError:
        return False


def search_params(quick: bool = False, episodes: int = 3) -> list[dict]:
    """Run grid search over grasp parameters.

    Args:
        quick: Use smaller grid (2 values per param).
        episodes: Episodes per config.

    Returns:
        List of result dicts per config.
    """
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    if quick:
        kp_values = [0.05, 0.1]
        approach_height_values = [0.06, 0.10]
        close_steps_values = [40, 60]
        max_retries_values = [1, 3]
    else:
        kp_values = [0.02, 0.05, 0.1]
        approach_height_values = [0.05, 0.08, 0.12]
        close_steps_values = [30, 50, 80]
        max_retries_values = [1, 3]

    param_grid = list(itertools.product(kp_values, approach_height_values, close_steps_values, max_retries_values))
    total_configs = len(param_grid)
    total_episodes = total_configs * episodes

    print(f"Parameter search: {total_configs} configs x {episodes} episodes = {total_episodes} total runs")

    results: list[dict] = []

    for idx, (kp, approach_height, close_steps, max_retries) in enumerate(param_grid):
        config_label = f"kp={kp},ah={approach_height},cs={close_steps},mr={max_retries}"
        print(f"\n[{idx + 1}/{total_configs}] Testing {config_label}...")

        config_successes = 0
        config_deltas: list[float] = []
        config_failures: dict[str, int] = {}

        for ep in range(episodes):
            env = RobosuiteEnvWrapper(render_mode="none")
            env.reset()

            from vlm_nsm.embodied.grasp_state_machine import GraspConfig, execute_grasp

            cfg = GraspConfig(
                kp=kp,
                approach_height=approach_height,
                close_steps=close_steps,
                max_retries=max_retries,
            )
            result = execute_grasp(env, "cube1", cfg)

            if result["success"]:
                config_successes += 1
            config_deltas.append(result["object_z_delta"])
            ftype = result.get("failure_type", "grasp_failed")
            config_failures[ftype] = config_failures.get(ftype, 0) + 1

            env.env.close()

        success_rate = config_successes / max(episodes, 1)
        mean_delta = sum(config_deltas) / max(len(config_deltas), 1)
        max_delta = max(config_deltas)

        print(f"  Success: {config_successes}/{episodes} ({success_rate * 100:.0f}%)")
        print(f"  Mean obj_z_delta: {mean_delta:.4f}m")
        print(f"  Failures: {config_failures}")

        results.append({
            "config": {
                "kp": kp,
                "approach_height": approach_height,
                "close_steps": close_steps,
                "max_retries": max_retries,
            },
            "episodes": episodes,
            "success_count": config_successes,
            "success_rate": success_rate,
            "mean_obj_z_delta": mean_delta,
            "max_obj_z_delta": max_delta,
            "failure_distribution": config_failures,
        })

    # Sort by success rate descending, then mean delta descending
    results.sort(key=lambda r: (r["success_rate"], r["mean_obj_z_delta"]), reverse=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Grid search grasp parameters")
    parser.add_argument("--quick", action="store_true", help="Use smaller grid (2 values per param)")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per config")
    args = parser.parse_args()

    if not _try_import_robosuite():
        print("ERROR: Robosuite not available.", file=sys.stderr)
        sys.exit(1)

    results = search_params(quick=args.quick, episodes=args.episodes)

    # Summary
    print("\n" + "=" * 60)
    print("TOP 5 CONFIGS (by success rate, then mean delta)")
    print("=" * 60)
    for i, r in enumerate(results[:5]):
        c = r["config"]
        print(f"{i + 1}. kp={c['kp']}, ah={c['approach_height']}, "
              f"cs={c['close_steps']}, mr={c['max_retries']}: "
              f"{r['success_rate'] * 100:.0f}% ({r['success_count']}/{r['episodes']}), "
              f"mean_delta={r['mean_obj_z_delta']:.4f}m")

    best = results[0]
    print(f"\nBest config: {best['config']}")
    print(f"  Success rate: {best['success_rate'] * 100:.0f}%")
    print(f"  Mean obj_z_delta: {best['mean_obj_z_delta']:.4f}m")

    # Save
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / "grasp_param_search.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
