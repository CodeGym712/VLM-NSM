"""Controlled PID vs P-controller comparison with same seeds + bootstrap CI."""

import os, sys, json, numpy as np
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

def run_eval(use_pid, n_ep=50, seed=42, gpu=0):
    """Run evaluation with controlled seed."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    np.random.seed(seed)

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode, build_stacking_goal

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                              cube_names=["cube1","cube2"], render_mode="none")
    cubes = ["cube1","cube2"]; ts = ["cube2","cube1"]
    planner = PlannerAdapter("bfs")
    plan = planner.solve(cubes=cubes, target_stack=ts)
    goal = build_stacking_goal(ts)

    successes = 0; all_steps = []; failures = {}
    for _ in range(n_ep):
        r = _run_robosuite_episode(env, plan.actions, cubes, goal,
            horizon=1000, use_pid_controller=use_pid,
            use_place_state_machine=True, holding_check_enabled=True,
            lift_early_exit_enabled=True)
        if r["goal_success"]: successes += 1
        all_steps.append(r["num_steps"])
        ft = r.get("failure_type")
        if ft: failures[ft] = failures.get(ft, 0) + 1

    return successes / n_ep, np.mean(all_steps), failures


def bootstrap_ci(data, n_resamples=1000):
    """95% CI via bootstrap."""
    data = np.array(data)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_resamples):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, [2.5, 97.5])


if __name__ == "__main__":
    seeds = [2000, 3000, 4000]
    p_results = []; pid_results = []

    for i, seed in enumerate(seeds):
        gpu = (i % 3) + 1
        print(f"Seed {seed} (GPU {gpu})...", flush=True)
        p_rate, p_steps, p_fail = run_eval(use_pid=False, n_ep=50, seed=seed, gpu=gpu)
        pid_rate, pid_steps, pid_fail = run_eval(use_pid=True, n_ep=50, seed=seed, gpu=gpu)
        p_results.append((p_rate, p_steps))
        pid_results.append((pid_rate, pid_steps))
        print(f"  P: {p_rate:.0%} ({p_steps:.0f} steps) {p_fail}")
        print(f"  PID: {pid_rate:.0%} ({pid_steps:.0f} steps) {pid_fail}")

    p_rates = [r[0] for r in p_results]
    pid_rates = [r[0] for r in pid_results]
    p_ci = bootstrap_ci(p_rates)
    pid_ci = bootstrap_ci(pid_rates)

    out = Path("generated/results/control/comparison_controlled")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "P": {"mean": float(np.mean(p_rates)), "std": float(np.std(p_rates)),
              "ci95": [float(p_ci[0]), float(p_ci[1])]},
        "PID": {"mean": float(np.mean(pid_rates)), "std": float(np.std(pid_rates)),
                "ci95": [float(pid_ci[0]), float(pid_ci[1])]},
        "delta": float(np.mean(pid_rates) - np.mean(p_rates)),
        "seeds": seeds, "ep_per_seed": 50,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nP:    {np.mean(p_rates):.1%} ± {np.std(p_rates):.1%} CI=[{p_ci[0]:.1%},{p_ci[1]:.1%}]")
    print(f"PID:  {np.mean(pid_rates):.1%} ± {np.std(pid_rates):.1%} CI=[{pid_ci[0]:.1%},{pid_ci[1]:.1%}]")
    print(f"Delta: {summary['delta']:+.1%}")
    print(f"Saved to {out / 'summary.json'}")
