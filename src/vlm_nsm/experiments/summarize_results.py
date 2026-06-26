from __future__ import annotations

import csv
import json
from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _safe_load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def _str(val) -> str:
    if val is None:
        return ""
    return str(val)


def _float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _int(val) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def collect_main_experiments() -> list[dict]:
    results_dir = PROJECT_ROOT / "generated" / "results"
    rows: list[dict] = []
    patterns = [
        ("hanoi_{n}_mock_internal_hanoi_symbolic.json", "hanoi", "mock", "internal_hanoi", "symbolic"),
        ("stacking_{n}_mock_internal_stacking_symbolic.json", "stacking", "mock", "internal_stacking", "symbolic"),
    ]
    for pattern_fmt, task, parser, planner, executor in patterns:
        for n in [3, 4, 5]:
            fname = pattern_fmt.replace("{n}", str(n))
            data = _safe_load(results_dir / fname)
            if not data:
                # fallback to old filename pattern
                old_fname = f"{task}_{n}_symbolic.json"
                data = _safe_load(results_dir / old_fname)
                if not data:
                    continue
                fname = old_fname
            rows.append({
                "task": task,
                "n_blocks": n,
                "parser": parser,
                "planner": planner,
                "executor": executor,
                "method": "symbolic_pipeline",
                "atomic_action_length": _int(data.get("atomic_action_length", 0)),
                "logical_plan_length": _int(data.get("logical_plan_length", 0)),
                "optimal_logical_length": _int(data.get("optimal_logical_length", 0)),
                "plan_optimality_ratio": _float(data.get("plan_optimality_ratio", 0.0)),
                "planner_success": data.get("planner_success", False),
                "execution_success": data.get("execution_success", False),
                "goal_success": data.get("goal_success", False),
                "error_type": _str(data.get("error_type", "")),
                "result_file": fname,
            })
    return rows


def collect_fixed_goal() -> list[dict]:
    results_dir = PROJECT_ROOT / "generated" / "results"
    rows: list[dict] = []
    for task in ["hanoi", "stacking"]:
        for n in [3, 4, 5]:
            data = _safe_load(results_dir / f"fixed_goal_{task}_{n}.json")
            if not data:
                continue
            action_count = data.get("action_count", 0)
            optimal = 2**n - 1 if task == "hanoi" else n - 1
            logical = action_count // 2 if task == "hanoi" else action_count
            rows.append({
                "task": task,
                "n_blocks": n,
                "parser": "none",
                "planner": f"internal_{task}",
                "executor": "symbolic",
                "method": "fixed_goal",
                "atomic_action_length": action_count,
                "logical_plan_length": logical,
                "optimal_logical_length": optimal,
                "plan_optimality_ratio": _float(data.get("plan_optimality_ratio", 0.0)),
                "planner_success": data.get("planner_success", False),
                "execution_success": data.get("execution_success", False),
                "goal_success": data.get("goal_success", False),
                "error_type": _str(data.get("error", "")),
                "result_file": f"fixed_goal_{task}_{n}.json",
            })
    return rows


def collect_direct_plan() -> list[dict]:
    results_dir = PROJECT_ROOT / "generated" / "results"
    rows: list[dict] = []
    for task in ["hanoi", "stacking"]:
        for n in [3, 4, 5]:
            data = _safe_load(results_dir / f"llm_direct_plan_{task}_{n}_oracle.json")
            if data:
                action_count = data.get("action_count", 0)
                optimal = 2**n - 1 if task == "hanoi" else n - 1
                logical = action_count // 2 if task == "hanoi" else action_count
                rows.append({
                    "task": task,
                    "n_blocks": n,
                    "parser": "none",
                    "planner": "direct_plan",
                    "executor": "symbolic",
                    "method": "oracle_direct_plan",
                    "atomic_action_length": action_count,
                    "logical_plan_length": logical,
                    "optimal_logical_length": optimal,
                    "plan_optimality_ratio": 1.0 if data.get("goal_success") else 0.0,
                    "planner_success": True,
                    "execution_success": data.get("execution_success", False),
                    "goal_success": data.get("goal_success", False),
                    "error_type": _str(data.get("error_type", "")),
                    "result_file": f"llm_direct_plan_{task}_{n}_oracle.json",
                })
            corrupted = _safe_load(results_dir / f"llm_direct_plan_{task}_{n}_corrupted.json")
            if corrupted and "results" in corrupted:
                for trial in corrupted["results"]:
                    action_count = trial.get("action_count", 0)
                    optimal = 2**n - 1 if task == "hanoi" else n - 1
                    rows.append({
                        "task": task,
                        "n_blocks": n,
                        "parser": "none",
                        "planner": "direct_plan",
                        "executor": "symbolic",
                        "method": f"corrupted_{trial.get('corruption_type', 'unknown')}",
                        "atomic_action_length": action_count,
                        "logical_plan_length": action_count // 2 if task == "hanoi" else action_count,
                        "optimal_logical_length": optimal,
                        "plan_optimality_ratio": 1.0 if trial.get("goal_success") else 0.0,
                        "planner_success": True,
                        "execution_success": trial.get("execution_success", False),
                        "goal_success": trial.get("goal_success", False),
                        "error_type": _str(trial.get("error_type", "")),
                        "result_file": f"llm_direct_plan_{task}_{n}_corrupted.json",
                    })
    return rows


def collect_ablation() -> list[dict]:
    results_dir = PROJECT_ROOT / "generated" / "results"
    rows: list[dict] = []
    for task in ["hanoi", "stacking"]:
        for n in [3, 4, 5]:
            data = _safe_load(results_dir / f"alignment_ablation_{task}_{n}.json")
            if not data or "results" not in data:
                continue
            for trial in data["results"]:
                rows.append({
                    "task": task,
                    "n_blocks": n,
                    "ablation_mode": trial.get("ablation_mode", ""),
                    "label": trial.get("label", ""),
                    "object_validity": trial.get("object_validity", False),
                    "predicate_validity": trial.get("predicate_validity", False),
                    "type_check_pass": trial.get("type_check_pass", False),
                    "goal_completeness": trial.get("goal_completeness", False),
                    "repaired": trial.get("repaired", False),
                    "error_type": _str(trial.get("error_type", "")),
                    "planner_success": trial.get("planner_success", False),
                    "execution_success": trial.get("execution_success", False),
                    "goal_success": trial.get("goal_success", False),
                    "result_file": f"alignment_ablation_{task}_{n}.json",
                })
    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")

    main_rows = collect_main_experiments()
    write_csv(main_rows, results_dir / "main_experiment_summary.csv")
    print(f"main_experiment_summary.csv: {len(main_rows)} rows")

    ablation_rows = collect_ablation()
    write_csv(ablation_rows, results_dir / "ablation_summary.csv")
    print(f"ablation_summary.csv: {len(ablation_rows)} rows")

    baseline_rows = collect_fixed_goal() + collect_direct_plan()
    write_csv(baseline_rows, results_dir / "baseline_summary.csv")
    print(f"baseline_summary.csv: {len(baseline_rows)} rows")


if __name__ == "__main__":
    main()
