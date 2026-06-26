"""Build Stage 19-D5 profile sweep and best-profile reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_FILES = {
    "default": "stage19_d5_profile_default_summary.json",
    "cube2_grasp_lower": "stage19_d5_profile_grasp_lower_summary.json",
    "cube2_place_higher": "stage19_d5_profile_place_higher_summary.json",
    "cube2_robust": "stage19_d5_profile_robust_summary.json",
}


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def profile_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"status": "missing"}
    failures = summary.get("per_failure_reason_counts", {})
    return {
        "status": "done",
        "success": summary.get("task_success_count", 0),
        "total": summary.get("executed_episodes", summary.get("total_episodes", 0)),
        "success_rate": summary.get("task_success_rate", 0.0),
        "grasp_failure": failures.get("grasp_failure", 0),
        "place_failure": failures.get("place_failure", 0),
        "timeout": failures.get("timeout", 0),
        "unknown": failures.get("unknown", 0),
        "unsafe_leaks": summary.get("unsafe_goal_leak_count", 0),
        "runtime_errors": summary.get("controller_runtime_errors", 0),
    }


def choose_best(metrics: dict[str, dict[str, Any]]) -> str | None:
    done = {k: v for k, v in metrics.items() if v.get("status") == "done"}
    if not done:
        return None
    return max(done, key=lambda k: (done[k]["success_rate"], -done[k]["grasp_failure"], -done[k]["place_failure"]))


def build_profile_sweep_report(results_dir: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    metrics = {
        profile: profile_metrics(load_summary(root / filename))
        for profile, filename in PROFILE_FILES.items()
    }
    best = choose_best(metrics)
    lines = [
        "# Stage 19-D5 Scripted Profile Sweep",
        "",
        "| Profile | Status | Success | Rate | Grasp Fail | Place Fail | Timeout | Unknown | Unsafe | Runtime Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, row in metrics.items():
        if row["status"] == "missing":
            lines.append(f"| {profile} | missing | - | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {profile} | done | {row['success']}/{row['total']} | {row['success_rate']:.1%} | "
                f"{row['grasp_failure']} | {row['place_failure']} | {row['timeout']} | {row['unknown']} | "
                f"{row['unsafe_leaks']} | {row['runtime_errors']} |"
            )
    lines += ["", f"Recommended best profile: `{best or 'pending'}`"]
    if best is None:
        lines += [
            "",
            "Profile summaries are missing. Run `scripts/run_stage19_d5_profile_sweep.sh` first.",
        ]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metrics": metrics, "best_profile": best}


def build_best_eval_report(
    summary_path: str | Path,
    output: str | Path,
    *,
    profile: str,
) -> dict[str, Any]:
    summary = load_summary(Path(summary_path))
    if summary is None:
        lines = [
            "# Stage 19-D5 Best Profile Evaluation",
            "",
            f"Best profile `{profile}` has not been evaluated yet.",
            "Run `scripts/run_stage19_d5_best_profile_eval.sh` after selecting the best sweep profile.",
        ]
        Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"status": "missing", "profile": profile}

    per_goal = summary.get("per_goal_success", {})
    failures = summary.get("per_failure_reason_counts", {})
    red = per_goal.get("red_on_green", {"success": 0, "total": 0})
    green = per_goal.get("green_on_red", {"success": 0, "total": 0})
    overall_rate = summary.get("task_success_rate", 0.0)
    red_rate = red["success"] / red["total"] if red.get("total") else 0.0
    green_rate = green["success"] / green["total"] if green.get("total") else 0.0
    verdict = "PASS" if (
        summary.get("unsafe_goal_leak_count", 0) == 0
        and summary.get("controller_runtime_errors", 0) == 0
        and overall_rate >= 0.80
        and green_rate >= 0.75
        and red_rate >= 0.80
    ) else "PARTIAL_PASS" if overall_rate >= 0.725 or green_rate >= 0.60 else "FAIL"

    lines = [
        "# Stage 19-D5 Best Profile Evaluation",
        "",
        f"Profile: `{profile}`",
        f"Verdict: **{verdict}**",
        "",
        "| Executor | red_on_green | green_on_red | Overall |",
        "|---|---:|---:|---:|",
        "| R1c scripted-only default | 85.0% | 60.0% | 72.5% |",
        "| D4 BC+fallback | 75.0% | 45.0% | 60.0% |",
        f"| D5 `{profile}` | {red_rate:.1%} | {green_rate:.1%} | {overall_rate:.1%} |",
        "",
        "## Failure Counts",
    ]
    for key, value in sorted(failures.items()):
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        f"Unsafe leaks: {summary.get('unsafe_goal_leak_count', 0)}",
        f"Runtime errors: {summary.get('controller_runtime_errors', 0)}",
    ]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "done", "profile": profile, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 19-D5 profile reports")
    parser.add_argument("--results_dir", default="generated/results/stage_19")
    parser.add_argument("--profile_sweep_output", default="generated/results/stage_19/stage19_d5_profile_sweep.md")
    parser.add_argument("--best_summary", default="generated/results/stage_19/stage19_d5_best_profile_eval_summary.json")
    parser.add_argument("--best_report", default="generated/results/stage_19/stage19_d5_best_profile_eval_report.md")
    parser.add_argument("--best_profile", default="cube2_robust")
    args = parser.parse_args()

    sweep = build_profile_sweep_report(args.results_dir, args.profile_sweep_output)
    best = sweep["best_profile"] or args.best_profile
    build_best_eval_report(args.best_summary, args.best_report, profile=best)
    print(f"Wrote {args.profile_sweep_output}")
    print(f"Wrote {args.best_report}")


if __name__ == "__main__":
    main()
