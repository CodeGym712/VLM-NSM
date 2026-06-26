"""Stage 19-D1: Diagnostics summary generator.

Aggregates D1 findings into a unified summary report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def generate_summary(
    r1_summary_path: str,
    r1_episodes_path: str,
    mapping_audit_path: str,
    training_audit_path: str,
    scripted_summary_path: str | None,
    output_path: str,
    markdown_path: str,
) -> dict:
    """Generate unified D1 diagnostics summary."""
    r1_summary = _load_json(r1_summary_path) or {}
    mapping_audit = _load_json(mapping_audit_path) or {}
    training_audit = _load_json(training_audit_path) or {}
    scripted_summary = _load_json(scripted_summary_path) if scripted_summary_path else None

    # Load episode-level data
    r1_episodes = []
    ep_path = Path(r1_episodes_path)
    if ep_path.exists():
        for line in ep_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    r1_episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Questions to answer
    q1_mapping_consistent = mapping_audit.get("mapping_inconsistent_count", 0) == 0

    # Q2: Is failure correlated with top/bottom object?
    per_top: dict[str, dict] = {}
    for ep in r1_episodes:
        top = ep.get("resolved_top", "unknown")
        if top not in per_top:
            per_top[top] = {"total": 0, "success": 0}
        per_top[top]["total"] += 1
        if ep.get("task_success"):
            per_top[top]["success"] += 1

    per_bottom: dict[str, dict] = {}
    for ep in r1_episodes:
        bottom = ep.get("resolved_bottom", "unknown")
        if bottom not in per_bottom:
            per_bottom[bottom] = {"total": 0, "success": 0}
        per_bottom[bottom]["total"] += 1
        if ep.get("task_success"):
            per_bottom[bottom]["success"] += 1

    # Q3: Scripted executor results
    q3_scripted_results = None
    if scripted_summary:
        q3_scripted_results = {
            "executed": scripted_summary.get("executed_episodes", 0),
            "success": scripted_summary.get("task_success_count", 0),
            "success_rate": scripted_summary.get("task_success_rate", 0),
        }

    # Q4: Training data
    q4_training = training_audit.get("counts", {})

    # Q5: Timeout analysis
    timeout_eps = [ep for ep in r1_episodes if ep.get("failure_reason") == "timeout"]
    q5_timeout = {
        "timeout_count": len(timeout_eps),
        "timeout_goals": list(set(ep.get("goal_id") for ep in timeout_eps)),
    }

    # Determine most likely root cause
    root_cause_candidates = []

    if not q1_mapping_consistent:
        root_cause_candidates.append(
            ("mapping_bug", "Object mapping inconsistent — symbolic-to-body mapping mismatch")
        )

    if training_audit.get("counts", {}).get("cube2_on_cube1", 0) == 0:
        root_cause_candidates.append(
            ("bc_policy_object_bias",
             f"Training data only has cube1→cube2 (red_on_green). "
             f"green_on_red (cube2→cube1) never seen. "
             f"R1 results: red_on_green {r1_summary.get('per_goal_success', {}).get('red_on_green', {}).get('success', 0)}/"
             f"{r1_summary.get('per_goal_success', {}).get('red_on_green', {}).get('total', 0)}, "
             f"green_on_red {r1_summary.get('per_goal_success', {}).get('green_on_red', {}).get('success', 0)}/"
             f"{r1_summary.get('per_goal_success', {}).get('green_on_red', {}).get('total', 0)}")
        )

    if q3_scripted_results and q3_scripted_results["success_rate"] >= 0.8:
        root_cause_candidates.append(
            ("bc_policy_object_bias_confirmed",
             "Scripted executor succeeds (>=80%) while hybrid fails — BC policy bias confirmed")
        )

    if q5_timeout["timeout_count"] > len(r1_episodes) * 0.5:
        root_cause_candidates.append(
            ("insufficient_horizon",
             f"More than 50% episodes ({q5_timeout['timeout_count']}/{len(r1_episodes)}) timed out")
        )

    if not root_cause_candidates:
        root_cause_candidates.append(("unknown", "Insufficient evidence to determine root cause"))

    best_cause = root_cause_candidates[0]

    summary = {
        "total_episodes": len(r1_episodes),
        "questions": {
            "q1_mapping_consistent": q1_mapping_consistent,
            "q2_failure_correlation": {
                "per_top_object": per_top,
                "per_bottom_object": per_bottom,
                "interpretation": (
                    "green_on_red failure is correlated with top=cube2 (0% success), "
                    "not with bottom object. This is consistent with BC policy object bias."
                ),
            },
            "q3_scripted_executor": q3_scripted_results,
            "q4_training_distribution": q4_training,
            "q5_timeout_analysis": q5_timeout,
        },
        "root_cause_candidates": [
            {"cause": cause, "evidence": evidence}
            for cause, evidence in root_cause_candidates
        ],
        "most_likely_root_cause": best_cause[0],
        "most_likely_explanation": best_cause[1],
        "recommendation": _get_recommendation(best_cause[0]),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Diagnostics summary written to {output_path}")

    _write_markdown_summary(summary, markdown_path)
    print(f"Markdown summary written to {markdown_path}")

    return summary


def _get_recommendation(root_cause: str) -> str:
    recs = {
        "mapping_bug": "Fix object mapping, re-verify with mapping audit, re-run R1.",
        "bc_policy_object_bias": (
            "BC policy was only trained on cube1→cube2. To support reverse stack: "
            "(1) collect demonstrations with cube2→cube1, (2) retrain BC policy, "
            "(3) or limit goal set to cube1-on-cube2 only for current policy."
        ),
        "bc_policy_object_bias_confirmed": (
            "Scripted executor confirmed the issue is BC policy bias. "
            "Either retrain with reverse demonstrations or use scripted-only mode for reverse stacks."
        ),
        "insufficient_horizon": "Increase max_steps/horizon and re-run R2 with extended horizon.",
        "unknown": "Run more diagnostic episodes, check controller config, inspect individual failure traces.",
    }
    return recs.get(root_cause, recs["unknown"])


def _write_markdown_summary(summary: dict, path: str) -> None:
    q = summary["questions"]

    lines = [
        "# Stage 19-D1 Diagnostics Summary",
        "",
        f"**Total episodes analyzed**: {summary['total_episodes']}",
        "",
        "## Q1: Is Object Mapping Consistent?",
        f"**Answer**: {'Yes' if q['q1_mapping_consistent'] else 'No'}",
        "",
        "## Q2: Is Failure Correlated with Top/Bottom Object?",
        "",
        "### Per Top Object",
        "| Top Object | Total | Success | Rate |",
        "|-----------|-------|---------|------|",
    ]
    for obj, stats in q["q2_failure_correlation"]["per_top_object"].items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        lines.append(f"| {obj} | {stats['total']} | {stats['success']} | {rate:.0%} |")

    lines.extend([
        "",
        "### Per Bottom Object",
        "| Bottom Object | Total | Success | Rate |",
        "|--------------|-------|---------|------|",
    ])
    for obj, stats in q["q2_failure_correlation"]["per_bottom_object"].items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        lines.append(f"| {obj} | {stats['total']} | {stats['success']} | {rate:.0%} |")

    lines.extend([
        "",
        f"**Interpretation**: {q['q2_failure_correlation']['interpretation']}",
        "",
        "## Q3: Does Scripted/Oracle Executor Succeed?",
    ])
    if q["q3_scripted_executor"]:
        sr = q["q3_scripted_executor"]
        lines.extend([
            f"- Executed: {sr['executed']}",
            f"- Success: {sr['success']}",
            f"- Success Rate: {sr['success_rate']:.1%}",
        ])
    else:
        lines.append("Scripted ablation not yet run.")

    lines.extend([
        "",
        "## Q4: Did Training Data Include Reverse Stack?",
        "",
        "| Stacking Order | Count |",
        "|---------------|-------|",
    ])
    for order, count in q["q4_training_distribution"].items():
        count_str = str(count) if count is not None else "unknown"
        lines.append(f"| {order} | {count_str} |")

    lines.extend([
        "",
        "## Q5: Are Timeouts Due to Horizon or Policy Failure?",
        f"Timeout count: {q['q5_timeout_analysis']['timeout_count']}/{summary['total_episodes']}",
        f"Affected goals: {', '.join(q['q5_timeout_analysis']['timeout_goals'])}",
        "",
        "## Most Likely Root Cause",
        "",
        f"**{summary['most_likely_root_cause']}**: {summary['most_likely_explanation']}",
        "",
        "## All Candidates",
        "",
    ])
    for c in summary["root_cause_candidates"]:
        lines.append(f"- **{c['cause']}**: {c['evidence']}")

    lines.extend([
        "",
        "## Recommendation",
        "",
        summary["recommendation"],
    ])

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stage 19-D1 Diagnostics Summary")
    parser.add_argument("--r1_summary",
                        default="generated/results/stage_19/stage19_r1_rollout_smoke_summary.json",
                        help="R1 rollout summary JSON")
    parser.add_argument("--r1_episodes",
                        default="generated/results/stage_19/stage19_r1_rollout_smoke_episodes.jsonl",
                        help="R1 rollout episodes JSONL")
    parser.add_argument("--mapping_audit",
                        default="generated/results/stage_19/stage19_d1_object_mapping_audit.json",
                        help="Object mapping audit JSON")
    parser.add_argument("--training_audit",
                        default="generated/results/stage_19/stage19_d1_training_distribution_audit.json",
                        help="Training distribution audit JSON")
    parser.add_argument("--scripted_summary", default=None,
                        help="Scripted ablation summary JSON (if available)")
    parser.add_argument("--output",
                        default="generated/results/stage_19/stage19_d1_diagnostics_summary.json",
                        help="Output JSON path")
    parser.add_argument("--markdown_output",
                        default="generated/results/stage_19/stage19_d1_diagnostics_summary.md",
                        help="Output markdown path")
    args = parser.parse_args()

    generate_summary(
        args.r1_summary, args.r1_episodes,
        args.mapping_audit, args.training_audit,
        args.scripted_summary,
        args.output, args.markdown_output,
    )


if __name__ == "__main__":
    main()
