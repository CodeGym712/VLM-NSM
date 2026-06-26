"""Stage 19-R1c: Placement / pose correlation analysis.

Analyzes scripted-only rollout diagnostics to determine whether
green_on_red failures correlate with specific object placement patterns.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_diagnostics(path: str) -> list[dict]:
    episodes: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            episodes.append(json.loads(line))
    return episodes


def analyze(episodes: list[dict]) -> dict:
    """Run placement analysis on diagnostics episodes."""

    # Overall stats
    total = len(episodes)
    successes = [e for e in episodes if e.get("task_success")]
    failures = [e for e in episodes if not e.get("task_success")]

    # Per-goal stats
    per_goal: dict[str, dict] = defaultdict(lambda: {"total": 0, "success": 0, "failures": []})
    for ep in episodes:
        gid = ep.get("goal_id", "unknown")
        per_goal[gid]["total"] += 1
        if ep.get("task_success"):
            per_goal[gid]["success"] += 1
        else:
            per_goal[gid]["failures"].append({
                "episode_id": ep.get("episode_id"),
                "seed": ep.get("seed"),
                "failure_reason": ep.get("failure_reason"),
                "top_region": ep.get("top_region"),
                "bottom_region": ep.get("bottom_region"),
                "initial_top_bottom_distance_xy": ep.get("initial_top_bottom_distance_xy"),
            })

    # Distance XY analysis
    distance_bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.50)]
    distance_analysis: list[dict] = []
    for lo, hi in distance_bins:
        in_bin = [e for e in episodes
                  if e.get("initial_top_bottom_distance_xy") is not None
                  and lo <= e["initial_top_bottom_distance_xy"] < hi]
        if not in_bin:
            continue
        succ = sum(1 for e in in_bin if e.get("task_success"))
        distance_analysis.append({
            "bin": f"{lo:.2f}-{hi:.2f}m",
            "total": len(in_bin),
            "success": succ,
            "rate": round(succ / len(in_bin), 3),
        })

    # Region analysis — per top/bottom region
    region_success: dict[str, dict] = defaultdict(lambda: {"total": 0, "success": 0})
    for ep in episodes:
        tr = ep.get("top_region") or "unknown"
        region_success[tr]["total"] += 1
        if ep.get("task_success"):
            region_success[tr]["success"] += 1

    # Relative XY sign analysis
    xy_sign_analysis: dict[str, dict] = defaultdict(lambda: {"total": 0, "success": 0})
    for ep in episodes:
        rel = ep.get("initial_relative_xy_top_minus_bottom")
        if rel is None:
            continue
        dx = rel.get("dx", 0)
        dy = rel.get("dy", 0)
        x_sign = "left" if dx < 0 else "right"
        y_sign = "front" if dy < 0 else "back"
        key = f"top_{x_sign}_bottom_{y_sign}"
        xy_sign_analysis[key]["total"] += 1
        if ep.get("task_success"):
            xy_sign_analysis[key]["success"] += 1

    # Per-seed analysis
    seed_analysis: dict[int, dict] = defaultdict(lambda: {"total": 0, "success": 0})
    for ep in episodes:
        seed = ep.get("seed", -1)
        seed_analysis[seed]["total"] += 1
        if ep.get("task_success"):
            seed_analysis[seed]["success"] += 1

    # Mean initial poses: success vs failure
    mean_poses = _compute_mean_poses(episodes)

    # Determine root cause hint
    green_stats = per_goal.get("green_on_red", {})
    red_stats = per_goal.get("red_on_red", {})
    red_success = per_goal.get("red_on_green", {}).get("success", 0)
    red_total = per_goal.get("red_on_green", {}).get("total", 1)
    green_success = green_stats.get("success", 0)
    green_total = green_stats.get("total", 1)

    # Check if green failures cluster in specific regions
    green_failures = green_stats.get("failures", [])
    green_fail_regions: dict[str, int] = defaultdict(int)
    for gf in green_failures:
        region_key = f"top={gf.get('top_region')}_bottom={gf.get('bottom_region')}"
        green_fail_regions[region_key] += 1

    root_cause_hint = "unknown"
    if green_success / max(green_total, 1) >= 0.70:
        root_cause_hint = "placement_variance_within_tolerance"
    elif len(set(gf.get("top_region") for gf in green_failures)) <= 2:
        root_cause_hint = "placement_sensitive_scripted_control"
    elif red_success / max(red_total, 1) >= 0.90:
        root_cause_hint = "goal_order_sensitive_scripted_control"

    return {
        "total_episodes": total,
        "success_count": len(successes),
        "failure_count": len(failures),
        "overall_success_rate": round(len(successes) / total, 3) if total > 0 else 0,
        "per_goal": {
            gid: {
                "total": stats["total"],
                "success": stats["success"],
                "rate": round(stats["success"] / stats["total"], 3) if stats["total"] > 0 else 0,
            }
            for gid, stats in per_goal.items()
        },
        "failure_reasons": _count_failure_reasons(episodes),
        "distance_xy_analysis": distance_analysis,
        "region_analysis": {
            tr: {"total": v["total"], "success": v["success"],
                 "rate": round(v["success"] / v["total"], 3) if v["total"] > 0 else 0}
            for tr, v in sorted(region_success.items())
        },
        "xy_sign_analysis": {
            k: {"total": v["total"], "success": v["success"],
                "rate": round(v["success"] / v["total"], 3) if v["total"] > 0 else 0}
            for k, v in sorted(xy_sign_analysis.items())
        },
        "seed_clusters": {
            str(k): {"total": v["total"], "success": v["success"]}
            for k, v in sorted(seed_analysis.items())
        },
        "mean_poses": mean_poses,
        "green_on_red_failure_region_clusters": dict(green_fail_regions),
        "root_cause_hint": root_cause_hint,
        "recommendation": _get_recommendation(
            round(len(successes) / total, 3) if total > 0 else 0,
            green_success / max(green_total, 1),
            root_cause_hint,
        ),
    }


def _compute_mean_poses(episodes: list[dict]) -> dict:
    succ_eps = [e for e in episodes if e.get("task_success")]
    fail_eps = [e for e in episodes if not e.get("task_success")]

    result: dict = {"success": {}, "failure": {}}
    for label, eps in [("success", succ_eps), ("failure", fail_eps)]:
        if not eps:
            continue
        for key in ["initial_cube1_pose", "initial_cube2_pose",
                     "initial_top_pose", "initial_bottom_pose",
                     "initial_top_bottom_distance_xy"]:
            values = [e.get(key) for e in eps if e.get(key) is not None]
            if not values:
                continue
            if isinstance(values[0], dict):
                result[label][key] = {
                    dim: round(sum(v.get(dim, 0) for v in values) / len(values), 4)
                    for dim in ["x", "y", "z"]
                }
            else:
                result[label][key] = round(sum(values) / len(values), 4)
    return result


def _count_failure_reasons(episodes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ep in episodes:
        fr = ep.get("failure_reason", "unknown")
        counts[fr] += 1
    return dict(counts)


def _get_recommendation(overall_rate: float, green_rate: float, hint: str) -> str:
    if overall_rate >= 0.80 and green_rate >= 0.70:
        return (
            "Scripted-only is reliable enough for bidirectional demo collection. "
            "Proceed to Stage 19-D2 reverse demo collection."
        )
    elif overall_rate >= 0.70:
        return (
            "Scripted-only is acceptable but has placement-dependent variance. "
            "For D2 demo collection, consider filtering by initial placement distance "
            "or using collect-until-success to ensure balanced success rates."
        )
    else:
        return (
            "Scripted-only has significant placement sensitivity. "
            "Before D2 demo collection, fix object placement positions "
            "or debug scripted controller on edge-case placements."
        )


def generate_report(analysis: dict, output_path: str, markdown_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Placement analysis written to {output_path}")

    lines = [
        "# Stage 19-R1c Placement Effects Analysis",
        "",
        f"**Total episodes**: {analysis['total_episodes']}",
        f"**Overall success rate**: {analysis['overall_success_rate']:.1%}",
        f"**Root cause hint**: {analysis['root_cause_hint']}",
        "",
        "## Per-Goal Success",
        "",
        "| Goal | Total | Success | Rate |",
        "|------|-------|---------|------|",
    ]
    for gid, stats in analysis["per_goal"].items():
        lines.append(f"| {gid} | {stats['total']} | {stats['success']} | {stats['rate']:.1%} |")

    lines.extend([
        "",
        "## Failure Reasons",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ])
    for reason, count in sorted(analysis["failure_reasons"].items()):
        lines.append(f"| {reason} | {count} |")

    lines.extend([
        "",
        "## Distance XY Analysis",
        "",
        "| Bin | Total | Success | Rate |",
        "|-----|-------|---------|------|",
    ])
    for bin_info in analysis.get("distance_xy_analysis", []):
        lines.append(
            f"| {bin_info['bin']} | {bin_info['total']} | "
            f"{bin_info['success']} | {bin_info['rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Top Object Region Analysis",
        "",
        "| Region | Total | Success | Rate |",
        "|--------|-------|---------|------|",
    ])
    for region, stats in sorted(analysis.get("region_analysis", {}).items()):
        lines.append(f"| {region} | {stats['total']} | {stats['success']} | {stats['rate']:.1%} |")

    lines.extend([
        "",
        "## Relative XY Sign Analysis (top relative to bottom)",
        "",
        "| Pattern | Total | Success | Rate |",
        "|---------|-------|---------|------|",
    ])
    for pattern, stats in sorted(analysis.get("xy_sign_analysis", {}).items()):
        lines.append(f"| {pattern} | {stats['total']} | {stats['success']} | {stats['rate']:.1%} |")

    lines.extend([
        "",
        "## Mean Initial Poses",
        "",
        "### Success Episodes",
    ])
    for key, val in analysis.get("mean_poses", {}).get("success", {}).items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")
    lines.append("### Failure Episodes")
    for key, val in analysis.get("mean_poses", {}).get("failure", {}).items():
        lines.append(f"- **{key}**: {val}")

    if analysis.get("green_on_red_failure_region_clusters"):
        lines.extend([
            "",
            "## Green-on-Red Failure Region Clusters",
            "",
            "| Region Pair | Count |",
            "|------------|-------|",
        ])
        for region_pair, count in sorted(
            analysis["green_on_red_failure_region_clusters"].items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"| {region_pair} | {count} |")

    lines.extend([
        "",
        "## Recommendation",
        "",
        analysis["recommendation"],
    ])

    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown report written to {markdown_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 19-R1c Placement Effects Analysis")
    parser.add_argument("--diagnostics", required=True,
                        help="Path to diagnostics JSONL")
    parser.add_argument("--output",
                        default="generated/results/stage_19/stage19_r1c_placement_effects.json",
                        help="Output JSON path")
    parser.add_argument("--markdown_output",
                        default="generated/results/stage_19/stage19_r1c_placement_effects.md",
                        help="Output markdown path")
    args = parser.parse_args()

    episodes = load_diagnostics(args.diagnostics)
    analysis = analyze(episodes)
    generate_report(analysis, args.output, args.markdown_output)


if __name__ == "__main__":
    main()
