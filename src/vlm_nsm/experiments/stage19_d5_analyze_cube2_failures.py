"""Stage 19-D5 cube2 failure diagnostics aggregation.

Aggregates existing Stage 19 scripted/fallback diagnostics without running
Robosuite. The focus is green_on_red failures where cube2/green_cube is the
top object.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source"] = source
            row["_line"] = i
            rows.append(row)
    return rows


def is_green_on_red_failure(row: dict[str, Any]) -> bool:
    goal_id = str(row.get("goal_id", ""))
    top = row.get("resolved_top")
    return (
        ("green_on_red" in goal_id or "d2_green_on_red" in goal_id)
        and top in ("cube2", "green_cube")
        and not bool(row.get("task_success", False))
    )


def analyze_cube2_failures(
    *,
    r1c_diagnostics: str | Path,
    d2_episodes: str | Path,
    d4_diagnostics: str | Path,
) -> dict[str, Any]:
    rows = []
    rows.extend(load_jsonl(r1c_diagnostics, "r1c_scripted_40"))
    rows.extend(load_jsonl(d2_episodes, "d2_reverse_collection"))
    rows.extend(load_jsonl(d4_diagnostics, "d4_fallback"))
    failures = [r for r in rows if is_green_on_red_failure(r)]

    reason_counts = Counter(str(r.get("failure_reason", "unknown") or "unknown") for r in failures)
    source_counts = Counter(str(r.get("_source", "unknown")) for r in failures)
    phase_counts = Counter(_infer_phase(r) for r in failures)
    region_counts = Counter(_region_key(r) for r in failures)

    missing_diagnostics = Counter()
    distances = []
    final_pose_errors = []
    top_displacements = []
    bottom_displacements = []
    for row in failures:
        for key in [
            "initial_cube2_pose",
            "initial_top_pose",
            "final_cube2_pose",
            "final_top_pose",
            "min_gripper_distance_to_top",
            "planned_grasp_pose",
            "planned_place_pose",
            "target_place_pose",
        ]:
            if row.get(key) is None:
                missing_diagnostics[key] += 1
        if row.get("initial_top_bottom_distance_xy") is not None:
            distances.append(float(row["initial_top_bottom_distance_xy"]))
        if row.get("final_pose_error") is not None:
            final_pose_errors.append(float(row["final_pose_error"]))
        if row.get("top_displacement") is not None:
            top_displacements.append(float(row["top_displacement"]))
        if row.get("bottom_displacement") is not None:
            bottom_displacements.append(float(row["bottom_displacement"]))

    recommendations = _recommend(reason_counts, phase_counts, missing_diagnostics)
    return {
        "scope": "green_on_red failures where resolved_top is cube2/green_cube",
        "total_input_rows": len(rows),
        "failure_count": len(failures),
        "failure_reason_distribution": dict(reason_counts),
        "source_distribution": dict(source_counts),
        "failure_phase_distribution": dict(phase_counts),
        "region_distribution": dict(region_counts),
        "initial_top_bottom_distance_xy": _stats(distances),
        "final_pose_error": _stats(final_pose_errors),
        "top_displacement": _stats(top_displacements),
        "bottom_displacement": _stats(bottom_displacements),
        "missing_diagnostics": dict(missing_diagnostics),
        "unknown_failure_count": int(reason_counts.get("unknown", 0)),
        "recommendations": recommendations,
        "failure_examples": [_brief_failure(r) for r in failures[:20]],
    }


def _infer_phase(row: dict[str, Any]) -> str:
    phase = row.get("skill_failure_phase")
    if phase:
        return str(phase)
    trace = row.get("phase_transition_trace") or {}
    if trace.get("grasp", 0) and not trace.get("lift", 0):
        return "grasp"
    if trace.get("reach_place", 0) and not row.get("task_success", False):
        return "place"
    return "unknown"


def _region_key(row: dict[str, Any]) -> str:
    return f"top={row.get('top_region') or 'unknown'},bottom={row.get('bottom_region') or 'unknown'}"


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _brief_failure(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row.get("_source"),
        "episode_id": row.get("episode_id"),
        "seed": row.get("seed"),
        "failure_reason": row.get("failure_reason"),
        "inferred_phase": _infer_phase(row),
        "initial_cube2_pose": row.get("initial_cube2_pose"),
        "final_cube2_pose": row.get("final_cube2_pose"),
        "phase_transition_trace": row.get("phase_transition_trace"),
    }


def _recommend(reason_counts: Counter, phase_counts: Counter, missing: Counter) -> list[str]:
    recs: list[str] = []
    grasp_like = reason_counts.get("grasp_failure", 0) + reason_counts.get("unknown", 0) + phase_counts.get("grasp", 0)
    place_like = reason_counts.get("place_failure", 0) + phase_counts.get("place", 0)
    if grasp_like:
        recs.append("adjust_grasp_height")
        recs.append("adjust_gripper_width")
    if place_like:
        recs.append("adjust_place_height")
        recs.append("add_stabilization_steps")
    if reason_counts.get("unknown", 0) or missing:
        recs.append("improve_failure_labeling")
    if not recs:
        recs.append("unknown")
    return list(dict.fromkeys(recs))


def write_markdown(analysis: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Stage 19-D5 Cube2 Failure Analysis",
        "",
        f"Scope: {analysis['scope']}",
        f"Input rows: {analysis['total_input_rows']}",
        f"Green-on-red failure rows: {analysis['failure_count']}",
        "",
        "## Failure Reasons",
    ]
    for key, value in sorted(analysis["failure_reason_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Inferred Failure Phases"]
    for key, value in sorted(analysis["failure_phase_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Missing Diagnostics"]
    for key, value in sorted(analysis["missing_diagnostics"].items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Recommendations"]
    for rec in analysis["recommendations"]:
        lines.append(f"- {rec}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage 19-D5 cube2 failures")
    parser.add_argument("--r1c_diagnostics", required=True)
    parser.add_argument("--d2_episodes", required=True)
    parser.add_argument("--d4_diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown_output", required=True)
    args = parser.parse_args()

    analysis = analyze_cube2_failures(
        r1c_diagnostics=args.r1c_diagnostics,
        d2_episodes=args.d2_episodes,
        d4_diagnostics=args.d4_diagnostics,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(analysis, args.markdown_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
