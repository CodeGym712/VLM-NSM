"""Stage 27 cube2/green failure diagnostics.

Reads oracle and live episode results from the matched eval directory,
separates green_on_red episodes from red_on_green, and produces a
failure-phase breakdown for green (cube2) stacking failures.

Outputs: JSON + markdown summary and per-failure details.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.  Returns empty list if missing."""
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
            row["_line"] = i
            rows.append(row)
    return rows


def _task_direction(row: dict[str, Any]) -> str:
    """Return 'green_on_red' or 'red_on_green' from the episode result."""
    # Prefer explicit field; fall back to goal_id / episode_id heuristics.
    td = row.get("task_direction") or row.get("task_direction_id")
    if td:
        return str(td)
    gid = str(row.get("goal_id", "") or row.get("episode_id", ""))
    if "green_on_red" in gid:
        return "green_on_red"
    if "red_on_green" in gid:
        return "red_on_green"
    return "unknown"


def _is_failure(row: dict[str, Any]) -> bool:
    """An episode is a failure when task_success is explicitly False."""
    return row.get("task_success") is False


def _classify_failure_phase(row: dict[str, Any]) -> str:
    """Map failure_reason to a canonical failure phase.

    Returns one of:
        grasp_failure, timeout, place_failure, success, unknown
    """
    reason = str(row.get("failure_reason", "") or "").strip().lower()
    if reason in ("nocontact", "grasp", "slip"):
        return "grasp_failure"
    if reason == "timeout":
        return "timeout"
    if reason == "place":
        return "place_failure"
    if reason == "none":
        return "success"
    return "unknown"


def _table_column_for_reason(reason: str) -> str:
    """Map a raw failure_reason value to a summary-table column key.

    The table columns are: nocontact, slip, timeout, place, unknown.
    """
    r = reason.strip().lower()
    if r == "nocontact":
        return "nocontact"
    if r in ("grasp", "slip"):
        return "slip"
    if r == "timeout":
        return "timeout"
    if r == "place":
        return "place"
    return "unknown"


def diagnose(
    *,
    oracle_jsonl: str | Path,
    live_jsonl: str | Path,
) -> dict[str, Any]:
    """Run cube2/green failure diagnostics over oracle and live episodes."""

    oracle_rows = load_jsonl(oracle_jsonl)
    live_rows = load_jsonl(live_jsonl)

    # Combine with a condition label.
    all_rows: list[dict[str, Any]] = []
    for r in oracle_rows:
        r["_condition"] = "oracle"
        all_rows.append(r)
    for r in live_rows:
        r["_condition"] = "live"
        all_rows.append(r)

    # --- Separate by task direction, keep only green failures ---
    green_rows = [r for r in all_rows if _task_direction(r) == "green_on_red"]
    red_rows = [r for r in all_rows if _task_direction(r) == "red_on_green"]

    green_failures = [r for r in green_rows if _is_failure(r)]
    green_successes = [r for r in green_rows if not _is_failure(r)]

    # --- Summarise per (task_direction, condition) ---
    summary_rows: list[dict[str, Any]] = []
    # Pivot keys are (task_direction, condition).
    for direction in ["green_on_red", "red_on_green"]:
        for cond in ["oracle", "live"]:
            bucket = [
                r
                for r in all_rows
                if _task_direction(r) == direction and r.get("_condition") == cond
            ]
            failures = [r for r in bucket if _is_failure(r)]
            if not bucket:
                continue  # skip empty buckets

            # Build per-reason counts for the table columns.
            reason_counter: Counter = Counter()
            for f in failures:
                reason = str(f.get("failure_reason", "") or "").strip().lower()
                reason_counter[_table_column_for_reason(reason)] += 1

            summary_rows.append(
                {
                    "task_direction": direction,
                    "condition": cond,
                    "n_total": len(bucket),
                    "n_failures": len(failures),
                    "nocontact": reason_counter.get("nocontact", 0),
                    "slip": reason_counter.get("slip", 0),
                    "timeout": reason_counter.get("timeout", 0),
                    "place": reason_counter.get("place", 0),
                    "unknown": reason_counter.get("unknown", 0),
                }
            )

    # --- Per-failure detail ---
    failure_detail: list[dict[str, Any]] = []
    for f in green_failures:
        failure_detail.append(
            {
                "episode_id": f.get("episode_id"),
                "seed": f.get("seed"),
                "condition": f.get("_condition"),
                "failure_reason": f.get("failure_reason"),
                "failure_phase": _classify_failure_phase(f),
                "task_direction": _task_direction(f),
            }
        )

    # --- Aggregate distribution across all green failures ---
    phase_counter: Counter = Counter()
    raw_reason_counter: Counter = Counter()
    for f in green_failures:
        phase_counter[_classify_failure_phase(f)] += 1
        reason = str(f.get("failure_reason", "") or "").strip().lower()
        raw_reason_counter[reason] += 1

    return {
        "scope": "Stage 27 cube2 (green) failure diagnostics",
        "n_oracle_total": len(oracle_rows),
        "n_live_total": len(live_rows),
        "n_green_total": len(green_rows),
        "n_red_total": len(red_rows),
        "n_green_success": len(green_successes),
        "n_green_failures": len(green_failures),
        "green_failure_phase_distribution": dict(phase_counter),
        "green_failure_reason_distribution": dict(raw_reason_counter),
        "summary_table": summary_rows,
        "failure_details": failure_detail,
    }


def write_markdown(analysis: dict[str, Any], path: str | Path) -> None:
    """Write a human-readable markdown report."""

    lines = [
        "# Stage 27 Cube2 / Green Failure Diagnostics",
        "",
        f"Scope: {analysis['scope']}",
        f"Oracle episodes loaded: {analysis['n_oracle_total']}",
        f"Live episodes loaded: {analysis['n_live_total']}",
        f"Green-on-red episodes: {analysis['n_green_total']}",
        f"Red-on-green episodes: {analysis['n_red_total']}",
        f"Green successes: {analysis['n_green_success']}",
        f"Green failures: {analysis['n_green_failures']}",
        "",
        "## Summary Table",
        "",
        "| task_direction | condition | n_total | n_failures | nocontact | slip | timeout | place | unknown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sr in analysis["summary_table"]:
        lines.append(
            f"| {sr['task_direction']} | {sr['condition']} "
            f"| {sr['n_total']} | {sr['n_failures']} "
            f"| {sr['nocontact']} | {sr['slip']} | {sr['timeout']} "
            f"| {sr['place']} | {sr['unknown']} |"
        )

    lines += [
        "",
        "## Green Failure Phase Distribution",
    ]
    for phase, count in sorted(analysis["green_failure_phase_distribution"].items()):
        lines.append(f"- {phase}: {count}")

    lines += [
        "",
        "## Green Failure Reason Distribution (raw)",
    ]
    for reason, count in sorted(
        analysis["green_failure_reason_distribution"].items()
    ):
        lines.append(f"- {reason}: {count}")

    lines += [
        "",
        "## Per-Failure Details",
    ]
    if analysis["failure_details"]:
        lines += [
            "",
            "| episode_id | seed | condition | failure_reason | failure_phase | task_direction |",
            "|---|---|---|---|---|---|",
        ]
        for fd in analysis["failure_details"]:
            lines.append(
                f"| {fd['episode_id']} | {fd['seed']} | {fd['condition']} "
                f"| {fd['failure_reason']} | {fd['failure_phase']} "
                f"| {fd['task_direction']} |"
            )
    else:
        lines.append("")
        lines.append("(no failures)")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 27 cube2 failure diagnostics"
    )
    parser.add_argument(
        "--oracle-jsonl",
        required=True,
        help="Path to oracle_episodes.jsonl",
    )
    parser.add_argument(
        "--live-jsonl",
        required=True,
        help="Path to live_episodes.jsonl",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path for JSON output (e.g. cube2_failure_diagnostics.json)",
    )
    parser.add_argument(
        "--output-md",
        required=True,
        help="Path for markdown output (e.g. cube2_failure_diagnostics.md)",
    )
    args = parser.parse_args()

    analysis = diagnose(
        oracle_jsonl=args.oracle_jsonl,
        live_jsonl=args.live_jsonl,
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(analysis, args.output_md)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
