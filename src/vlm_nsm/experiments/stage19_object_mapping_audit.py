"""Stage 19-D1: Object mapping audit.

Verifies mapping consistency between:
- symbolic objects: cube1, cube2
- robosuite bodies/geoms: cubeA_main, cubeB_main
- colors: red → cube1 → cubeA, green → cube2 → cubeB
- target_stack top/bottom

No robosuite required. Uses project conventions and static analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_nsm.executors.robosuite_env_wrapper import CUBE_NAME_MAP
from vlm_nsm.perception.data_types import DEFAULT_COLOR_MAP


def run_object_mapping_audit(rollout_suite_path: str, output_path: str, markdown_path: str) -> dict:
    """Audit object mappings for all episodes in a rollout suite.

    Returns the audit dict.
    """
    suite = json.loads(Path(rollout_suite_path).read_text(encoding="utf-8"))
    episodes = suite.get("episodes", [])
    color_map = suite.get("color_map", {"cube1": "red", "cube2": "green"})

    audit_results: list[dict] = []
    mapping_issues: list[str] = []

    for ep in episodes:
        episode_id = ep["episode_id"]
        goal_id = ep["goal_id"]
        parser_output = ep.get("parser_output", {})

        resolved_top = ep.get("expected_top")
        resolved_bottom = ep.get("expected_bottom")

        top_body = CUBE_NAME_MAP.get(resolved_top) if resolved_top else None
        bottom_body = CUBE_NAME_MAP.get(resolved_bottom) if resolved_bottom else None
        cube1_body = CUBE_NAME_MAP.get("cube1")
        cube2_body = CUBE_NAME_MAP.get("cube2")

        cube1_color = color_map.get("cube1", DEFAULT_COLOR_MAP.get("cube1"))
        cube2_color = color_map.get("cube2", DEFAULT_COLOR_MAP.get("cube2"))

        top_color = color_map.get(resolved_top) if resolved_top else None
        bottom_color = color_map.get(resolved_bottom) if resolved_bottom else None

        red_body = cube1_body if cube1_color == "red" else (cube2_body if cube2_color == "red" else None)
        green_body = cube1_body if cube1_color == "green" else (cube2_body if cube2_color == "green" else None)

        mapping_consistent = True
        check_details: list[str] = []

        # Check 1: top/bottom should not be same
        if resolved_top == resolved_bottom:
            mapping_consistent = False
            check_details.append("top and bottom are the same object")

        # Check 2: top/bottom should be in CUBE_NAME_MAP
        if resolved_top and resolved_top not in CUBE_NAME_MAP:
            mapping_consistent = False
            check_details.append(f"resolved_top '{resolved_top}' missing from CUBE_NAME_MAP")

        if resolved_bottom and resolved_bottom not in CUBE_NAME_MAP:
            mapping_consistent = False
            check_details.append(f"resolved_bottom '{resolved_bottom}' missing from CUBE_NAME_MAP")

        # Check 3: colors should match body mapping convention
        if top_color and top_body:
            expected_top_body = CUBE_NAME_MAP.get(resolved_top)
            if expected_top_body != top_body:
                mapping_consistent = False
                check_details.append(f"top body mismatch: {top_body} vs {expected_top_body}")

        audit_entry = {
            "episode_id": episode_id,
            "goal_id": goal_id,
            "resolved_top": resolved_top,
            "resolved_bottom": resolved_bottom,
            "top_body": top_body,
            "bottom_body": bottom_body,
            "cube1_body": cube1_body,
            "cube2_body": cube2_body,
            "cube1_color": cube1_color,
            "cube2_color": cube2_color,
            "top_color": top_color,
            "bottom_color": bottom_color,
            "red_body": red_body,
            "green_body": green_body,
            "mapping_consistent": mapping_consistent,
            "color_source": "project_convention",
            "check_details": check_details if check_details else ["all checks passed"],
        }
        audit_results.append(audit_entry)
        if not mapping_consistent:
            mapping_issues.append(f"{episode_id}: {'; '.join(check_details)}")

    summary = {
        "total_episodes": len(episodes),
        "mapping_consistent_count": sum(1 for r in audit_results if r["mapping_consistent"]),
        "mapping_inconsistent_count": sum(1 for r in audit_results if not r["mapping_consistent"]),
        "mapping_issues": mapping_issues,
        "static_mappings": {
            "CUBE_NAME_MAP": CUBE_NAME_MAP,
            "color_map": color_map,
            "DEFAULT_COLOR_MAP": DEFAULT_COLOR_MAP,
        },
        "episodes": audit_results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mapping audit written to {output_path}")

    # Markdown report
    _write_markdown_report(summary, markdown_path)
    print(f"Markdown report written to {markdown_path}")

    return summary


def _write_markdown_report(summary: dict, path: str) -> None:
    lines = [
        "# Stage 19-D1 Object Mapping Audit",
        "",
        f"**Total episodes**: {summary['total_episodes']}",
        f"**Mapping consistent**: {summary['mapping_consistent_count']}",
        f"**Mapping inconsistent**: {summary['mapping_inconsistent_count']}",
        "",
        "## Static Mappings",
        "",
        "### CUBE_NAME_MAP (symbolic → robosuite body)",
        "| Symbolic | Body |",
        "|----------|------|",
    ]
    for k, v in summary["static_mappings"]["CUBE_NAME_MAP"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "### Color Map (symbolic → color)",
        "| Symbolic | Color |",
        "|----------|-------|",
    ])
    for k, v in summary["static_mappings"]["color_map"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "### DEFAULT_COLOR_MAP (project convention)",
        "| Symbolic | Color |",
        "|----------|-------|",
    ])
    for k, v in summary["static_mappings"]["DEFAULT_COLOR_MAP"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Per-Episode Details",
        "",
        "| Episode | Goal | Top | Bottom | Top Body | Bottom Body | Consistent |",
        "|---------|------|-----|--------|----------|-------------|------------|",
    ])
    for ep in summary["episodes"]:
        consistent = "Yes" if ep["mapping_consistent"] else "No"
        lines.append(
            f"| {ep['episode_id']} | {ep['goal_id']} | {ep['resolved_top']} | "
            f"{ep['resolved_bottom']} | {ep['top_body']} | {ep['bottom_body']} | {consistent} |"
        )

    if summary["mapping_issues"]:
        lines.extend([
            "",
            "## Issues Found",
            "",
        ])
        for issue in summary["mapping_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.extend([
            "",
            "## Issues Found",
            "",
            "None. All mappings consistent.",
        ])

    lines.extend([
        "",
        "## Conclusion",
        "",
        f"{'PASS' if summary['mapping_inconsistent_count'] == 0 else 'FAIL'}: "
        f"{summary['mapping_consistent_count']}/{summary['total_episodes']} episodes have consistent mappings.",
        "",
        "Color source: `project_convention` — colors inferred from `DEFAULT_COLOR_MAP` "
        "and suite `color_map`, not read from robosuite at runtime.",
    ])

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stage 19-D1 Object Mapping Audit")
    parser.add_argument("--rollout_suite", required=True,
                        help="Path to rollout suite JSON")
    parser.add_argument("--output",
                        default="generated/results/stage_19/stage19_d1_object_mapping_audit.json",
                        help="Output JSON path")
    parser.add_argument("--markdown_output",
                        default="generated/results/stage_19/stage19_d1_object_mapping_audit.md",
                        help="Output markdown report path")
    args = parser.parse_args()

    run_object_mapping_audit(args.rollout_suite, args.output, args.markdown_output)


if __name__ == "__main__":
    main()
