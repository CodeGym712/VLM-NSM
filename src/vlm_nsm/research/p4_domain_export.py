"""P4 typed symbolic domain export from P3 operator induction artifacts.

P4 is intentionally still an offline symbolic stage.  It turns the P3
operator clauses and repaired plan skeletons into a typed, PDDL-like domain
and audits whether the exported operators can replay the demonstration goals.
The module does not depend on MuJoCo / robosuite.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from vlm_nsm.research.p3_symbolic_induction import (
    Atom,
    _value_matches_aliases,
    atom_to_list,
    goal_specs_from_mapping,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


P4_OPERATOR_ORDER = ["pick", "place_on", "place_in", "open", "turn_on", "push_to_region"]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_atom_list(rows: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(row) for row in rows]


def canonical_p4_domain(
    p3_clauses: Sequence[Mapping[str, Any]],
    push_candidate: Mapping[str, Any] | None = None,
    split_place: bool = True,
) -> dict[str, Any]:
    """Build a typed P4 domain from P3 clauses plus optional server push evidence."""
    support_by_name = {str(clause.get("name")): dict(clause) for clause in p3_clauses}
    push_support = dict(push_candidate or {})

    operators = [
        {
            "name": "pick",
            "parameters": [{"name": "?obj", "type": "object"}],
            "preconditions_pos": [["handempty"]],
            "preconditions_neg": [["holding", "?obj"]],
            "effects_add": [["holding", "?obj"]],
            "effects_del": [["handempty"]],
            "source_operator": "pick",
            "support_count": int(support_by_name.get("pick", {}).get("support_count", 0)),
            "confidence": support_by_name.get("pick", {}).get("confidence", "conditional"),
            "risk_notes": list(support_by_name.get("pick", {}).get("risk_notes", [])),
        },
        {
            "name": "place_on",
            "parameters": [
                {"name": "?obj", "type": "object"},
                {"name": "?target", "type": "support"},
            ],
            "preconditions_pos": [["holding", "?obj"]],
            "preconditions_neg": [],
            "effects_add": [["handempty"], ["on", "?obj", "?target"]],
            "effects_del": [["holding", "?obj"]],
            "source_operator": "place",
            "support_count": int(support_by_name.get("place", {}).get("relation_counts", {}).get("on", 0)),
            "confidence": "conditional",
            "risk_notes": [
                "Split from P3 fused place(on-or-in).",
                "Some support may come from goal-implied final-state repair.",
            ],
        },
        {
            "name": "place_in",
            "parameters": [
                {"name": "?obj", "type": "object"},
                {"name": "?container", "type": "container"},
            ],
            "preconditions_pos": [["holding", "?obj"]],
            "preconditions_neg": [],
            "effects_add": [["handempty"], ["in", "?obj", "?container"]],
            "effects_del": [["holding", "?obj"]],
            "source_operator": "place",
            "support_count": int(support_by_name.get("place", {}).get("relation_counts", {}).get("in", 0)),
            "confidence": "conditional",
            "risk_notes": [
                "Split from P3 fused place(on-or-in).",
                "Open-container preconditions are not asserted in v1 because bowl-in-bowl and drawer-in differ.",
            ],
        },
        {
            "name": "open",
            "parameters": [{"name": "?drawer", "type": "openable"}],
            "preconditions_pos": [["closed", "?drawer"]],
            "preconditions_neg": [],
            "effects_add": [["open", "?drawer"]],
            "effects_del": [["closed", "?drawer"]],
            "source_operator": "open",
            "support_count": int(support_by_name.get("open", {}).get("support_count", 0)),
            "confidence": support_by_name.get("open", {}).get("confidence", "trusted"),
            "risk_notes": list(support_by_name.get("open", {}).get("risk_notes", [])),
        },
        {
            "name": "turn_on",
            "parameters": [{"name": "?device", "type": "device"}],
            "preconditions_pos": [["deactivated", "?device"]],
            "preconditions_neg": [],
            "effects_add": [["activated", "?device"]],
            "effects_del": [["deactivated", "?device"]],
            "source_operator": "turn_on",
            "support_count": int(support_by_name.get("turn_on", {}).get("support_count", 0)),
            "confidence": support_by_name.get("turn_on", {}).get("confidence", "trusted"),
            "risk_notes": list(support_by_name.get("turn_on", {}).get("risk_notes", [])),
        },
        {
            "name": "push_to_region",
            "parameters": [
                {"name": "?obj", "type": "object"},
                {"name": "?region", "type": "region"},
            ],
            "preconditions_pos": [],
            "preconditions_neg": [],
            "effects_add": [["at_region", "?obj", "?region"]],
            "effects_del": [],
            "source_operator": "push_to_region",
            "support_count": int(push_support.get("support_count", 0)),
            "confidence": push_support.get("confidence", "missing_server_validation"),
            "risk_notes": list(push_support.get("risk_notes", ["requires_server_at_region_validation"])),
        },
    ]

    if not split_place:
        fused_place = {
            "name": "place",
            "parameters": [
                {"name": "?obj", "type": "object"},
                {"name": "?target", "type": "support"},
            ],
            "preconditions_pos": [["holding", "?obj"]],
            "preconditions_neg": [],
            "effects_add": [["handempty"], ["on-or-in", "?obj", "?target"]],
            "effects_del": [["holding", "?obj"]],
            "source_operator": "place",
            "support_count": int(support_by_name.get("place", {}).get("support_count", 0)),
            "confidence": "baseline_fused",
            "risk_notes": [
                "Fused baseline keeps P3 place(on-or-in) semantics.",
                "This is useful for replay comparison but less PDDL-clean than place_on/place_in.",
            ],
        }
        operators = [op for op in operators if op["name"] not in {"place_on", "place_in"}]
        operators.insert(1, fused_place)

    return {
        "version": "p4_domain_v1",
        "name": "libero_induced_typed_domain_v1" if split_place else "libero_induced_fused_place_domain_v1",
        "mode": "typed_split_place" if split_place else "fused_place_baseline",
        "source": {
            "p3_operator_clauses": "p3_operator_clauses_v2.json",
            "server_push_candidate": "p3_push_operator_candidate.json",
        },
        "types": {
            "object": "movable task object",
            "support": "surface or object that can support another object",
            "container": "container or openable receptacle used by in(obj, container)",
            "openable": "drawer or articulated fixture",
            "device": "activatable fixture",
            "region": "named spatial region",
        },
        "predicates": [
            {"name": "handempty", "parameters": []},
            {"name": "holding", "parameters": [{"name": "?obj", "type": "object"}]},
            {
                "name": "on",
                "parameters": [{"name": "?obj", "type": "object"}, {"name": "?target", "type": "support"}],
            },
            {
                "name": "in",
                "parameters": [{"name": "?obj", "type": "object"}, {"name": "?container", "type": "container"}],
            },
            {"name": "open", "parameters": [{"name": "?drawer", "type": "openable"}]},
            {"name": "closed", "parameters": [{"name": "?drawer", "type": "openable"}]},
            {"name": "activated", "parameters": [{"name": "?device", "type": "device"}]},
            {"name": "deactivated", "parameters": [{"name": "?device", "type": "device"}]},
            {
                "name": "at_region",
                "parameters": [{"name": "?obj", "type": "object"}, {"name": "?region", "type": "region"}],
            },
        ]
        + (
            []
            if split_place
            else [
                {
                    "name": "on-or-in",
                    "parameters": [{"name": "?obj", "type": "object"}, {"name": "?target", "type": "support"}],
                }
            ]
        ),
        "operators": operators,
        "boundary_notes": [
            "This is a typed symbolic planning domain export, not an end-to-end VLA policy.",
            "P4 v1 validates goal replay consistency, not physical execution success.",
            "place_on/place_in split the P3 fused place relation for PDDL compatibility.",
            "The fused_place_baseline mode is for ablation and should not be used as the final PDDL domain.",
        ],
    }


def pddl_atom(atom: Sequence[str]) -> str:
    if len(atom) == 1:
        return f"({atom[0]})"
    return f"({atom[0]} {' '.join(atom[1:])})"


def render_pddl_like_domain(domain: Mapping[str, Any]) -> str:
    """Render a conservative PDDL-like text file for inspection/reporting."""
    lines = [
        f"(define (domain {domain['name']})",
        "  (:requirements :strips :typing)",
        "  (:types object support container openable device region)",
        "  (:predicates",
    ]
    for pred in domain["predicates"]:
        params = " ".join(f"{p['name']} - {p['type']}" for p in pred["parameters"])
        lines.append(f"    ({pred['name']}{(' ' + params) if params else ''})")
    lines.append("  )")
    for op in domain["operators"]:
        params = " ".join(f"{p['name']} - {p['type']}" for p in op["parameters"])
        preconds = [pddl_atom(atom) for atom in op["preconditions_pos"]]
        preconds.extend(f"(not {pddl_atom(atom)})" for atom in op["preconditions_neg"])
        effects = [pddl_atom(atom) for atom in op["effects_add"]]
        effects.extend(f"(not {pddl_atom(atom)})" for atom in op["effects_del"])
        precondition_expr = f"(and {' '.join(preconds)})" if preconds else "(and)"
        effect_expr = f"(and {' '.join(effects)})" if effects else "(and)"
        lines.extend(
            [
                f"  (:action {op['name']}",
                f"    :parameters ({params})",
                f"    :precondition {precondition_expr}",
                f"    :effect {effect_expr}",
                "  )",
            ]
        )
    lines.append(")")
    return "\n".join(lines) + "\n"


def operator_name_for_step(step: Mapping[str, Any], split_place: bool = True) -> str:
    op = str(step.get("operator"))
    if op != "place":
        return op
    if not split_place:
        return "place"
    relation = str((step.get("bindings") or {}).get("relation", "on-or-in"))
    if relation == "in":
        return "place_in"
    if relation == "on":
        return "place_on"
    return "place_on_or_in_unresolved"


def normalize_step_for_domain(step: Mapping[str, Any], split_place: bool = True) -> dict[str, Any]:
    normalized = dict(step)
    normalized["p4_operator"] = operator_name_for_step(step, split_place=split_place)
    return normalized


def merge_server_push_steps(
    skeletons: Sequence[Mapping[str, Any]],
    at_region_eval: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Add server-validated push_to_region steps to the three push demos."""
    eval_demos = dict((at_region_eval or {}).get("demos", {}))
    patched: list[dict[str, Any]] = []
    for skeleton in skeletons:
        item = dict(skeleton)
        steps = [dict(step) for step in skeleton.get("steps", [])]
        task = str(skeleton.get("task", ""))
        demo = str(skeleton.get("demo", ""))
        if task == "push_the_plate_to_the_front_of_the_stove" and demo in eval_demos:
            info = dict(eval_demos[demo])
            if info.get("final_ok") is True:
                t_end = int(info.get("T", 0)) - 1
                steps.append(
                    {
                        "operator": "push_to_region",
                        "bindings": {"obj": "plate_1", "region": "main_table_stove_front_region"},
                        "t_start": max(0, t_end - int(info.get("atr_frames", 1)) + 1),
                        "t_end": t_end,
                        "source": "server_validated_at_region_patch",
                        "confidence": "conditional",
                        "risk_notes": ["at_region_validated_server", "support_count_3"],
                    }
                )
        steps.sort(key=lambda step: (int(step.get("t_start", 10**9)), int(step.get("t_end", 10**9))))
        item["steps"] = steps
        item["operator_sequence"] = [str(step.get("operator")) for step in steps]
        item["p4_operator_sequence"] = [operator_name_for_step(step) for step in steps]
        patched.append(item)
    return patched


def step_satisfies_goal_p4(step: Mapping[str, Any], goal: Mapping[str, Any], split_place: bool = True) -> bool:
    op = operator_name_for_step(step, split_place=split_place)
    bindings = dict(step.get("bindings") or {})
    atom = tuple(goal["atom"])
    aliases = goal.get("arg_aliases") or []
    pred = atom[0]
    if pred == "open" and op == "open":
        return _value_matches_aliases(bindings.get("drawer"), aliases[0] if aliases else [atom[1]])
    if pred == "activated" and op == "turn_on":
        return _value_matches_aliases(bindings.get("device"), aliases[0] if aliases else [atom[1]])
    if pred == "on" and op in {"place_on", "place_on_or_in_unresolved"}:
        return _binary_binding_matches_goal(bindings, "target", atom, aliases)
    if pred == "in" and op == "place_in":
        return _binary_binding_matches_goal(bindings, "target", atom, aliases)
    if pred in {"on", "in"} and op == "place":
        return _binary_binding_matches_goal(bindings, "target", atom, aliases)
    if pred == "at_region" and op == "push_to_region":
        return _binary_binding_matches_goal(bindings, "region", atom, aliases)
    return False


def _binary_binding_matches_goal(
    bindings: Mapping[str, Any],
    second_key: str,
    atom: Sequence[str],
    aliases: Sequence[Sequence[str]],
) -> bool:
    obj_aliases = aliases[0] if aliases else [atom[1]]
    target_aliases = aliases[1] if len(aliases) > 1 else [atom[2]]
    return _value_matches_aliases(bindings.get("obj"), obj_aliases) and _value_matches_aliases(
        bindings.get(second_key), target_aliases
    )


def replay_p4_domain(
    skeletons: Sequence[Mapping[str, Any]],
    goals_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    domain: Mapping[str, Any],
    split_place: bool = True,
) -> dict[str, Any]:
    operator_names = {str(op["name"]) for op in domain["operators"]}
    rows: list[dict[str, Any]] = []
    for skeleton in skeletons:
        task = str(skeleton.get("task", ""))
        goals = list(goals_by_task.get(task, []))
        steps = [normalize_step_for_domain(step, split_place=split_place) for step in skeleton.get("steps", [])]
        invalid_steps = [step for step in steps if step["p4_operator"] not in operator_names]
        matched: list[Atom] = []
        missing: list[Atom] = []
        for goal in goals:
            atom = tuple(goal["atom"])
            if any(step_satisfies_goal_p4(step, goal, split_place=split_place) for step in steps):
                matched.append(atom)
            else:
                missing.append(atom)

        if not goals:
            status = "no_goal_mapping"
            reason = "missing_bddl_goal_mapping"
        elif invalid_steps:
            status = "invalid_plan"
            reason = "operator_not_in_domain"
        elif missing:
            status = "failed"
            reason = "missing_goal_effect"
        else:
            status = "covered"
            reason = "all_goals_matched_by_p4_domain"
        rows.append(
            {
                "demo_id": skeleton.get("demo_id"),
                "file": skeleton.get("file"),
                "demo": skeleton.get("demo"),
                "task": task,
                "goals": [atom_to_list(tuple(goal["atom"])) for goal in goals],
                "matched_goals": [atom_to_list(atom) for atom in matched],
                "missing_goals": [atom_to_list(atom) for atom in missing],
                "operator_sequence": [step["p4_operator"] for step in steps],
                "invalid_operators": [step["p4_operator"] for step in invalid_steps],
                "status": status,
                "reason": reason,
            }
        )

    status_counts = Counter(row["status"] for row in rows)
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        per_task[str(row["task"])][str(row["status"])] += 1
    return {
        "summary": {
            "demo_count": len(rows),
            "covered": status_counts.get("covered", 0),
            "failed": status_counts.get("failed", 0),
            "invalid_plan": status_counts.get("invalid_plan", 0),
            "no_goal_mapping": status_counts.get("no_goal_mapping", 0),
            "status_counts": dict(status_counts),
            "operator_counts": dict(Counter(op for row in rows for op in row["operator_sequence"])),
            "per_task": {task: dict(counts) for task, counts in sorted(per_task.items())},
        },
        "demos": rows,
    }


def render_replay_report(replay: Mapping[str, Any], domain: Mapping[str, Any]) -> str:
    summary = replay["summary"]
    lines = [
        "# P4 Typed Domain Replay Report",
        "",
        f"- Domain: `{domain['name']}`",
        f"- Demo count: {summary['demo_count']}",
        f"- Covered: {summary['covered']}/{summary['demo_count']}",
        f"- Failed: {summary['failed']}",
        f"- Invalid plans: {summary['invalid_plan']}",
        f"- Operator counts: `{summary['operator_counts']}`",
        "",
        "## Operators",
        "",
        "| Operator | Support | Confidence |",
        "|---|---:|---|",
    ]
    for op in domain["operators"]:
        lines.append(f"| {op['name']} | {op['support_count']} | {op['confidence']} |")
    lines.extend(["", "## Per-task status", "", "| Task | covered | failed | invalid |", "|---|---:|---:|---:|"])
    for task, counts in summary["per_task"].items():
        lines.append(f"| {task} | {counts.get('covered', 0)} | {counts.get('failed', 0)} | {counts.get('invalid_plan', 0)} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This report validates symbolic goal replay over P3/P4 artifacts.",
            "- It does not claim physical robot execution success.",
            "- `push_to_region` is included only when server-side `at_region` validation is present.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_replay_variant(name: str, replay: Mapping[str, Any], notes: Sequence[str]) -> dict[str, Any]:
    summary = dict(replay["summary"])
    demo_count = int(summary.get("demo_count", 0))
    covered = int(summary.get("covered", 0))
    return {
        "name": name,
        "demo_count": demo_count,
        "covered": covered,
        "coverage": covered / demo_count if demo_count else 0.0,
        "failed": int(summary.get("failed", 0)),
        "invalid_plan": int(summary.get("invalid_plan", 0)),
        "no_goal_mapping": int(summary.get("no_goal_mapping", 0)),
        "operator_counts": summary.get("operator_counts", {}),
        "status_counts": summary.get("status_counts", {}),
        "notes": list(notes),
    }


def run_p4_ablation_suite(
    p3_dir: Path,
    p3_clauses: Sequence[Mapping[str, Any]],
    goals_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    push_candidate: Mapping[str, Any] | None,
    at_region_eval: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Run lightweight P4 ablations over existing symbolic artifacts."""
    repaired_skeletons = read_jsonl(p3_dir / "p3_plan_skeletons_v2.jsonl")
    local_skeletons_path = p3_dir / "p3_plan_skeletons.jsonl"
    local_skeletons = read_jsonl(local_skeletons_path) if local_skeletons_path.exists() else repaired_skeletons

    typed_domain = canonical_p4_domain(p3_clauses, push_candidate, split_place=True)
    fused_domain = canonical_p4_domain(p3_clauses, push_candidate, split_place=False)

    variants: list[dict[str, Any]] = []

    full_replay = replay_p4_domain(
        merge_server_push_steps(repaired_skeletons, at_region_eval),
        goals_by_task,
        typed_domain,
        split_place=True,
    )
    variants.append(
        summarize_replay_variant(
            "full_typed_split",
            full_replay,
            ["P3 v2 repaired skeletons", "server at_region patch", "place split into place_on/place_in"],
        )
    )

    no_push_replay = replay_p4_domain(repaired_skeletons, goals_by_task, typed_domain, split_place=True)
    variants.append(
        summarize_replay_variant(
            "no_push_patch",
            no_push_replay,
            ["P3 v2 repaired skeletons", "server at_region patch disabled"],
        )
    )

    no_goal_repair_replay = replay_p4_domain(
        merge_server_push_steps(local_skeletons, at_region_eval),
        goals_by_task,
        typed_domain,
        split_place=True,
    )
    variants.append(
        summarize_replay_variant(
            "no_goal_repair",
            no_goal_repair_replay,
            ["P3 v1 local skeletons", "server at_region patch enabled", "final-state goal repair disabled"],
        )
    )

    local_only_replay = replay_p4_domain(local_skeletons, goals_by_task, typed_domain, split_place=True)
    variants.append(
        summarize_replay_variant(
            "local_only_effect",
            local_only_replay,
            ["P3 v1 local skeletons", "server at_region patch disabled", "final-state goal repair disabled"],
        )
    )

    fused_place_replay = replay_p4_domain(
        merge_server_push_steps(repaired_skeletons, at_region_eval),
        goals_by_task,
        fused_domain,
        split_place=False,
    )
    variants.append(
        summarize_replay_variant(
            "fused_place_full",
            fused_place_replay,
            ["P3 v2 repaired skeletons", "server at_region patch", "place remains fused as place(on-or-in)"],
        )
    )

    full_covered = variants[0]["covered"] if variants else 0
    for variant in variants:
        variant["delta_covered_vs_full"] = int(variant["covered"]) - int(full_covered)

    report = {
        "stage": "P4-B",
        "variant_count": len(variants),
        "full_variant": "full_typed_split",
        "variants": variants,
        "interpretation": {
            "server_at_region_patch_gain": int(variants[0]["covered"]) - int(variants[1]["covered"]),
            "goal_repair_gain": int(variants[0]["covered"]) - int(variants[2]["covered"]),
            "local_to_full_gain": int(variants[0]["covered"]) - int(variants[3]["covered"]),
            "typed_split_coverage_delta": int(variants[0]["covered"]) - int(variants[4]["covered"]),
            "typed_split_note": (
                "Coverage can match fused_place_full, but split place_on/place_in is the PDDL-clean representation "
                "used for the final domain."
            ),
        },
    }
    return report, render_ablation_report(report)


def render_ablation_report(report: Mapping[str, Any]) -> str:
    variants = list(report["variants"])
    lines = [
        "# P4-B Baseline and Ablation Report",
        "",
        f"- Variant count: {report['variant_count']}",
        f"- Full variant: `{report['full_variant']}`",
        "",
        "## Summary",
        "",
        "| Variant | Covered | Failed | Invalid | Delta vs full | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in variants:
        notes = "; ".join(item.get("notes", []))
        lines.append(
            f"| {item['name']} | {item['covered']}/{item['demo_count']} | {item['failed']} | "
            f"{item['invalid_plan']} | {item['delta_covered_vs_full']} | {notes} |"
        )
    interp = report["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Server at_region patch gain: {interp['server_at_region_patch_gain']} demos.",
            f"- Goal-repair gain: {interp['goal_repair_gain']} demos.",
            f"- Local-only to full gain: {interp['local_to_full_gain']} demos.",
            f"- Typed split coverage delta vs fused place: {interp['typed_split_coverage_delta']} demos.",
            f"- Typed split note: {interp['typed_split_note']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_p4_pipeline(
    p3_dir: Path,
    package_dir: Path,
    server_validation_dir: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    p3_clauses = read_json(p3_dir / "p3_operator_clauses_v2.json")
    skeletons = read_jsonl(p3_dir / "p3_plan_skeletons_v2.jsonl")

    push_candidate: Mapping[str, Any] | None = None
    at_region_eval: Mapping[str, Any] | None = None
    if server_validation_dir and (server_validation_dir / "p3_push_operator_candidate.json").exists():
        push_candidate = read_json(server_validation_dir / "p3_push_operator_candidate.json")
    if server_validation_dir and (server_validation_dir / "at_region_replacement_eval.json").exists():
        at_region_eval = read_json(server_validation_dir / "at_region_replacement_eval.json")

    domain = canonical_p4_domain(p3_clauses, push_candidate)
    patched_skeletons = merge_server_push_steps(skeletons, at_region_eval)
    goals_by_task = goal_specs_from_mapping(package_dir)
    replay = replay_p4_domain(patched_skeletons, goals_by_task, domain)
    ablations, ablation_md = run_p4_ablation_suite(
        p3_dir=p3_dir,
        p3_clauses=p3_clauses,
        goals_by_task=goals_by_task,
        push_candidate=push_candidate,
        at_region_eval=at_region_eval,
    )

    write_json(out_dir / "p4_domain.json", domain)
    write_text(out_dir / "p4_domain.pddl", render_pddl_like_domain(domain))
    write_jsonl(out_dir / "p4_plan_skeletons.jsonl", patched_skeletons)
    write_json(out_dir / "p4_domain_replay_eval.json", replay)
    write_text(out_dir / "p4_domain_replay_report.md", render_replay_report(replay, domain))
    write_json(out_dir / "p4_ablation_summary.json", ablations)
    write_text(out_dir / "p4_ablation_report.md", ablation_md)
    summary = {
        "ok": replay["summary"]["covered"] == replay["summary"]["demo_count"],
        "stage": "P4",
        "p3_dir": str(p3_dir),
        "package_dir": str(package_dir),
        "server_validation_dir": str(server_validation_dir) if server_validation_dir else None,
        "out_dir": str(out_dir),
        "domain_operator_count": len(domain["operators"]),
        "demo_count": replay["summary"]["demo_count"],
        "covered": replay["summary"]["covered"],
        "failed": replay["summary"]["failed"],
        "invalid_plan": replay["summary"]["invalid_plan"],
        "operator_counts": replay["summary"]["operator_counts"],
        "ablation_interpretation": ablations["interpretation"],
        "status": "P4_PASS" if replay["summary"]["covered"] == replay["summary"]["demo_count"] else "P4_NEEDS_REPAIR",
    }
    write_json(out_dir / "p4_pipeline_summary.json", summary)
    return summary
