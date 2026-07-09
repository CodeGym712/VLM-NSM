"""P5 symbolic planning audit over the P4 typed domain.

P4 validates that a typed domain can replay demonstration-level goal effects.
P5-A takes the next step: construct a planning problem from the initial
PredicateTrace state and search for a plan with the exported domain.  This is
still symbolic and offline; it does not execute a robot policy.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vlm_nsm.research.p3_symbolic_induction import (
    Atom,
    atom_to_list,
    atom_to_str,
    goal_specs_from_mapping,
    read_json,
    read_jsonl,
    sort_atoms,
    write_json,
    write_jsonl,
)
from vlm_nsm.research.p4_domain_export import write_text


State = frozenset[Atom]


MOVABLE_NAME_PREFIXES = (
    "akita_",
    "cream_cheese",
    "wine_bottle",
    "plate",
    "cookies",
    "glazed_",
)


@dataclass(frozen=True)
class GroundAction:
    name: str
    bindings: tuple[tuple[str, str], ...]
    preconditions_pos: tuple[Atom, ...]
    preconditions_neg: tuple[Atom, ...]
    effects_add: tuple[Atom, ...]
    effects_del: tuple[Atom, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.name,
            "bindings": dict(self.bindings),
            "preconditions_pos": [atom_to_list(atom) for atom in self.preconditions_pos],
            "preconditions_neg": [atom_to_list(atom) for atom in self.preconditions_neg],
            "effects_add": [atom_to_list(atom) for atom in self.effects_add],
            "effects_del": [atom_to_list(atom) for atom in self.effects_del],
        }


def atom_from_list(row: Sequence[str]) -> Atom:
    return tuple(str(item) for item in row)


def substitute_atom(template: Sequence[str], bindings: Mapping[str, str]) -> Atom:
    values: list[str] = []
    for item in template:
        value = str(item)
        values.append(bindings.get(value, value))
    return tuple(values)


def applicable(action: GroundAction, state: State) -> bool:
    return all(atom in state for atom in action.preconditions_pos) and all(
        atom not in state for atom in action.preconditions_neg
    )


def apply_action(action: GroundAction, state: State) -> State:
    updated = set(state)
    updated.difference_update(action.effects_del)
    updated.update(action.effects_add)
    return frozenset(updated)


def goals_satisfied(state: State, goals: Sequence[Mapping[str, Any]]) -> bool:
    return all(any(atom_satisfies_goal_strict(atom, goal) for atom in state) for goal in goals)


def atom_satisfies_goal_strict(atom: Atom, goal: Mapping[str, Any]) -> bool:
    """Exact alias matching for planner audits.

    P3/P4 replay uses permissive alias matching to bridge simulator naming
    differences.  P5 searches over symbolic states, so prefix matching is too
    loose: e.g. bottom drawer must not satisfy a top-drawer goal.
    """
    goal_atom = tuple(goal["atom"])
    aliases = goal.get("arg_aliases") or []
    if atom[0] != goal_atom[0] or len(atom) != len(goal_atom):
        return False
    pred = goal_atom[0]
    for idx in range(1, len(goal_atom)):
        candidates = set(str(item) for item in (aliases[idx - 1] if idx - 1 < len(aliases) else [goal_atom[idx]]))
        candidates.add(str(goal_atom[idx]))
        value = str(atom[idx])
        if value in candidates:
            continue
        if pred in {"activated", "deactivated"} and _device_alias_matches(value, candidates):
            continue
        if pred == "at_region" and idx == 1 and _body_alias_matches(value, candidates):
            continue
        if pred in {"on", "in"} and idx == 1 and _body_alias_matches(value, candidates):
            continue
        if pred == "on" and idx == 2 and _body_alias_matches(value, candidates):
            continue
        if pred == "in" and idx == 2 and _body_alias_matches(value, candidates):
            continue
        if pred == "open" and idx == 1 and _drawer_level_alias_matches(value, candidates):
            continue
        else:
            return False
    return True


def _device_alias_matches(value: str, candidates: set[str]) -> bool:
    for candidate in candidates:
        if value in {candidate, f"{candidate}_button", f"{candidate}_main", f"{candidate}_burner_plate"}:
            return True
        if candidate in {value, f"{value}_button", f"{value}_main", f"{value}_burner_plate"}:
            return True
    return False


def _body_alias_matches(value: str, candidates: set[str]) -> bool:
    for candidate in candidates:
        if value == candidate:
            return True
        if value == f"{candidate}_main" or candidate == f"{value}_main":
            return True
    return False


def _drawer_level_alias_matches(value: str, candidates: set[str]) -> bool:
    # Keep drawer levels exact; only bridge explicit region/level spelling.
    for candidate in candidates:
        if value == candidate:
            return True
        if candidate.endswith("_region") and value == candidate[: -len("_region")] + "_level":
            return True
        if value.endswith("_region") and candidate == value[: -len("_region")] + "_level":
            return True
    return False


def group_initial_states(normalized_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        grouped[f"{row['file']}::{row['demo']}"].append(row)
    initial: dict[str, dict[str, Any]] = {}
    for demo_id, rows in grouped.items():
        first = min(rows, key=lambda item: int(item["t"]))
        initial[demo_id] = {
            "demo_id": demo_id,
            "file": first["file"],
            "demo": first["demo"],
            "task": first["task"],
            "state": frozenset(atom_from_list(atom) for atom in first.get("trusted_predicates", [])),
            "t": int(first["t"]),
            "T": int(first["T"]),
        }
    return initial


def infer_type_universe(
    initial_state: Iterable[Atom],
    goals: Sequence[Mapping[str, Any]],
    skeleton_steps: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, set[str]]:
    universe: dict[str, set[str]] = {
        "object": set(),
        "support": set(),
        "container": set(),
        "openable": set(),
        "device": set(),
        "region": set(),
    }

    def add_atom(atom: Atom) -> None:
        if not atom:
            return
        pred = atom[0]
        if pred == "holding" and len(atom) == 2:
            universe["object"].add(atom[1])
        elif pred == "on" and len(atom) == 3:
            universe["object"].add(atom[1])
            universe["support"].add(atom[2])
        elif pred == "in" and len(atom) == 3:
            universe["object"].add(atom[1])
            universe["container"].add(atom[2])
        elif pred in {"open", "closed"} and len(atom) == 2:
            universe["openable"].add(atom[1])
            universe["container"].add(atom[1])
        elif pred in {"activated", "deactivated"} and len(atom) == 2:
            universe["device"].add(atom[1])
        elif pred == "at_region" and len(atom) == 3:
            universe["object"].add(atom[1])
            universe["region"].add(atom[2])

    for atom in initial_state:
        add_atom(atom)

    for goal in goals:
        atom = tuple(goal["atom"])
        add_atom(atom)
        aliases = goal.get("arg_aliases") or []
        for idx, candidates in enumerate(aliases):
            if idx + 1 >= len(atom):
                continue
            pred = atom[0]
            if pred in {"on", "in", "at_region"} and idx == 0:
                universe["object"].update(str(item) for item in candidates)
            elif pred == "on" and idx == 1:
                universe["support"].update(str(item) for item in candidates)
            elif pred == "in" and idx == 1:
                universe["container"].update(str(item) for item in candidates)
            elif pred == "at_region" and idx == 1:
                universe["region"].update(str(item) for item in candidates)
            elif pred == "open":
                universe["openable"].update(str(item) for item in candidates)
                universe["container"].update(str(item) for item in candidates)
            elif pred == "activated":
                universe["device"].update(str(item) for item in candidates)

    for step in skeleton_steps or []:
        bindings = dict(step.get("bindings") or {})
        op = str(step.get("operator"))
        if "obj" in bindings:
            universe["object"].add(str(bindings["obj"]))
        if "target" in bindings:
            target = str(bindings["target"])
            relation = str(bindings.get("relation", "on"))
            if relation == "in":
                universe["container"].add(target)
            else:
                universe["support"].add(target)
        if "drawer" in bindings:
            universe["openable"].add(str(bindings["drawer"]))
            universe["container"].add(str(bindings["drawer"]))
        if "device" in bindings:
            universe["device"].add(str(bindings["device"]))
        if "region" in bindings:
            universe["region"].add(str(bindings["region"]))

    # Conservative cross-typing for LIBERO fixtures and receptacles.
    universe["support"].update(universe["container"])
    universe["container"].update(name for name in universe["support"] if "cabinet" in name or "bowl" in name)
    return universe


def ground_actions(domain: Mapping[str, Any], universe: Mapping[str, set[str]]) -> list[GroundAction]:
    actions: list[GroundAction] = []
    for op in domain.get("operators", []):
        params = list(op.get("parameters", []))
        if not params:
            binding_rows = [()]
        else:
            candidate_lists: list[list[str]] = []
            for param in params:
                candidates = sorted(universe.get(str(param["type"]), set()))
                if not candidates:
                    candidate_lists = []
                    break
                candidate_lists.append(candidates)
            binding_rows = list(product(*candidate_lists)) if candidate_lists else []

        for values in binding_rows:
            bindings = {str(param["name"]): str(value) for param, value in zip(params, values)}
            actions.append(
                GroundAction(
                    name=str(op["name"]),
                    bindings=tuple(sorted(bindings.items())),
                    preconditions_pos=tuple(substitute_atom(atom, bindings) for atom in op.get("preconditions_pos", [])),
                    preconditions_neg=tuple(substitute_atom(atom, bindings) for atom in op.get("preconditions_neg", [])),
                    effects_add=tuple(substitute_atom(atom, bindings) for atom in op.get("effects_add", [])),
                    effects_del=tuple(substitute_atom(atom, bindings) for atom in op.get("effects_del", [])),
                )
            )
    return sorted(actions, key=lambda action: (action.name, action.bindings))


def bfs_plan(
    initial_state: State,
    goals: Sequence[Mapping[str, Any]],
    actions: Sequence[GroundAction],
    max_depth: int = 4,
    max_expansions: int = 20000,
) -> tuple[list[GroundAction] | None, dict[str, Any]]:
    if goals_satisfied(initial_state, goals):
        return [], {"expanded": 0, "visited": 1, "reason": "goals_initially_satisfied"}

    queue = deque([(initial_state, [])])
    visited = {initial_state}
    expanded = 0

    while queue:
        state, plan = queue.popleft()
        if len(plan) >= max_depth:
            continue
        expanded += 1
        if expanded > max_expansions:
            return None, {"expanded": expanded, "visited": len(visited), "reason": "max_expansions_exceeded"}
        for action in actions:
            if not applicable(action, state):
                continue
            next_state = apply_action(action, state)
            if next_state in visited:
                continue
            next_plan = [*plan, action]
            if goals_satisfied(next_state, goals):
                return next_plan, {"expanded": expanded, "visited": len(visited) + 1, "reason": "plan_found"}
            visited.add(next_state)
            queue.append((next_state, next_plan))

    return None, {"expanded": expanded, "visited": len(visited), "reason": "search_exhausted"}


def skeleton_steps_by_demo(path: Path) -> dict[str, list[Mapping[str, Any]]]:
    rows = read_jsonl(path)
    return {str(row["demo_id"]): list(row.get("steps", [])) for row in rows}


def run_p5_planning_audit(
    p3_dir: Path,
    p4_dir: Path,
    package_dir: Path,
    out_dir: Path,
    max_depth: int = 4,
    max_expansions: int = 20000,
) -> dict[str, Any]:
    domain = read_json(p4_dir / "p4_domain.json")
    normalized_rows = read_jsonl(p3_dir / "p3_normalized_traces.jsonl")
    initial_by_demo = group_initial_states(normalized_rows)
    goals_by_task = goal_specs_from_mapping(package_dir)
    steps_by_demo = skeleton_steps_by_demo(p4_dir / "p4_plan_skeletons.jsonl")

    rows: list[dict[str, Any]] = []
    for demo_id, problem in sorted(initial_by_demo.items()):
        task = str(problem["task"])
        goals = list(goals_by_task.get(task, []))
        initial_state = problem["state"]
        universe = infer_type_universe(initial_state, goals, steps_by_demo.get(demo_id, []))
        actions = ground_actions(domain, universe)
        plan, stats = bfs_plan(
            initial_state,
            goals,
            actions,
            max_depth=max_depth,
            max_expansions=max_expansions,
        )
        status = "solved" if plan is not None else "failed"
        missing_goals = []
        if plan is None:
            missing_goals = [
                tuple(goal["atom"])
                for goal in goals
                if not any(atom_satisfies_goal_strict(atom, goal) for atom in initial_state)
            ]
        rows.append(
            {
                "demo_id": demo_id,
                "file": problem["file"],
                "demo": problem["demo"],
                "task": task,
                "status": status,
                "reason": stats["reason"],
                "plan_length": len(plan) if plan is not None else None,
                "plan": [action.to_dict() for action in plan] if plan is not None else [],
                "goals": [atom_to_list(tuple(goal["atom"])) for goal in goals],
                "missing_goals_if_failed": [atom_to_list(atom) for atom in missing_goals],
                "initial_atom_count": len(initial_state),
                "initial_atoms": [atom_to_list(atom) for atom in sort_atoms(initial_state)],
                "type_counts": {key: len(value) for key, value in sorted(universe.items())},
                "ground_action_count": len(actions),
                "search": stats,
            }
        )

    status_counts = Counter(row["status"] for row in rows)
    plan_lengths = [int(row["plan_length"]) for row in rows if row["plan_length"] is not None]
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        per_task[str(row["task"])][str(row["status"])] += 1

    summary = {
        "ok": status_counts.get("solved", 0) == len(rows),
        "stage": "P5-A",
        "demo_count": len(rows),
        "solved": status_counts.get("solved", 0),
        "failed": status_counts.get("failed", 0),
        "status_counts": dict(status_counts),
        "max_depth": max_depth,
        "max_expansions": max_expansions,
        "plan_length_counts": dict(Counter(plan_lengths)),
        "avg_plan_length": sum(plan_lengths) / len(plan_lengths) if plan_lengths else 0.0,
        "per_task": {task: dict(counts) for task, counts in sorted(per_task.items())},
        "status": "P5_INITIAL_PLANNING_PASS" if status_counts.get("solved", 0) == len(rows) else "P5_NEEDS_REPAIR",
    }

    write_json(out_dir / "p5_initial_planning_summary.json", summary)
    write_jsonl(out_dir / "p5_initial_planning_results.jsonl", rows)
    write_text(out_dir / "p5_initial_planning_report.md", render_p5_report(summary, rows))
    return summary


def render_p5_report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# P5-A Initial-State Symbolic Planning Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Demo count: {summary['demo_count']}",
        f"- Solved: {summary['solved']}/{summary['demo_count']}",
        f"- Failed: {summary['failed']}",
        f"- Max depth: {summary['max_depth']}",
        f"- Plan length counts: `{summary['plan_length_counts']}`",
        f"- Average plan length: {summary['avg_plan_length']:.2f}",
        "",
        "## Per-task status",
        "",
        "| Task | solved | failed |",
        "|---|---:|---:|",
    ]
    for task, counts in summary["per_task"].items():
        lines.append(f"| {task} | {counts.get('solved', 0)} | {counts.get('failed', 0)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "P5-A performs symbolic planner search from each demo's initial PredicateTrace state.",
            "It does not reuse the demonstration skeleton as the plan, and it does not execute a robot policy.",
            "This is the first check that the P4 typed domain is planner-usable from initial states.",
            "",
            "## Example plans",
            "",
        ]
    )
    for row in rows[:8]:
        plan = " -> ".join(str(step["operator"]) for step in row["plan"])
        lines.extend(
            [
                f"### {row['demo_id']}",
                "",
                f"- Task: `{row['task']}`",
                f"- Status: `{row['status']}`",
                f"- Plan length: `{row['plan_length']}`",
                f"- Plan: `{plan}`",
                "",
            ]
        )
    return "\n".join(lines)


def is_likely_movable(name: str) -> bool:
    return name.startswith(MOVABLE_NAME_PREFIXES)


def movable_symbols_in_state(state: Iterable[Atom]) -> set[str]:
    symbols: set[str] = set()
    for atom in state:
        if len(atom) >= 2 and atom[0] in {"holding", "on", "in", "at_region"} and is_likely_movable(atom[1]):
            symbols.add(atom[1])
        if len(atom) == 3 and atom[0] == "on" and is_likely_movable(atom[2]):
            symbols.add(atom[2])
        if len(atom) == 3 and atom[0] == "in" and is_likely_movable(atom[2]):
            symbols.add(atom[2])
    return symbols


def required_movable_goal_symbols(goals: Sequence[Mapping[str, Any]]) -> set[str]:
    required: set[str] = set()
    for goal in goals:
        atom = tuple(goal["atom"])
        if len(atom) >= 2 and atom[0] in {"on", "in", "at_region"} and is_likely_movable(atom[1]):
            required.add(atom[1])
        if len(atom) == 3 and atom[0] in {"on", "in"} and is_likely_movable(atom[2]):
            required.add(atom[2])
    return required


def validate_plan_grounding(plan: Sequence[GroundAction], initial_state: State) -> list[str]:
    present = movable_symbols_in_state(initial_state)
    issues: list[str] = []
    for idx, action in enumerate(plan):
        bindings = dict(action.bindings)
        obj = bindings.get("?obj")
        if action.name in {"pick", "push_to_region", "place_on", "place_in"} and obj and is_likely_movable(obj):
            if obj not in present:
                issues.append(f"step_{idx}:{action.name}:absent_object:{obj}")
        target = bindings.get("?target") or bindings.get("?container")
        if action.name in {"place_on", "place_in"} and target and is_likely_movable(target):
            if target not in present:
                issues.append(f"step_{idx}:{action.name}:absent_target:{target}")
    return issues


def run_p5_composition_audit(
    p3_dir: Path,
    p4_dir: Path,
    package_dir: Path,
    out_dir: Path,
    max_depth: int = 4,
    max_expansions: int = 20000,
) -> dict[str, Any]:
    """Cross product initial states with task goals for composition planning.

    This is a transfer/composition audit, not a retraining split.  It asks:
    given a source demo's initial symbolic state, which other task goals are
    plausibly plannable with the current typed domain?
    """
    domain = read_json(p4_dir / "p4_domain.json")
    normalized_rows = read_jsonl(p3_dir / "p3_normalized_traces.jsonl")
    initial_by_demo = group_initial_states(normalized_rows)
    goals_by_task = goal_specs_from_mapping(package_dir)
    target_tasks = sorted(goals_by_task)

    rows: list[dict[str, Any]] = []
    for demo_id, problem in sorted(initial_by_demo.items()):
        source_task = str(problem["task"])
        initial_state = problem["state"]
        present_movable = movable_symbols_in_state(initial_state)
        for target_task in target_tasks:
            goals = list(goals_by_task.get(target_task, []))
            required = required_movable_goal_symbols(goals)
            category = "same_task" if target_task == source_task else "cross_task"
            if not required.issubset(present_movable):
                rows.append(
                    {
                        "source_demo_id": demo_id,
                        "source_task": source_task,
                        "target_task": target_task,
                        "category": category,
                        "status": "skipped_absent_object",
                        "required_movable": sorted(required),
                        "present_movable": sorted(present_movable),
                        "missing_movable": sorted(required - present_movable),
                        "plan_length": None,
                        "plan": [],
                        "reason": "required_movable_not_in_initial_state",
                    }
                )
                continue

            universe = infer_type_universe(initial_state, goals, skeleton_steps=[])
            actions = ground_actions(domain, universe)
            plan, stats = bfs_plan(
                initial_state,
                goals,
                actions,
                max_depth=max_depth,
                max_expansions=max_expansions,
            )
            if plan is None:
                status = "failed"
                issues: list[str] = []
            else:
                issues = validate_plan_grounding(plan, initial_state)
                status = "invalid_plan" if issues else "solved"
            rows.append(
                {
                    "source_demo_id": demo_id,
                    "source_task": source_task,
                    "target_task": target_task,
                    "category": category,
                    "status": status,
                    "required_movable": sorted(required),
                    "present_movable": sorted(present_movable),
                    "missing_movable": [],
                    "plan_length": len(plan) if plan is not None else None,
                    "plan": [action.to_dict() for action in plan] if plan is not None else [],
                    "grounding_issues": issues,
                    "ground_action_count": len(actions),
                    "reason": stats["reason"],
                    "search": stats,
                }
            )

    summary = summarize_composition_rows(rows, max_depth=max_depth, max_expansions=max_expansions)
    write_json(out_dir / "p5_composition_planning_summary.json", summary)
    write_jsonl(out_dir / "p5_composition_planning_results.jsonl", rows)
    write_text(out_dir / "p5_composition_planning_report.md", render_composition_report(summary, rows))
    return summary


def summarize_composition_rows(rows: Sequence[Mapping[str, Any]], max_depth: int, max_expansions: int) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in rows)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category_counts[str(row["category"])][str(row["status"])] += 1
    attempted = len(rows) - status_counts.get("skipped_absent_object", 0)
    solved = status_counts.get("solved", 0)
    return {
        "ok": status_counts.get("invalid_plan", 0) == 0,
        "stage": "P5-B",
        "problem_count": len(rows),
        "attempted": attempted,
        "solved": solved,
        "failed": status_counts.get("failed", 0),
        "invalid_plan": status_counts.get("invalid_plan", 0),
        "skipped_absent_object": status_counts.get("skipped_absent_object", 0),
        "attempted_solve_rate": solved / attempted if attempted else 0.0,
        "status_counts": dict(status_counts),
        "category_status_counts": {category: dict(counts) for category, counts in sorted(category_counts.items())},
        "max_depth": max_depth,
        "max_expansions": max_expansions,
        "status": "P5_COMPOSITION_AUDIT_PASS" if status_counts.get("invalid_plan", 0) == 0 else "P5_COMPOSITION_HAS_INVALID_PLANS",
        "boundary": "This is a compatibility-filtered cross-task symbolic planning audit, not a held-out retraining experiment.",
    }


def render_composition_report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# P5-B Composition Planning Audit Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Problems: {summary['problem_count']}",
        f"- Attempted: {summary['attempted']}",
        f"- Solved: {summary['solved']}",
        f"- Failed: {summary['failed']}",
        f"- Invalid plans: {summary['invalid_plan']}",
        f"- Skipped absent-object: {summary['skipped_absent_object']}",
        f"- Attempted solve rate: {summary['attempted_solve_rate']:.3f}",
        "",
        "## Category status",
        "",
        "| Category | solved | failed | invalid | skipped |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, counts in summary["category_status_counts"].items():
        lines.append(
            f"| {category} | {counts.get('solved', 0)} | {counts.get('failed', 0)} | "
            f"{counts.get('invalid_plan', 0)} | {counts.get('skipped_absent_object', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit crosses each source demo initial state with every known task goal.",
            "Problems whose required movable objects are absent from the initial state are skipped,",
            "so the solve rate should be read as a compatibility-filtered symbolic planning score.",
            "It is not yet a leave-one-task-template-out retraining experiment.",
            "",
            "## Example solved cross-task plans",
            "",
        ]
    )
    examples = [row for row in rows if row["category"] == "cross_task" and row["status"] == "solved"][:8]
    for row in examples:
        plan = " -> ".join(str(step["operator"]) for step in row["plan"])
        lines.extend(
            [
                f"### {row['source_demo_id']} -> {row['target_task']}",
                "",
                f"- Source task: `{row['source_task']}`",
                f"- Plan length: `{row['plan_length']}`",
                f"- Plan: `{plan}`",
                "",
            ]
        )
    return "\n".join(lines)
