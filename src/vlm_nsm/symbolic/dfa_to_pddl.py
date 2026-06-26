# DFA→PDDL converter — direct conversion without ASP solver.
# For small state spaces (2-3 block stacking), ASP is overkill.
# For 4+ cubes, the full ASP pipeline can be added later.

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def dfa_graph_to_pddl(
    graph_json_path: str,
    output_dir: str = "generated/results/symbolic",
    domain_name: str = "stacking",
) -> dict[str, str]:
    """Convert a GraphLearner JSON to PDDL domain + problem files.

    Args:
        graph_json_path: Path to *_graph.json from GraphLearner.
        output_dir: Directory to write domain.pddl and problem.pddl.
        domain_name: Domain name for PDDL.

    Returns:
        Dict with "domain" and "problem" keys containing PDDL strings.
    """
    with open(graph_json_path) as f:
        data = json.load(f)

    states = data["states"]
    edges = data["edges"]

    # ── Extract predicate vocabulary ──────────────────────────────────
    all_predicates = set()
    for preds in states.values():
        for p in preds:
            # Parse "('on','cube1','cube2')" → ("on", 2)
            name, args = _parse_predicate(p)
            all_predicates.add((name, len(args)))

    # ── Extract object types ──────────────────────────────────────────
    all_objects = set()
    for preds in states.values():
        for p in preds:
            _, args = _parse_predicate(p)
            for arg in args:
                all_objects.add(arg)

    blocks = sorted([o for o in all_objects if o.startswith("cube")])
    tables = ["table"]

    # ── Extract actions ───────────────────────────────────────────────
    action_labels = sorted(set(e["label"] for e in edges))

    # Build domain PDDL
    domain = f"""; Auto-generated PDDL domain from GraphLearner
; {len(states)} states, {len(edges)} edges, {len(action_labels)} actions

(define (domain {domain_name})
  (:requirements :strips :typing)

  (:types
    block table - object
  )

  (:predicates
"""
    for pred_name, arity in sorted(all_predicates):
        params = " ".join(f"?x{i+1} - object" for i in range(arity))
        domain += f"    ({pred_name} {params})\n"
    domain += "  )\n\n"

    # Generate actions from state transitions
    for label in action_labels:
        label_edges = [e for e in edges if e["label"] == label]
        domain += f"  (:action {label}\n"
        domain += f"    :parameters (?x - block ?y - block)\n"

        # Preconditions from source states (intersection)
        src_preds_sets = []
        for e in label_edges:
            src = set(states[str(e["from"])])
            src_preds_sets.append(src)
        pre = set.intersection(*src_preds_sets) if src_preds_sets else set()

        domain += "    :precondition (and\n"
        for p in sorted(pre):
            name, args = _parse_predicate(p)
            domain += f"      ({name} {' '.join(args)})\n"
        domain += "    )\n"

        # Effects from state transitions
        add_effects = set()
        del_effects = set()
        for e in label_edges:
            src = set(states[str(e["from"])])
            tgt = set(states[str(e["to"])])
            add_effects.update(tgt - src)
            del_effects.update(src - tgt)

        domain += "    :effect (and\n"
        for p in sorted(add_effects):
            name, args = _parse_predicate(p)
            domain += f"      ({name} {' '.join(args)})\n"
        for p in sorted(del_effects):
            name, args = _parse_predicate(p)
            domain += f"      (not ({name} {' '.join(args)}))\n"
        domain += "    )\n"
        domain += "  )\n\n"
    domain += ")\n"

    # Build problem PDDL (initial state = s0, goal = any terminal state)
    init_state = states.get("0", [])
    terminal_states = [
        name for name in states
        if not any(e["from"] == int(name) for e in edges)
    ]

    problem = f"""; Auto-generated PDDL problem from GraphLearner

(define (problem {domain_name}_problem)
  (:domain {domain_name})

  (:objects
"""
    for b in blocks:
        problem += f"    {b} - block\n"
    for t in tables:
        problem += f"    {t} - object\n"
    problem += "  )\n\n"

    problem += "  (:init\n"
    for p in sorted(init_state):
        name, args = _parse_predicate(p)
        problem += f"    ({name} {' '.join(args)})\n"
    problem += "  )\n\n"

    if terminal_states:
        goal_state = states[terminal_states[0]]
        problem += "  (:goal (and\n"
        for p in sorted(goal_state):
            name, args = _parse_predicate(p)
            problem += f"    ({name} {' '.join(args)})\n"
        problem += "  ))\n"
    problem += ")\n"

    # Write to files
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "domain.pddl").write_text(domain)
    (out / "problem.pddl").write_text(problem)

    return {"domain": domain, "problem": problem}


def _parse_predicate(pred_str: str) -> tuple[str, tuple[str, ...]]:
    """Parse \"('on','cube1','cube2')\" → (\"on\", (\"cube1\",\"cube2\"))."""
    s = pred_str.strip()
    s = s.strip("()")
    parts = [p.strip().strip("'\"") for p in s.split(",")]
    parts = [p for p in parts if p]  # remove empty strings
    return parts[0], tuple(parts[1:]) if len(parts) > 1 else ()


def _pddl_pred(name: str, args: tuple[str, ...]) -> str:
    """Format a predicate for PDDL output."""
    if args:
        return f"({name} {' '.join(args)})"
    return f"({name})"
