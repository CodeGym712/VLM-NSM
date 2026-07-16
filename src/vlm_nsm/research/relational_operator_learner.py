"""Train-only, provenance-aware relational Operator induction.

The learner is intentionally small: it is a research baseline/seed, not a
planner or a perception system.  It consumes event evidence only and keeps
operator families anonymous until post-hoc evaluation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

Atom = tuple[str, ...]

TYPE_SCHEMA: dict[tuple[str, int], tuple[str, ...]] = {
    ("holding", 1): ("object",),
    ("on", 2): ("object", "support"),
    ("in", 2): ("object", "container"),
    ("open", 1): ("openable",),
    ("closed", 1): ("openable",),
    ("activated", 1): ("device",),
    ("deactivated", 1): ("device",),
    ("at_region", 2): ("object", "region"),
    ("handempty", 0): (),
}


def _atoms(row: dict[str, Any], fields: Iterable[str]) -> list[Atom]:
    return [tuple(str(value) for value in atom) for field in fields for atom in row.get(field, [])]


def effect_signature(row: dict[str, Any]) -> tuple[tuple[str, int, str], ...]:
    parts = [(atom[0], len(atom) - 1, "add") for atom in _atoms(row, ["added"])]
    parts += [(atom[0], len(atom) - 1, "del") for atom in _atoms(row, ["deleted"])]
    return tuple(sorted(parts))


def _role_order(row: dict[str, Any]) -> list[str]:
    # Effect arguments define the operator interface.  Only before-state
    # arguments that are already connected to an effect role are retained;
    # unrelated scene objects must never become operator variables.
    ordered: list[str] = []
    for atom in _atoms(row, ["added", "deleted"]):
        for value in atom[1:]:
            if value not in ordered:
                ordered.append(value)
    for atom in _atoms(row, ["before"]):
        if all(value in ordered for value in atom[1:]):
            continue
        # A before-only argument is not safely bindable from an operator
        # effect, so it is deliberately excluded from the learned clause.
    return ordered


def role_values(row: dict[str, Any]) -> dict[str, str]:
    """Return canonical role -> ground constant mapping for one event."""
    return {f"?r{index}": value for index, value in enumerate(_role_order(row))}


def family_binding(row: dict[str, Any], family: dict[str, Any]) -> dict[str, str]:
    """Bind learned roles to an event using effect atoms, not scene order.

    Scene ``before`` atoms contain distractor entities whose presence can vary
    across episodes.  Binding roles by the order of all scene atoms therefore
    silently swaps constants.  Operator arguments are instead anchored to the
    learned add/delete effects; this is the stable relational interface used
    by the applicability evaluator.
    """
    binding: dict[str, str] = {}
    for family_field, row_field in (("add_effects", "added"), ("delete_effects", "deleted")):
        family_atoms = [tuple(str(value) for value in atom) for atom in family.get(family_field, [])]
        row_atoms = [tuple(str(value) for value in atom) for atom in row.get(row_field, [])]
        family_atoms.sort()
        row_atoms.sort()
        used: set[int] = set()
        for f_atom in family_atoms:
            for index, r_atom in enumerate(row_atoms):
                if index in used or f_atom[0] != r_atom[0] or len(f_atom) != len(r_atom):
                    continue
                used.add(index)
                for f_value, r_value in zip(f_atom[1:], r_atom[1:]):
                    if f_value.startswith("?"):
                        binding.setdefault(f_value, r_value)
                break
    return binding


def precondition_roles(family: dict[str, Any]) -> set[str]:
    """Return only the learned roles needed to test applicability.

    An operator effect may contain a destination or other role that is not
    mentioned in any precondition.  Applicability is a predicate over the
    current state, so requiring such an effect-only role to be resolved turns
    a definite failed precondition into a spurious ``unresolved`` outcome.
    """
    return {
        value
        for atom in family.get("preconditions", [])
        for value in atom[1:]
        if isinstance(value, str) and value.startswith("?")
    }


def applicability_binding(row: dict[str, Any], family: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Bind only precondition roles and report any that remain unresolved."""
    event_binding = family_binding(row, family)
    roles = precondition_roles(family)
    return (
        {role: event_binding[role] for role in sorted(roles) if role in event_binding},
        [role for role in sorted(roles) if role not in event_binding],
    )


def canonicalize_event(row: dict[str, Any]) -> dict[str, Any]:
    constants = _role_order(row)
    roles = {value: f"?r{index}" for index, value in enumerate(constants)}

    def convert(atom: Atom) -> list[str]:
        return [atom[0], *(roles[value] for value in atom[1:])]

    def unique(fields: Iterable[str]) -> list[list[str]]:
        values = {
            json.dumps(convert(atom), sort_keys=True)
            for atom in _atoms(row, fields)
            if all(value in roles for value in atom[1:])
        }
        return [json.loads(value) for value in sorted(values)]

    typed_roles: dict[str, str] = {}
    for atom in _atoms(row, ["added", "deleted", "before"]):
        signature = TYPE_SCHEMA.get((atom[0], len(atom) - 1), ())
        for value, type_name in zip(atom[1:], signature):
            if value in roles:
                typed_roles.setdefault(roles[value], type_name)
    return {
        "roles": [{"name": roles[value], "type": typed_roles.get(roles[value], "entity")} for value in constants],
        "before": unique(["before"]),
        "added": unique(["added"]),
        "deleted": unique(["deleted"]),
    }


def _select(counter: Counter[str], total: int, threshold: float) -> list[list[str]]:
    return [json.loads(atom) for atom, count in sorted(counter.items()) if count / total >= threshold]


def learn_operator_families(rows: list[dict[str, Any]], min_support: float = 0.8) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[str, int, str], ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[effect_signature(row)].append(row)

    families: list[dict[str, Any]] = []
    for index, (signature, members) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        canonical = [canonicalize_event(row) for row in members]
        before = Counter(json.dumps(atom, sort_keys=True) for item in canonical for atom in item["before"])
        additions = Counter(json.dumps(atom, sort_keys=True) for item in canonical for atom in item["added"])
        deletions = Counter(json.dumps(atom, sort_keys=True) for item in canonical for atom in item["deleted"])
        preconditions = _select(before, len(members), min_support)
        family = {
            "family": f"anonymous_family_{index:03d}",
            "effect_signature": [list(part) for part in signature],
            "roles": canonical[0]["roles"] if canonical else [],
            "preconditions": preconditions,
            "add_effects": _select(additions, len(members), 1.0),
            "delete_effects": _select(deletions, len(members), 1.0),
            "support": len(members),
            "negative_witnesses": [
                {"literal": json.loads(atom), "present": count, "total": len(members), "support": count / len(members)}
                for atom, count in sorted(before.items())
                if count < len(members)
            ],
            "provenance": {
                "training_only": True,
                "source_events": [f"{row.get('file', '')}::{row.get('demo', '')}::{row.get('t_start', '')}" for row in members],
                "ignored_fields": ["operator_type", "task", "goal", "p4_domain", "human_event"],
                "method": "effect_signature+typed_role+support_threshold",
                "min_support": min_support,
            },
        }
        families.append(family)
    return families


def evaluate_events(rows: list[dict[str, Any]], families: list[dict[str, Any]]) -> dict[str, float]:
    by_signature = {tuple(tuple(part) for part in family["effect_signature"]): family for family in families}
    covered = effect_match = applicable = 0
    for row in rows:
        family = by_signature.get(effect_signature(row))
        if family is None:
            continue
        covered += 1
        observed = canonicalize_event(row)
        binding = family_binding(row, family)
        required_ground = {tuple(binding.get(value, value) for value in atom) for atom in family["preconditions"]}
        seen = {tuple(str(value) for value in atom) for atom in row.get("before", [])}
        applicable += int(required_ground.issubset(seen))
        expected = set(json.dumps(atom, sort_keys=True) for atom in family["add_effects"] + family["delete_effects"])
        actual = set(json.dumps(atom, sort_keys=True) for atom in observed["added"] + observed["deleted"])
        effect_match += int(expected == actual)
    total = len(rows)
    return {
        "coverage": covered / total if total else 0.0,
        "applicability_recall_positive": applicable / covered if covered else 0.0,
        "effect_exact_match": effect_match / covered if covered else 0.0,
    }
