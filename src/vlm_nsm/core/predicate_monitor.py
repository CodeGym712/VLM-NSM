from __future__ import annotations

from vlm_nsm.core.data_types import Predicate, PredicateTuple


def _to_tuple(predicate: Predicate | PredicateTuple | list[str]) -> PredicateTuple:
    if isinstance(predicate, Predicate):
        return predicate.as_tuple()
    return tuple(predicate)


def goal_satisfied(current_state: set[PredicateTuple], goal_predicates: list[Predicate | PredicateTuple | list[str]], mode: str = "symbolic") -> bool:
    if mode != "symbolic":
        raise NotImplementedError(f"predicate monitor mode is not implemented: {mode}")
    return all(_to_tuple(pred) in current_state for pred in goal_predicates)


def compare_symbolic_physical(symbolic_state, physical_state):
    symbolic = {_to_tuple(pred) for pred in symbolic_state}
    physical = {_to_tuple(pred) for pred in physical_state}
    missing = sorted(symbolic - physical)
    extra = sorted(physical - symbolic)
    denom = max(len(symbolic | physical), 1)
    return {
        "mismatch_rate": (len(missing) + len(extra)) / denom,
        "missing_in_physical": missing,
        "extra_in_physical": extra,
    }
