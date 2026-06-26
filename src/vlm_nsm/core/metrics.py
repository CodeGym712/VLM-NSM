from __future__ import annotations

from typing import Any


def hanoi_optimal_moves(n_blocks: int) -> int:
    return 2**n_blocks - 1


def summarize_run(n_blocks: int, plan_len: int, planner_success: bool, execution_success: bool, goal_success: bool) -> dict:
    optimal = hanoi_optimal_moves(n_blocks)
    logical = plan_len // 2 if plan_len else 0
    return {
        "n_blocks": n_blocks,
        "planner_success": planner_success,
        "atomic_action_length": plan_len,
        "logical_plan_length": logical,
        "optimal_logical_length": optimal,
        "plan_optimality_ratio": logical / optimal if optimal and logical else 0.0,
        "execution_success": execution_success,
        "goal_success": goal_success,
    }


def summarize_language_variation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    parsed = [s for s in samples if s.get("json_parse_success")]
    aligned = [s for s in samples if s.get("alignment_success")]

    return {
        "num_samples": total,
        "json_parse_rate": _rate(samples, "json_parse_success"),
        "alignment_success_rate": _rate(samples, "alignment_success"),
        "alignment_repair_rate": _rate(aligned, "repaired"),
        "predicate_validity_rate": _validity_rate(parsed, "invalid_predicate"),
        "object_validity_rate": _validity_rate(parsed, "invalid_object"),
        "type_check_pass_rate": _validity_rate(parsed, "type_error"),
        "goal_completeness_rate": _validity_rate(parsed, "incomplete_goal"),
        "planner_solvability_rate": _rate(samples, "planner_success"),
        "execution_success_rate": _rate(samples, "execution_success"),
        "goal_success_rate": _rate(samples, "goal_success"),
        "average_input_tokens": _average(samples, "input_tokens"),
        "average_output_tokens": _average(samples, "output_tokens"),
        "average_inference_time": _average(samples, "inference_time"),
    }


def _rate(samples: list[dict[str, Any]], key: str) -> float | None:
    if not samples:
        return None
    return sum(1 for sample in samples if bool(sample.get(key))) / len(samples)


def _validity_rate(samples: list[dict[str, Any]], failing_error_type: str) -> float | None:
    if not samples:
        return None
    return sum(1 for sample in samples if sample.get("error_type") != failing_error_type) / len(samples)


def _average(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [sample.get(key) for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def reality_gap(n_symbolic_success: int, n_physical_success: int) -> float | None:
    """Compute the Reality Gap between symbolic and physical execution.

    $$
    Reality Gap = (N_symbolic_success - N_physical_success) / N_symbolic_success * 100%
    $$

    Args:
        n_symbolic_success: Number of successes in symbolic execution.
        n_physical_success: Number of successes in physical (embodied) execution.

    Returns:
        Reality Gap percentage, or None if N_symbolic_success is 0.
    """
    if n_symbolic_success <= 0:
        return None
    return (n_symbolic_success - n_physical_success) / n_symbolic_success * 100.0
