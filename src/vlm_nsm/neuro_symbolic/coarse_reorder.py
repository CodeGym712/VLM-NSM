"""Coarse Reorder — when CoarseOptionVerifier rejects a program, attempt BFS repair.

Handles the 3-block top-down ordering case:
  Input:  pick(cube1), place(cube1,cube2), pick(cube2), place(cube2,cube3)
  Verifier rejects: cube2 not clear (cube1 on top)
  Repair:
    1. Extract final goal predicates from the rejected program.
    2. Call BFS PlannerAdapter("bfs").solve(cubes, target_stack) to get bottom-up plan.
    3. Convert BFS actions → coarse pick/place steps.
    4. Re-verify through CoarseOptionVerifier.
    5. Return repaired coarse steps (or None if unrepairable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReorderResult:
    """Result of BFS reorder repair attempt."""
    success: bool = False
    original_steps: list = field(default_factory=list)
    repaired_steps: list = field(default_factory=list)
    rejection_reason: str = ""
    bfs_plan: list = field(default_factory=list)
    bfs_states_visited: int = 0
    verification_passed: bool = False
    error: str = ""


def extract_target_stack(coarse_steps: list) -> tuple[list[str], list[list[str]]]:
    """Extract target stack (bottom→top) and goal predicates from coarse steps.

    Uses parent chain traversal: for on(A, B), B is parent of A.
    Bottom object has no parent (sits on table). Build chain bottom→top.

    Args:
        coarse_steps: List of coarse step dicts.

    Returns:
        (target_stack, goal_predicates)
        target_stack: bottom-to-top order (e.g. ["cube3", "cube2", "cube1"])
        goal_predicates: [["on", "cube1", "cube2"], ...]
    """
    goal_preds = []
    parent_of: dict[str, str] = {}  # child → what it sits on
    all_objs: set[str] = set()

    for step in coarse_steps:
        if isinstance(step, dict):
            op = step.get("op", "")
            obj = step.get("object", "")
            tgt = step.get("target", "")
            pred = step.get("predicate", [])
        elif hasattr(step, 'op'):
            op = step.op
            obj = step.object
            tgt = getattr(step, 'target', '')
            pred = getattr(step, 'predicate', [])
        else:
            continue

        if op == "place" and obj and tgt and tgt != "table":
            parent_of[str(obj)] = str(tgt)
            all_objs.add(str(obj))
            all_objs.add(str(tgt))
            goal_preds.append(["on", str(obj), str(tgt)])
        elif op == "achieve" and pred:
            pred_name = str(pred[0]).lower() if pred else ""
            if pred_name == "on" and len(pred) >= 3:
                top, bottom = str(pred[1]), str(pred[2])
                if bottom != "table":
                    parent_of[top] = bottom
                    all_objs.add(top)
                    all_objs.add(bottom)
                    goal_preds.append(["on", top, bottom])

    if not parent_of:
        return [], goal_preds

    # Find bottom: object that is never a child (no parent pointing to it)
    children = set(parent_of.keys())
    bottom = None
    for obj in all_objs:
        if obj not in children:
            bottom = obj
            break

    if bottom is None:
        # Cycle or all are children — pick arbitrary
        bottom = next(iter(all_objs - children)) if (all_objs - children) else next(iter(all_objs))

    # Build chain bottom→top
    stack = [bottom]
    current = bottom
    while len(stack) <= len(all_objs) + 1:  # safety limit
        # Find child that sits on current
        found = False
        for child, parent in parent_of.items():
            if parent == current and child not in stack:
                stack.append(child)
                current = child
                found = True
                break
        if not found:
            break

    return stack, goal_preds


def bfs_reorder_coarse(coarse_steps: list, objects: list[str],
                       rejection_reasons: list[str] | None = None) -> ReorderResult:
    """Attempt BFS reorder of a rejected coarse program.

    Only triggers if the rejection is due to ordering (not forbidden ops,
    unknown objects, self-place, etc.).

    Args:
        coarse_steps: The rejected coarse step list.
        objects: Valid object IDs.
        rejection_reasons: Specific rejection reasons from verifier.

    Returns:
        ReorderResult with repaired steps or error.
    """
    result = ReorderResult(
        original_steps=list(coarse_steps),
        rejection_reason="; ".join(rejection_reasons) if rejection_reasons else "unknown",
    )

    # Check if rejection is repairable (ordering issue, not structural)
    reasons_str = " ".join(rejection_reasons or [])
    unrepairable = {"forbidden", "unknown", "self-place", "empty", "phase-level"}
    if any(kw in reasons_str.lower() for kw in unrepairable):
        result.error = f"unrepairable rejection: {reasons_str[:200]}"
        return result

    # Extract target stack from original program
    target_stack, goal_preds = extract_target_stack(coarse_steps)
    if not target_stack and not goal_preds:
        result.error = "could not extract target_stack or goal_predicates"
        return result

    # If we have goal_preds but no clear stack, try to build from predicates
    cubes = [o for o in objects if o != "table"]
    if not target_stack and goal_preds:
        # Use the on predicates to build stack
        for gp in goal_preds:
            if gp[0] == "on" and len(gp) >= 3:
                top, bottom = gp[1], gp[2]
                if bottom != "table":
                    target_stack = [bottom, top]
                    break
        if not target_stack:
            # Default: all cubes in order
            target_stack = list(cubes)

    # Call BFS planner
    try:
        from vlm_nsm.core.planner_adapter import PlannerAdapter
        planner = PlannerAdapter("bfs")
        plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    except Exception as e:
        result.error = f"bfs_planner_error: {e}"
        return result

    if not plan_result.success:
        result.error = f"bfs_no_plan: {plan_result.error}"
        result.bfs_states_visited = getattr(plan_result, 'states_visited', 0)
        return result

    # Convert BFS actions → coarse pick/place steps
    repaired_steps = []
    for action in plan_result.actions:
        if hasattr(action, 'name') and hasattr(action, 'args'):
            name = action.name
            args = action.args
        elif hasattr(action, 'name'):
            name = action.name
            args = getattr(action, 'args', [])
        else:
            continue

        if name == "move" and len(args) >= 3:
            obj, src, tgt = args[0], args[1], args[2]
            if tgt != "table" and tgt != src:
                repaired_steps.append({"op": "pick", "object": str(obj)})
                repaired_steps.append({"op": "place", "object": str(obj), "target": str(tgt)})
            elif tgt == "table":
                repaired_steps.append({"op": "pick", "object": str(obj)})
                repaired_steps.append({"op": "place", "object": str(obj), "target": "table"})
        elif name in ("move_on", "MoveOn") and len(args) >= 2:
            obj, tgt = str(args[0]), str(args[1])
            repaired_steps.append({"op": "pick", "object": obj})
            repaired_steps.append({"op": "place", "object": obj, "target": tgt})
        elif name in ("move_to_table", "MoveToTable") and len(args) >= 1:
            obj = str(args[0])
            repaired_steps.append({"op": "pick", "object": obj})
            repaired_steps.append({"op": "place", "object": obj, "target": "table"})

    if not repaired_steps:
        result.error = "bfs_plan_conversion_empty"
        return result

    result.bfs_plan = [str(a) if hasattr(a, '__str__') else repr(a) for a in plan_result.actions]
    result.bfs_states_visited = getattr(plan_result, 'states_visited', 0)

    # Re-verify repaired steps
    from vlm_nsm.neuro_symbolic.verifier_bank import CoarseOptionVerifier
    verifier = CoarseOptionVerifier(valid_objects=objects)
    report = verifier.check(coarse_steps=repaired_steps)
    result.verification_passed = report.passed

    if report.passed:
        result.success = True
        result.repaired_steps = repaired_steps
    else:
        result.error = f"repaired_program_verification_failed: {report.explain()}"

    return result


def reorder_or_fail(coarse_steps: list, objects: list[str],
                    rejection_reasons: list[str]) -> tuple[list | None, ReorderResult]:
    """Main entry point: try BFS reorder, return repaired steps or None.

    Returns:
        (repaired_steps_or_None, ReorderResult)
    """
    result = bfs_reorder_coarse(coarse_steps, objects, rejection_reasons)
    if result.success:
        return result.repaired_steps, result
    return None, result
