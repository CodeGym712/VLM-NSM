from __future__ import annotations

from vlm_nsm.core.data_types import Action, PlanResult


class InternalStackingPlanner:
    def solve(self, cubes: list[str], target_stack: list[str]) -> PlanResult:
        if not target_stack:
            return PlanResult(False, [], "", "target_stack cannot be empty")
        missing = [cube for cube in target_stack if cube not in cubes]
        if missing:
            return PlanResult(False, [], "", f"unknown cubes in target_stack: {missing}")

        actions = [
            Action("move", [obj, "table", target])
            for target, obj in zip(target_stack, target_stack[1:])
        ]
        raw = "\n".join(f"{idx}: move {' '.join(action.args)}" for idx, action in enumerate(actions))
        return PlanResult(True, actions, raw, None)
