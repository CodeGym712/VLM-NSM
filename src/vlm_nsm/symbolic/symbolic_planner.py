"""Symbolic planner using BFS over predicate states."""

from __future__ import annotations

from collections import deque
from typing import Any

from vlm_nsm.symbolic.predicates import SymbolicSceneGraph, ontable, clear, handempty
from vlm_nsm.symbolic.operators import SymbolicAction, MoveToTable, MoveOn
from vlm_nsm.symbolic.goals import SymbolicGoal


class PlanResult:
    def __init__(self, success: bool, actions: list[SymbolicAction] | None = None,
                 reason: str = "", states_visited: int = 0):
        self.success = success
        self.actions = actions or []
        self.reason = reason
        self.states_visited = states_visited

    def __len__(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "plan_length": len(self.actions),
            "actions": [str(a) for a in self.actions],
            "reason": self.reason,
            "states_visited": self.states_visited,
        }


class SymbolicPlanner:
    """BFS planner for stacking tasks using MoveOn + MoveToTable operators."""

    def __init__(self, max_states: int = 10000):
        self.max_states = max_states

    def solve(self, initial: SymbolicSceneGraph, goal: SymbolicGoal) -> PlanResult:
        # Goal already satisfied?
        if goal.predicates.issubset(initial.predicates):
            return PlanResult(True, [], "goal already satisfied", 0)

        obj_ids = list(initial.objects.keys())
        visited = set()
        queue = deque([(initial, [])])
        states_visited = 0

        while queue and states_visited < self.max_states:
            state, plan = queue.popleft()
            state_key = frozenset(p.to_tuple() for p in state.predicates)

            if state_key in visited:
                continue
            visited.add(state_key)
            states_visited += 1

            # Generate applicable actions
            for action in self._generate_actions(state, obj_ids):
                if not action.is_applicable(state):
                    continue
                new_state = action.apply(state)

                if goal.predicates.issubset(new_state.predicates):
                    return PlanResult(True, plan + [action], "goal reached", states_visited)

                new_key = frozenset(p.to_tuple() for p in new_state.predicates)
                if new_key not in visited:
                    queue.append((new_state, plan + [action]))

        return PlanResult(False, reason=f"no plan found ({states_visited} states visited)",
                         states_visited=states_visited)

    def _generate_actions(self, state: SymbolicSceneGraph, obj_ids: list[str]) -> list[SymbolicAction]:
        actions = []
        for x in obj_ids:
            if state.has("clear", x) and state.has("graspable", x) and state.has("handempty"):
                # MoveToTable
                if not state.has("ontable", x):
                    actions.append(MoveToTable(x))
                # MoveOn to other clear objects
                for y in obj_ids:
                    if x != y and state.has("clear", y):
                        actions.append(MoveOn(x, y))
        return actions
