from __future__ import annotations

from vlm_nsm.core.data_types import Action, PlanResult as CommonPlanResult
from vlm_nsm.planners.internal_hanoi_planner import InternalHanoiPlanner
from vlm_nsm.planners.internal_stacking_planner import InternalStackingPlanner
from vlm_nsm.planners.metric_ff_adapter import MetricFFAdapter


class BFSStackingAdapter:
    """BFS symbolic planner adapter — converts to/from common Action format."""

    def __init__(self):
        from vlm_nsm.symbolic.symbolic_planner import SymbolicPlanner
        from vlm_nsm.symbolic.predicates import SymbolicSceneGraph
        from vlm_nsm.symbolic.goals import SymbolicGoal
        self._planner = SymbolicPlanner()
        self._SceneGraph = SymbolicSceneGraph
        self._Goal = SymbolicGoal

    def solve(self, cubes: list[str], target_stack: list[str]) -> CommonPlanResult:
        # Build initial state: all cubes on table, hand empty, all graspable + clear
        from vlm_nsm.symbolic.predicates import ObjectSymbol, ontable, clear, handempty, graspable
        initial = self._SceneGraph()
        for c in cubes:
            initial.objects[c] = ObjectSymbol(object_id=c, obj_type="cube", color="red", size=0.04)
            initial.add_predicate(ontable(c))
            initial.add_predicate(clear(c))
            initial.add_predicate(graspable(c))
        initial.add_predicate(handempty())

        # Build goal: target_stack is bottom-to-top
        from vlm_nsm.symbolic.goals import StackOrderGoal
        goal = StackOrderGoal(target_stack)

        # Run BFS
        bfs_result = self._planner.solve(initial, goal)
        if not bfs_result.success:
            return CommonPlanResult(
                False, [],
                bfs_result.reason,
                f"states_visited={bfs_result.states_visited}",
            )

        # Convert SymbolicAction → Action
        actions = []
        for sym_action in bfs_result.actions:
            name = sym_action.name  # "move_on" or "move_to_table"
            if name == "move_on":
                # MoveOn(x, y): pick x, place on y
                x, y = sym_action.args
                actions.append(Action("move", [x, "table", y]))
            elif name == "move_to_table":
                x = sym_action.args[0]
                actions.append(Action("move", [x, "table", "table"]))

        return CommonPlanResult(True, actions, "bfs_success",
                                f"states={bfs_result.states_visited}")


class PlannerAdapter:
    """Unified planner interface with auto-selection.

    Options:
      - "bfs": BFS symbolic planner (optimal for ≤1000 states)
      - "internal_stacking": rule-based (fast, stacking only)
      - "internal_hanoi": recursive (Hanoi only)
      - "metric_ff": PDDL + heuristic search (>1000 states)
    """

    def __init__(self, planner_name: str = "bfs", **kwargs):
        self.planner_name = planner_name
        if planner_name == "bfs":
            self.impl = BFSStackingAdapter()
        elif planner_name == "metric_ff":
            self.impl = MetricFFAdapter(**kwargs)
        elif planner_name == "internal_hanoi":
            ff_kwargs = {k: v for k, v in kwargs.items()
                         if k in ("source_peg", "aux_peg", "target_peg")}
            self.impl = InternalHanoiPlanner(**ff_kwargs)
        elif planner_name == "internal_stacking":
            self.impl = InternalStackingPlanner()
        elif planner_name == "fast_downward":
            self.impl = None
        else:
            raise ValueError(f"unknown planner: {planner_name}")

    def solve(self, domain_path=None, problem_path=None, **kwargs) -> CommonPlanResult:
        if self.planner_name == "fast_downward":
            raise NotImplementedError("Fast Downward reserved for future work")
        if self.planner_name == "internal_hanoi":
            return self.impl.solve(kwargs["n_blocks"])
        if self.planner_name in ("bfs", "internal_stacking"):
            return self.impl.solve(kwargs["cubes"], kwargs["target_stack"])
        return self.impl.solve(domain_path, problem_path)
