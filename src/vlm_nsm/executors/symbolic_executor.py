from __future__ import annotations

from vlm_nsm.core.data_types import Action, ExecutionResult, Predicate, PredicateTuple
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.executors.base_executor import BaseExecutor


class SymbolicStateExecutor(BaseExecutor):
    def __init__(self):
        self.state: set[PredicateTuple] = set()

    def reset(self, init_state):
        self.state = {self._pred_tuple(p) for p in init_state}

    def execute(self, action: Action):
        name = action.name.lower()
        if name == "pick":
            self._execute_pick(action.args)
        elif name == "place":
            self._execute_place(action.args)
        elif name == "move":
            self._execute_move(action.args)
        else:
            raise ValueError(f"unsupported action: {action.name}")

    def execute_plan(self, actions: list[Action]) -> ExecutionResult:
        for idx, action in enumerate(actions):
            try:
                self.execute(action)
            except Exception as exc:
                return ExecutionResult(False, set(self.state), idx, str(exc))
        return ExecutionResult(True, set(self.state), None, None)

    def get_state(self):
        return set(self.state)

    def check_success(self, goal_predicates):
        return goal_satisfied(self.state, goal_predicates)

    def _execute_pick(self, args: list[str]) -> None:
        if len(args) != 2:
            raise ValueError("pick expects args: disk support")
        disk, support = args
        self._require(("on", disk, support))
        self._require(("clear", disk))
        self._require(("free-gripper",))
        self.state.discard(("on", disk, support))
        self.state.add(("clear", support))
        self.state.discard(("free-gripper",))
        self.state.add(("grasped", disk))

    def _execute_place(self, args: list[str]) -> None:
        if len(args) != 2:
            raise ValueError("place expects args: disk target")
        disk, target = args
        self._require(("grasped", disk))
        self._require(("clear", target))
        self._require(("smaller", disk, target))
        self.state.discard(("grasped", disk))
        self.state.add(("free-gripper",))
        self.state.add(("on", disk, target))
        self.state.add(("clear", disk))
        self.state.discard(("clear", target))

    def _execute_move(self, args: list[str]) -> None:
        if len(args) != 3:
            raise ValueError("move expects args: disk source target")
        disk, source, target = args
        self._require(("on", disk, source))
        self._require(("clear", disk))
        self._require(("clear", target))
        self._require(("smaller", disk, target))
        self._require(("free-gripper",))
        self.state.discard(("on", disk, source))
        self.state.discard(("clear", target))
        self.state.add(("clear", source))
        self.state.add(("on", disk, target))
        self.state.add(("clear", disk))

    def _require(self, pred: PredicateTuple) -> None:
        if pred not in self.state:
            raise ValueError(f"precondition failed: {pred}")

    @staticmethod
    def _pred_tuple(predicate) -> PredicateTuple:
        if isinstance(predicate, Predicate):
            return predicate.as_tuple()
        return tuple(predicate)
