from __future__ import annotations

from vlm_nsm.core.data_types import Action, PlanResult


class InternalHanoiPlanner:
    def __init__(self, source_peg: str = "peg1", aux_peg: str = "peg2", target_peg: str = "peg3"):
        self.source_peg = source_peg
        self.aux_peg = aux_peg
        self.target_peg = target_peg

    def solve(self, n_blocks: int) -> PlanResult:
        if n_blocks <= 0:
            return PlanResult(False, [], "", "n_blocks must be positive")
        moves: list[tuple[str, str]] = []
        self._move_tower(n_blocks, self.source_peg, self.target_peg, self.aux_peg, moves)

        stacks = {
            self.source_peg: [f"cube{i}" for i in range(n_blocks, 0, -1)],
            self.aux_peg: [],
            self.target_peg: [],
        }
        actions: list[Action] = []
        for source, target in moves:
            disk = stacks[source].pop()
            support = stacks[source][-1] if stacks[source] else source
            place_target = stacks[target][-1] if stacks[target] else target
            actions.append(Action("pick", [disk, support]))
            actions.append(Action("place", [disk, place_target]))
            stacks[target].append(disk)
        return PlanResult(True, actions, self._format_raw(actions), None)

    def _move_tower(self, n: int, source: str, target: str, aux: str, moves: list[tuple[str, str]]) -> None:
        if n == 0:
            return
        self._move_tower(n - 1, source, aux, target, moves)
        moves.append((source, target))
        self._move_tower(n - 1, aux, target, source, moves)

    @staticmethod
    def _format_raw(actions: list[Action]) -> str:
        return "\n".join(f"{idx}: {action.name} {' '.join(action.args)}" for idx, action in enumerate(actions))
