"""Symbolic operators: MoveToTable, MoveOn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_nsm.symbolic.predicates import (
    SymbolicSceneGraph, Predicate,
    on, ontable, clear, handempty,
)


@dataclass
class SymbolicAction:
    name: str
    args: list[str]
    preconditions: set[Predicate] = field(default_factory=set)
    effects_add: set[Predicate] = field(default_factory=set)
    effects_del: set[Predicate] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "preconditions": [str(p) for p in sorted(self.preconditions, key=str)],
            "effects_add": [str(p) for p in sorted(self.effects_add, key=str)],
            "effects_del": [str(p) for p in sorted(self.effects_del, key=str)],
        }

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def is_applicable(self, state: SymbolicSceneGraph) -> bool:
        return all(p in state.predicates for p in self.preconditions)

    def apply(self, state: SymbolicSceneGraph) -> SymbolicSceneGraph:
        new_state = state.clone()
        for p in self.effects_del:
            new_state.predicates.discard(p)
        for p in self.effects_add:
            new_state.predicates.add(p)
        return new_state


class MoveToTable(SymbolicAction):
    def __init__(self, x: str):
        pre = {Predicate("clear", (x,)), Predicate("graspable", (x,)), Predicate("handempty", ())}
        add = {Predicate("ontable", (x,)), Predicate("clear", (x,)), Predicate("handempty", ())}
        super().__init__("move_to_table", [x], preconditions=pre, effects_add=add, effects_del=set())

    def apply(self, state: SymbolicSceneGraph) -> SymbolicSceneGraph:
        new = state.clone()
        x = self.args[0]
        # Remove old on/ontable for x, clear previous support
        for p in list(new.predicates):
            if p.name == "on" and p.args[0] == x:
                support = p.args[1]
                new.predicates.discard(p)
                new.predicates.add(Predicate("clear", (support,)))
            elif p.name == "ontable" and p.args[0] == x:
                new.predicates.discard(p)
            elif p.name == "holding" and p.args[0] == x:
                new.predicates.discard(p)
        new.predicates.add(Predicate("ontable", (x,)))
        new.predicates.add(Predicate("clear", (x,)))
        new.predicates.add(Predicate("handempty", ()))
        return new


class MoveOn(SymbolicAction):
    def __init__(self, x: str, y: str):
        pre = {
            Predicate("clear", (x,)), Predicate("clear", (y,)),
            Predicate("graspable", (x,)), Predicate("handempty", ()),
        }
        add = {Predicate("on", (x, y)), Predicate("clear", (x,)), Predicate("handempty", ())}
        # Fix C7: effects_del must match what apply() actually deletes
        dele = {Predicate("clear", (y,))}
        super().__init__("move_on", [x, y], preconditions=pre, effects_add=add, effects_del=dele)

    def apply(self, state: SymbolicSceneGraph) -> SymbolicSceneGraph:
        new = state.clone()
        x, y = self.args[0], self.args[1]
        # Remove old on/ontable for x, clear previous support
        for p in list(new.predicates):
            if p.name == "on" and p.args[0] == x:
                support = p.args[1]
                new.predicates.discard(p)
                new.predicates.add(Predicate("clear", (support,)))
            elif p.name == "ontable" and p.args[0] == x:
                new.predicates.discard(p)
            elif p.name == "holding" and p.args[0] == x:
                new.predicates.discard(p)
        # y is no longer clear
        new.predicates.discard(Predicate("clear", (y,)))
        new.predicates.add(Predicate("on", (x, y)))
        new.predicates.add(Predicate("clear", (x,)))
        new.predicates.add(Predicate("handempty", ()))
        return new
