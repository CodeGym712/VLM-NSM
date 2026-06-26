"""Symbolic predicates, object symbols, and scene graph."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

PREDICATE_NAMES = [
    "on", "ontable", "clear", "holding", "handempty",
    "reachable", "graspable", "stable", "perception_available",
]


@dataclass
class ObjectSymbol:
    object_id: str
    color: str = ""
    obj_type: str = "cube"
    size: float | None = None
    pose_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "color": self.color,
            "type": self.obj_type,
            "size": self.size,
            "pose_source": self.pose_source,
        }


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[str, ...]
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self):
        if self.name not in PREDICATE_NAMES:
            raise ValueError(f"Unknown predicate: {self.name}")

    def __hash__(self):
        return hash((self.name, self.args))

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return NotImplemented
        return self.name == other.name and self.args == other.args

    def to_tuple(self) -> tuple[str, ...]:
        return (self.name,) + self.args

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": list(self.args),
            "confidence": self.confidence,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


# Convenience constructors
def on(x: str, y: str, conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("on", (x, y), conf, src)


def ontable(x: str, conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("ontable", (x,), conf, src)


def clear(x: str, conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("clear", (x,), conf, src)


def holding(x: str, conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("holding", (x,), conf, src)


def handempty(conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("handempty", (), conf, src)


def reachable(x: str, conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("reachable", (x,), conf, src)


def graspable(x: str, conf: float = 1.0, src: str = "unknown") -> Predicate:
    return Predicate("graspable", (x,), conf, src)


@dataclass
class SymbolicSceneGraph:
    objects: dict[str, ObjectSymbol] = field(default_factory=dict)
    predicates: set[Predicate] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)
    source_scene_state_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def has(self, name: str, *args: str) -> bool:
        return Predicate(name, args) in self.predicates

    def get_objects_satisfying(self, name: str) -> list[str]:
        result = []
        for p in self.predicates:
            if p.name == name and len(p.args) >= 1:
                result.append(p.args[0])
        return sorted(set(result))

    def get_relation_targets(self, obj_id: str, relation: str) -> list[str]:
        targets = []
        for p in self.predicates:
            if p.name == relation and len(p.args) >= 2 and p.args[0] == obj_id:
                targets.append(p.args[1])
        return sorted(set(targets))

    def get_support(self, obj_id: str) -> str | None:
        for p in self.predicates:
            if p.name == "on" and p.args[0] == obj_id:
                return p.args[1]
        return None

    def get_stack_order(self) -> list[str]:
        """Return objects in bottom-to-top stack order."""
        bottoms = [o for o in self.objects if not self.get_support(o) or self.has("ontable", o)]
        if not bottoms:
            return []
        order = []
        for b in sorted(bottoms):
            current = b
            while current:
                order.append(current)
                tops = [p.args[0] for p in self.predicates if p.name == "on" and p.args[1] == current]
                current = tops[0] if tops else None
        return order

    def add_predicate(self, p: Predicate):
        self.predicates.add(p)

    def remove_predicate(self, p: Predicate):
        self.predicates.discard(p)

    def clone(self) -> "SymbolicSceneGraph":
        return SymbolicSceneGraph(
            objects={k: ObjectSymbol(**v.__dict__) for k, v in self.objects.items()},
            predicates=set(self.predicates),
            timestamp=time.time(),
            source_scene_state_id=self.source_scene_state_id,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "predicates": [p.to_dict() for p in sorted(self.predicates, key=str)],
            "timestamp": self.timestamp,
            "source_scene_state_id": self.source_scene_state_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        preds = ", ".join(str(p) for p in sorted(self.predicates, key=str))
        return f"SceneGraph(objects={list(self.objects.keys())}, predicates=[{preds}])"
