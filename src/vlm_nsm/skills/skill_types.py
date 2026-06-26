"""Skill-level data types for hierarchical control.

These types are deliberately independent of Robosuite so that parser/planner
outputs, scripted controllers, and future learned skill policies can share one
stable command interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


SkillName = Literal["pick", "place", "place_on", "pick_place", "stack"]


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata) if metadata else {}


@dataclass
class SkillCommand:
    """A single high-level skill command."""

    skill_name: SkillName | str
    object_id: str
    target_object_id: str | None = None
    target_pose: list[float] | None = None
    relation: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 2.0: operator semantics from SymbolicAction
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    expected_add_effects: list[dict[str, Any]] = field(default_factory=list)
    expected_del_effects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "object_id": self.object_id,
            "target_object_id": self.target_object_id,
            "target_pose": list(self.target_pose) if self.target_pose is not None else None,
            "relation": self.relation,
            "source": self.source,
            "metadata": _metadata_copy(self.metadata),
            "preconditions": self.preconditions,
            "expected_add_effects": self.expected_add_effects,
            "expected_del_effects": self.expected_del_effects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCommand":
        return cls(
            skill_name=data["skill_name"],
            object_id=data["object_id"],
            target_object_id=data.get("target_object_id"),
            target_pose=list(data["target_pose"]) if data.get("target_pose") is not None else None,
            relation=data.get("relation"),
            source=data.get("source"),
            metadata=_metadata_copy(data.get("metadata")),
            preconditions=data.get("preconditions", []),
            expected_add_effects=data.get("expected_add_effects", []),
            expected_del_effects=data.get("expected_del_effects", []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "SkillCommand":
        return cls.from_dict(json.loads(payload))


def _serialize_source_plan(source_plan: list[Any] | None) -> list[Any] | None:
    if source_plan is None:
        return None

    serialized: list[Any] = []
    for item in source_plan:
        if hasattr(item, "name") and hasattr(item, "args"):
            serialized.append({"name": item.name, "args": list(item.args)})
        elif isinstance(item, dict):
            serialized.append(dict(item))
        elif isinstance(item, (list, tuple)):
            serialized.append(list(item))
        else:
            serialized.append(str(item))
    return serialized


@dataclass
class SkillPlan:
    """A sequence of skill commands derived from a symbolic plan."""

    commands: list[SkillCommand]
    source_plan: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": [command.to_dict() for command in self.commands],
            "source_plan": _serialize_source_plan(self.source_plan),
            "metadata": _metadata_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillPlan":
        return cls(
            commands=[SkillCommand.from_dict(item) for item in data.get("commands", [])],
            source_plan=data.get("source_plan"),
            metadata=_metadata_copy(data.get("metadata")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "SkillPlan":
        return cls.from_dict(json.loads(payload))


@dataclass
class SkillExecutionResult:
    """Result from executing one skill command."""

    success: bool
    skill_command: SkillCommand
    steps: int
    failure_type: str | None = None
    trajectory_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "skill_command": self.skill_command.to_dict(),
            "steps": self.steps,
            "failure_type": self.failure_type,
            "trajectory_path": self.trajectory_path,
            "metrics": _metadata_copy(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillExecutionResult":
        return cls(
            success=bool(data["success"]),
            skill_command=SkillCommand.from_dict(data["skill_command"]),
            steps=int(data["steps"]),
            failure_type=data.get("failure_type"),
            trajectory_path=data.get("trajectory_path"),
            metrics=_metadata_copy(data.get("metrics")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "SkillExecutionResult":
        return cls.from_dict(json.loads(payload))
