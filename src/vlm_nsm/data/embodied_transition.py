"""Typed data contract for failure-aware symbolic operator learning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "embodied_transition/v1"
PREDICATE_SOURCES = {"oracle", "geometry", "detector", "vlm", "derived", "unknown"}


class FailureAttribution(str, Enum):
    """Causal attribution used to decide whether a trace may revise a domain."""

    SUCCESS = "SUCCESS"
    LANGUAGE = "LANGUAGE"
    SYMBOLIC_INVALID = "SYMBOLIC_INVALID"
    OBSERVATION_UNCERTAIN = "OBSERVATION_UNCERTAIN"
    CAPABILITY_OOD = "CAPABILITY_OOD"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    EVIDENCE_INVALID = "EVIDENCE_INVALID"

    @property
    def symbolic_evidence_policy(self) -> str:
        return {
            FailureAttribution.SUCCESS: "positive",
            FailureAttribution.SYMBOLIC_INVALID: "precondition_negative",
            FailureAttribution.MODEL_MISMATCH: "counterexample",
            FailureAttribution.OBSERVATION_UNCERTAIN: "defer",
            FailureAttribution.LANGUAGE: "exclude",
            FailureAttribution.CAPABILITY_OOD: "exclude",
            FailureAttribution.CONTROLLER_FAILURE: "exclude",
            FailureAttribution.EVIDENCE_INVALID: "exclude",
        }[self]


@dataclass(frozen=True)
class PredicateObservation:
    """One grounded predicate together with observation provenance."""

    name: str
    args: tuple[str, ...]
    truth_value: bool
    source: str
    confidence: float
    observed_at: str
    freshness_seconds: float
    evidence_id: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("predicate name must not be empty")
        if self.source not in PREDICATE_SOURCES:
            raise ValueError(f"unknown predicate source: {self.source}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.freshness_seconds < 0.0:
            raise ValueError("freshness_seconds must be non-negative")
        if not self.evidence_id:
            raise ValueError("evidence_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["args"] = list(self.args)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredicateObservation":
        values = dict(data)
        values["args"] = tuple(values.get("args", ()))
        return cls(**values)


def _effect_to_dict(effect: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"add": [], "delete": []}
    for kind in result:
        for predicate in effect.get(kind, []):
            if isinstance(predicate, PredicateObservation):
                result[kind].append(predicate.to_dict())
            else:
                result[kind].append(dict(predicate))
    return result


@dataclass(frozen=True)
class EmbodiedTransition:
    """A versioned transition whose evidence can be audited and replayed."""

    transition_id: str
    run_id: str
    sequence_index: int
    timestamp: str
    task_id: str
    scene_id: str
    seed: int
    pre_predicates: tuple[PredicateObservation, ...]
    post_predicates: tuple[PredicateObservation, ...]
    grounded_option: Mapping[str, Any]
    expected_effect: Mapping[str, Any]
    observed_effect: Mapping[str, Any]
    observer: Mapping[str, str]
    executor: Mapping[str, str]
    continuous_context: Mapping[str, Any]
    controller_diagnostics: Mapping[str, Any]
    success: bool
    raw_failure_type: str | None
    attribution: FailureAttribution
    attribution_confidence: float
    evidence_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.transition_id or not self.run_id:
            raise ValueError("transition_id and run_id must not be empty")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")
        if not 0.0 <= self.attribution_confidence <= 1.0:
            raise ValueError("attribution_confidence must be in [0, 1]")
        if self.success and self.attribution is not FailureAttribution.SUCCESS:
            raise ValueError("successful transition requires SUCCESS attribution")
        if not self.success and self.attribution is FailureAttribution.SUCCESS:
            raise ValueError("failed transition cannot use SUCCESS attribution")
        for component_name, component in (("observer", self.observer), ("executor", self.executor)):
            if not component.get("name") or not component.get("version"):
                raise ValueError(f"{component_name} requires name and version")

    @property
    def symbolic_evidence_policy(self) -> str:
        return self.attribution.symbolic_evidence_policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "run_id": self.run_id,
            "sequence_index": self.sequence_index,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "pre_predicates": [p.to_dict() for p in self.pre_predicates],
            "post_predicates": [p.to_dict() for p in self.post_predicates],
            "grounded_option": dict(self.grounded_option),
            "expected_effect": _effect_to_dict(self.expected_effect),
            "observed_effect": _effect_to_dict(self.observed_effect),
            "observer": dict(self.observer),
            "executor": dict(self.executor),
            "continuous_context": dict(self.continuous_context),
            "controller_diagnostics": dict(self.controller_diagnostics),
            "success": self.success,
            "raw_failure_type": self.raw_failure_type,
            "attribution": self.attribution.value,
            "attribution_confidence": self.attribution_confidence,
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbodiedTransition":
        values = dict(data)
        values["pre_predicates"] = tuple(
            PredicateObservation.from_dict(p) for p in values.get("pre_predicates", [])
        )
        values["post_predicates"] = tuple(
            PredicateObservation.from_dict(p) for p in values.get("post_predicates", [])
        )
        values["attribution"] = FailureAttribution(values["attribution"])
        values["evidence_ids"] = tuple(values.get("evidence_ids", ()))
        return cls(**values)

    def content_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
