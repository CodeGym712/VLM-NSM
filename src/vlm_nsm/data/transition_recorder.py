"""Recorder adapter from option execution results to EmbodiedTransition v1."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vlm_nsm.core.predicate_scene_graph import PredicateSceneGraph
from vlm_nsm.data.embodied_transition import (
    EmbodiedTransition,
    FailureAttribution,
    PredicateObservation,
)
from vlm_nsm.embodied.skills.base_skill import SkillResult


def _iso_timestamp(epoch_seconds: float | None = None) -> str:
    if epoch_seconds is None:
        value = datetime.now(timezone.utc)
    else:
        value = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _scene_observations(
    graph: PredicateSceneGraph,
    transition_id: str,
    phase: str,
) -> tuple[PredicateObservation, ...]:
    observations: list[PredicateObservation] = []
    for index, predicate in enumerate(sorted(graph.predicates, key=str)):
        observed_epoch = predicate.observed_at if predicate.observed_at is not None else graph.timestamp
        freshness = predicate.freshness_seconds
        if freshness is None:
            freshness = max(0.0, graph.timestamp - observed_epoch)
        observations.append(PredicateObservation(
            name=predicate.name,
            args=predicate.args,
            truth_value=True,
            source=predicate.source,
            confidence=predicate.confidence,
            observed_at=_iso_timestamp(observed_epoch),
            freshness_seconds=freshness,
            evidence_id=predicate.evidence_id or f"{transition_id}:{phase}:{index}",
        ))
    return tuple(observations)


def _normalize_expected_effect(
    effect: Mapping[str, Any],
    transition_id: str,
    timestamp: str,
) -> dict[str, list[PredicateObservation]]:
    normalized: dict[str, list[PredicateObservation]] = {"add": [], "delete": []}
    for kind in normalized:
        for index, predicate in enumerate(effect.get(kind, [])):
            if isinstance(predicate, PredicateObservation):
                observation = predicate
            else:
                observation = PredicateObservation(
                    name=predicate["name"],
                    args=tuple(predicate.get("args", ())),
                    truth_value=(kind == "add"),
                    source=predicate.get("source", "derived"),
                    confidence=float(predicate.get("confidence", 1.0)),
                    observed_at=predicate.get("observed_at", timestamp),
                    freshness_seconds=float(predicate.get("freshness_seconds", 0.0)),
                    evidence_id=predicate.get(
                        "evidence_id", f"{transition_id}:expected:{kind}:{index}"
                    ),
                )
            normalized[kind].append(observation)
    return normalized


def _observed_effect(
    pre: tuple[PredicateObservation, ...],
    post: tuple[PredicateObservation, ...],
) -> dict[str, list[PredicateObservation]]:
    pre_map = {(item.name, item.args): item for item in pre if item.truth_value}
    post_map = {(item.name, item.args): item for item in post if item.truth_value}
    added = [post_map[key] for key in sorted(post_map.keys() - pre_map.keys())]
    deleted = [replace(pre_map[key], truth_value=False) for key in sorted(pre_map.keys() - post_map.keys())]
    return {"add": added, "delete": deleted}


class EmbodiedTransitionRecorder:
    """Append complete, versioned option transitions to a JSONL dataset."""

    def __init__(
        self,
        output_path: str | Path,
        run_id: str,
        task_id: str,
        scene_id: str,
        seed: int,
        observer: Mapping[str, str],
        executor: Mapping[str, str],
    ) -> None:
        self.output_path = Path(output_path)
        self.run_id = run_id
        self.task_id = task_id
        self.scene_id = scene_id
        self.seed = seed
        self.observer = dict(observer)
        self.executor = dict(executor)

    def record_option(
        self,
        *,
        sequence_index: int,
        pre_scene: PredicateSceneGraph,
        post_scene: PredicateSceneGraph,
        grounded_option: Mapping[str, Any],
        expected_effect: Mapping[str, Any],
        result: SkillResult,
        attribution: FailureAttribution | None = None,
        attribution_confidence: float = 1.0,
        continuous_context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EmbodiedTransition:
        if result.success:
            resolved_attribution = FailureAttribution.SUCCESS
        elif attribution is None or attribution is FailureAttribution.SUCCESS:
            raise ValueError("failed option requires explicit attribution")
        else:
            resolved_attribution = attribution

        transition_id = f"{self.run_id}:{sequence_index:06d}"
        timestamp = _iso_timestamp()
        pre = _scene_observations(pre_scene, transition_id, "pre")
        post = _scene_observations(post_scene, transition_id, "post")
        expected = _normalize_expected_effect(expected_effect, transition_id, timestamp)
        observed = _observed_effect(pre, post)
        evidence_ids = tuple(dict.fromkeys(
            item.evidence_id
            for collection in (
                pre,
                post,
                tuple(expected["add"]),
                tuple(expected["delete"]),
                tuple(observed["add"]),
                tuple(observed["delete"]),
            )
            for item in collection
        ))
        record_metadata = {
            "steps": result.steps,
            "pre_scene_state_id": pre_scene.source_scene_state_id,
            "post_scene_state_id": post_scene.source_scene_state_id,
            **dict(metadata or {}),
        }
        transition = EmbodiedTransition(
            transition_id=transition_id,
            run_id=self.run_id,
            sequence_index=sequence_index,
            timestamp=timestamp,
            task_id=self.task_id,
            scene_id=self.scene_id,
            seed=self.seed,
            pre_predicates=pre,
            post_predicates=post,
            grounded_option=dict(grounded_option),
            expected_effect=expected,
            observed_effect=observed,
            observer=self.observer,
            executor=self.executor,
            continuous_context=dict(continuous_context or {}),
            controller_diagnostics=dict(result.diagnostics),
            success=result.success,
            raw_failure_type=result.failure_type,
            attribution=resolved_attribution,
            attribution_confidence=attribution_confidence,
            evidence_ids=evidence_ids,
            metadata=record_metadata,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(transition.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return transition
