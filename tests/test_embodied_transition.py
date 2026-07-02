"""Unit tests for typed embodied-transition records."""

from dataclasses import replace

import pytest

from vlm_nsm.data.embodied_transition import (
    EmbodiedTransition,
    FailureAttribution,
    PredicateObservation,
)


def observation(evidence_id: str, name: str = "handempty") -> PredicateObservation:
    return PredicateObservation(
        name=name,
        args=(),
        truth_value=True,
        source="oracle",
        confidence=1.0,
        observed_at="2026-07-02T01:02:03Z",
        freshness_seconds=0.0,
        evidence_id=evidence_id,
    )


def transition(**changes) -> EmbodiedTransition:
    base = EmbodiedTransition(
        transition_id="transition-001",
        run_id="run-001",
        sequence_index=0,
        timestamp="2026-07-02T01:02:03Z",
        task_id="stack-2",
        scene_id="layout-001",
        seed=7,
        pre_predicates=(observation("pre-001"),),
        post_predicates=(observation("post-001"),),
        grounded_option={"name": "pick", "arguments": ["cube1"]},
        expected_effect={"add": [], "delete": []},
        observed_effect={"add": [], "delete": []},
        observer={"name": "mujoco_oracle", "version": "v1"},
        executor={"name": "atomic_pid", "version": "executor_v1"},
        continuous_context={},
        controller_diagnostics={},
        success=True,
        raw_failure_type=None,
        attribution=FailureAttribution.SUCCESS,
        attribution_confidence=1.0,
        evidence_ids=("pre-001", "post-001"),
        metadata={},
    )
    return replace(base, **changes)


def test_round_trip_preserves_typed_fields():
    record = transition()

    restored = EmbodiedTransition.from_dict(record.to_dict())

    assert restored == record
    assert restored.attribution is FailureAttribution.SUCCESS
    assert restored.pre_predicates[0].args == ()


def test_content_hash_is_stable_across_dictionary_order():
    first = transition(continuous_context={"z": 1, "a": 2})
    second = transition(continuous_context={"a": 2, "z": 1})

    assert first.content_hash() == second.content_hash()


@pytest.mark.parametrize(
    ("attribution", "policy"),
    [
        (FailureAttribution.SUCCESS, "positive"),
        (FailureAttribution.SYMBOLIC_INVALID, "precondition_negative"),
        (FailureAttribution.MODEL_MISMATCH, "counterexample"),
        (FailureAttribution.OBSERVATION_UNCERTAIN, "defer"),
        (FailureAttribution.LANGUAGE, "exclude"),
        (FailureAttribution.CAPABILITY_OOD, "exclude"),
        (FailureAttribution.CONTROLLER_FAILURE, "exclude"),
        (FailureAttribution.EVIDENCE_INVALID, "exclude"),
    ],
)
def test_symbolic_evidence_policy_is_cause_specific(attribution, policy):
    assert attribution.symbolic_evidence_policy == policy


def test_predicate_observation_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        PredicateObservation(
            name="clear",
            args=("cube1",),
            truth_value=True,
            source="geometry",
            confidence=1.5,
            observed_at="2026-07-02T01:02:03Z",
            freshness_seconds=0.0,
            evidence_id="bad-confidence",
        )


def test_success_transition_requires_success_attribution():
    with pytest.raises(ValueError, match="successful transition"):
        transition(attribution=FailureAttribution.CONTROLLER_FAILURE)
