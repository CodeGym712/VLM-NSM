"""Tests for recording option executions as EmbodiedTransition v1 JSONL."""

import json

import jsonschema
import pytest

from vlm_nsm.core.predicate_scene_graph import PredicateSceneGraph
from vlm_nsm.data.embodied_transition import FailureAttribution
from vlm_nsm.data.transition_dataset import load_schema
from vlm_nsm.data.transition_recorder import EmbodiedTransitionRecorder
from vlm_nsm.embodied.skills.base_skill import SkillResult
from vlm_nsm.symbolic.predicates import Predicate


def scene(state_id: str, predicate_name: str) -> PredicateSceneGraph:
    graph = PredicateSceneGraph()
    graph.source_scene_state_id = state_id
    graph.timestamp = 1_782_955_323.0
    args = () if predicate_name == "handempty" else ("cube1",)
    graph.add_predicate(Predicate(
        predicate_name,
        args,
        confidence=1.0,
        source="oracle",
        observed_at=1_782_955_323.0,
        freshness_seconds=0.0,
        evidence_id=f"{state_id}-{predicate_name}",
    ))
    return graph


def recorder(tmp_path) -> EmbodiedTransitionRecorder:
    return EmbodiedTransitionRecorder(
        output_path=tmp_path / "transitions.jsonl",
        run_id="run-001",
        task_id="stack-2",
        scene_id="layout-001",
        seed=7,
        observer={"name": "mujoco_oracle", "version": "v1"},
        executor={"name": "atomic_pid", "version": "executor_v1"},
    )


def test_records_schema_valid_option_transition(tmp_path):
    target = recorder(tmp_path)

    transition = target.record_option(
        sequence_index=0,
        pre_scene=scene("pre", "handempty"),
        post_scene=scene("post", "holding"),
        grounded_option={"name": "pick", "arguments": ["cube1"]},
        expected_effect={
            "add": [{"name": "holding", "args": ["cube1"]}],
            "delete": [{"name": "handempty", "args": []}],
        },
        result=SkillResult(True, steps=42, diagnostics={"contact_detected": True}),
    )

    row = json.loads((tmp_path / "transitions.jsonl").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(load_schema()).validate(row)
    assert transition.transition_id == "run-001:000000"
    assert row["attribution"] == "SUCCESS"
    assert row["observed_effect"]["add"][0]["name"] == "holding"
    assert row["controller_diagnostics"]["contact_detected"] is True


def test_failed_option_requires_explicit_causal_attribution(tmp_path):
    target = recorder(tmp_path)

    with pytest.raises(ValueError, match="explicit attribution"):
        target.record_option(
            sequence_index=0,
            pre_scene=scene("pre", "handempty"),
            post_scene=scene("post", "handempty"),
            grounded_option={"name": "pick", "arguments": ["cube1"]},
            expected_effect={"add": [], "delete": []},
            result=SkillResult(False, steps=10, failure_type="nocontact"),
        )


def test_failed_option_preserves_raw_failure_and_attribution(tmp_path):
    target = recorder(tmp_path)

    transition = target.record_option(
        sequence_index=1,
        pre_scene=scene("pre", "handempty"),
        post_scene=scene("post", "handempty"),
        grounded_option={"name": "pick", "arguments": ["cube1"]},
        expected_effect={"add": [], "delete": []},
        result=SkillResult(False, steps=10, failure_type="nocontact"),
        attribution=FailureAttribution.CONTROLLER_FAILURE,
        attribution_confidence=0.9,
    )

    assert transition.raw_failure_type == "nocontact"
    assert transition.attribution is FailureAttribution.CONTROLLER_FAILURE
    assert transition.symbolic_evidence_policy == "exclude"
