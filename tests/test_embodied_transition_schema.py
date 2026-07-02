"""Contract tests for EmbodiedTransition v1 JSON Schema."""

import json
from pathlib import Path

import jsonschema
import pytest


SCHEMA_PATH = Path("schemas/embodied_transition_v1.schema.json")


def valid_record() -> dict:
    predicate = {
        "name": "handempty",
        "args": [],
        "truth_value": True,
        "source": "oracle",
        "confidence": 1.0,
        "observed_at": "2026-07-02T01:02:03Z",
        "freshness_seconds": 0.0,
        "evidence_id": "obs-pre-001",
    }
    return {
        "schema_version": "embodied_transition/v1",
        "transition_id": "transition-001",
        "run_id": "run-001",
        "sequence_index": 0,
        "timestamp": "2026-07-02T01:02:03Z",
        "task_id": "stack-2",
        "scene_id": "layout-001",
        "seed": 7,
        "pre_predicates": [predicate],
        "post_predicates": [predicate | {"evidence_id": "obs-post-001"}],
        "grounded_option": {"name": "pick", "arguments": ["cube1"]},
        "expected_effect": {"add": [], "delete": []},
        "observed_effect": {"add": [], "delete": []},
        "observer": {"name": "mujoco_oracle", "version": "v1"},
        "executor": {"name": "atomic_pid", "version": "executor_v1"},
        "continuous_context": {},
        "controller_diagnostics": {},
        "success": True,
        "raw_failure_type": None,
        "attribution": "SUCCESS",
        "attribution_confidence": 1.0,
        "evidence_ids": ["obs-pre-001", "obs-post-001"],
        "metadata": {},
    }


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_valid_record_matches_schema():
    jsonschema.Draft202012Validator(load_schema()).validate(valid_record())


@pytest.mark.parametrize(
    "field",
    ["transition_id", "run_id", "pre_predicates", "post_predicates", "executor", "attribution"],
)
def test_rejects_missing_research_critical_field(field):
    record = valid_record()
    del record[field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(load_schema()).validate(record)


def test_rejects_unversioned_executor():
    record = valid_record()
    del record["executor"]["version"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(load_schema()).validate(record)


def test_rejects_predicate_without_evidence_id():
    record = valid_record()
    del record["pre_predicates"][0]["evidence_id"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(load_schema()).validate(record)
