from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vlm_nsm.experiments.eval_referential_semantics import (
    build_audit_lookup,
    classify_result,
    compute_semantic_metrics,
    load_audit,
    load_eval_results,
)


# ---- Fixtures ----

@pytest.fixture
def sample_audit_entries():
    return [
        {
            "instruction_id": "ref_001",
            "instruction_text": "Place the nearer cube onto the farther cube.",
            "semantic_validity": "objectively_groundable",
            "reference_type": "robot_relative_near_far",
            "recommended_expected_status": "grounded",
        },
        {
            "instruction_id": "ref_002",
            "instruction_text": "Put the cube that is on the table onto the other cube.",
            "semantic_validity": "objectively_ambiguous",
            "reference_type": "on_table",
            "recommended_expected_status": "ungroundable",
        },
        {
            "instruction_id": "ref_003",
            "instruction_text": "Take the first cube you see and put it on the second one.",
            "semantic_validity": "unsupported_reference_frame",
            "reference_type": "visual_order",
            "recommended_expected_status": "ungroundable",
        },
        {
            "instruction_id": "ref_004",
            "instruction_text": "Place the bright cube on the dull cube.",
            "semantic_validity": "unsupported_reference_frame",
            "reference_type": "color_attribute",
            "recommended_expected_status": "ungroundable",
        },
    ]


@pytest.fixture
def sample_eval_results(sample_audit_entries):
    """Build eval results from audit entries plus some non-referential ones."""
    results = []
    for entry in sample_audit_entries:
        results.append({
            "instruction": entry["instruction_text"],
            "predicted_bindings": None,
            "expected_bindings": None,
            "status": None,
            "failure_type": None,
        })
    # Add non-referential instructions
    results.append({
        "instruction": "Put the red cube on the green cube.",
        "predicted_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": None,
        "failure_type": None,
    })
    results.append({
        "instruction": "Stack the green cube on the red cube.",
        "predicted_bindings": {"top block": "cube2", "bottom block": "cube1"},
        "expected_bindings": {"top block": "cube2", "bottom block": "cube1"},
        "status": None,
        "failure_type": None,
    })
    return {"total": len(results), "results": results}


@pytest.fixture
def audit_file(sample_audit_entries, tmp_path):
    p = tmp_path / "audit.json"
    p.write_text(json.dumps({"referential_instructions": sample_audit_entries}))
    return p


@pytest.fixture
def eval_file(sample_eval_results, tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(sample_eval_results))
    return p


# ---- Load tests ----

def test_load_audit(audit_file):
    entries = load_audit(audit_file)
    assert len(entries) == 4
    assert entries[0]["instruction_id"] == "ref_001"


def test_load_audit_missing_key(tmp_path):
    p = tmp_path / "bad_audit.json"
    p.write_text(json.dumps({"other": []}))
    with pytest.raises(ValueError, match="referential_instructions"):
        load_audit(p)


def test_load_audit_not_found():
    with pytest.raises(FileNotFoundError):
        load_audit("/nonexistent/path.json")


def test_load_eval_results(eval_file):
    data = load_eval_results(eval_file)
    assert data["total"] == 6
    assert len(data["results"]) == 6


def test_load_eval_results_missing_key(tmp_path):
    p = tmp_path / "bad_eval.json"
    p.write_text(json.dumps({"foo": []}))
    with pytest.raises(ValueError, match="results"):
        load_eval_results(p)


# ---- Lookup tests ----

def test_build_audit_lookup(sample_audit_entries):
    lookup = build_audit_lookup(sample_audit_entries)
    assert len(lookup) == 4
    assert lookup["Place the nearer cube onto the farther cube."]["instruction_id"] == "ref_001"


# ---- Classification tests ----

def test_classify_groundable_correct(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Place the nearer cube onto the farther cube.",
        "predicted_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": None,
        "failure_type": None,
    }
    c = classify_result(result, audit["Place the nearer cube onto the farther cube."])
    assert c["semantic_validity"] == "objectively_groundable"
    assert c["binding_correct"] is True
    assert c["correct_outcome"] is True
    assert c["unsafe_guess"] is False
    assert c["over_rejection"] is False


def test_classify_groundable_wrong_binding(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Place the nearer cube onto the farther cube.",
        "predicted_bindings": {"top block": "cube2", "bottom block": "cube1"},
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": None,
        "failure_type": None,
    }
    c = classify_result(result, audit["Place the nearer cube onto the farther cube."])
    assert c["binding_correct"] is False
    assert c["correct_outcome"] is False
    assert c["unsafe_guess"] is False
    assert c["over_rejection"] is False


def test_classify_groundable_over_rejection(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Place the nearer cube onto the farther cube.",
        "predicted_bindings": None,
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": "ungroundable",
        "failure_type": "ungroundable_object",
    }
    c = classify_result(result, audit["Place the nearer cube onto the farther cube."])
    assert c["system_is_ungroundable"] is True
    assert c["binding_correct"] is False
    assert c["correct_outcome"] is False
    assert c["over_rejection"] is True
    assert c["unsafe_guess"] is False


def test_classify_ambiguous_correct_rejection(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Put the cube that is on the table onto the other cube.",
        "predicted_bindings": None,
        "expected_bindings": None,
        "status": "ungroundable",
        "failure_type": "ungroundable_object",
    }
    c = classify_result(result, audit["Put the cube that is on the table onto the other cube."])
    assert c["semantic_validity"] == "objectively_ambiguous"
    assert c["system_is_ungroundable"] is True
    assert c["correct_outcome"] is True
    assert c["unsafe_guess"] is False
    assert c["over_rejection"] is False


def test_classify_ambiguous_unsafe_guess(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Put the cube that is on the table onto the other cube.",
        "predicted_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": None,
        "failure_type": None,
    }
    c = classify_result(result, audit["Put the cube that is on the table onto the other cube."])
    assert c["unsafe_guess"] is True
    assert c["correct_outcome"] is False


def test_classify_unsupported_correct_rejection(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Place the bright cube on the dull cube.",
        "predicted_bindings": None,
        "expected_bindings": None,
        "status": "ungroundable",
        "failure_type": "ungroundable_object",
    }
    c = classify_result(result, audit["Place the bright cube on the dull cube."])
    assert c["semantic_validity"] == "unsupported_reference_frame"
    assert c["correct_outcome"] is True


def test_classify_unsupported_unsafe_guess(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Take the first cube you see and put it on the second one.",
        "predicted_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": None,
        "failure_type": None,
    }
    c = classify_result(result, audit["Take the first cube you see and put it on the second one."])
    assert c["unsafe_guess"] is True
    assert c["correct_outcome"] is False


def test_classify_non_referential(sample_audit_entries):
    audit = build_audit_lookup(sample_audit_entries)
    result = {
        "instruction": "Put the red cube on the green cube.",
        "predicted_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "expected_bindings": {"top block": "cube1", "bottom block": "cube2"},
        "status": None,
        "failure_type": None,
    }
    c = classify_result(result, audit.get("Put the red cube on the green cube."))
    assert c["has_audit_entry"] is False
    assert c["semantic_validity"] == "non_referential"
    assert c["binding_correct"] is True


# ---- Metrics tests ----

def make_classification(instr_id, validity, recommended, system_ungroundable, binding_correct, ref_type="test"):
    return {
        "instruction": f"test {instr_id}",
        "has_audit_entry": True,
        "instruction_id": instr_id,
        "semantic_validity": validity,
        "reference_type": ref_type,
        "recommended_expected_status": recommended,
        "system_is_ungroundable": system_ungroundable,
        "binding_correct": binding_correct,
        "correct_outcome": (
            (system_ungroundable and recommended == "ungroundable")
            or (not system_ungroundable and recommended == "grounded" and binding_correct)
        ),
        "unsafe_guess": not system_ungroundable and recommended == "ungroundable",
        "over_rejection": system_ungroundable and recommended == "grounded",
        "evaluator_mismatch": False,
    }


def test_compute_metrics_all_perfect():
    classifications = [
        make_classification("g1", "objectively_groundable", "grounded", False, True),
        make_classification("g2", "objectively_groundable", "grounded", False, True),
        make_classification("a1", "objectively_ambiguous", "ungroundable", True, None),
        make_classification("u1", "unsupported_reference_frame", "ungroundable", True, None),
    ]
    m = compute_semantic_metrics(classifications)
    assert m["groundable_referential_accuracy"] == 1.0
    assert m["ambiguous_rejection_accuracy"] == 1.0
    assert m["unsupported_reference_rejection_accuracy"] == 1.0
    assert m["referential_policy_score"] == 1.0
    assert m["unsafe_guess_count"] == 0
    assert m["over_rejection_count"] == 0


def test_compute_metrics_all_failures():
    classifications = [
        make_classification("g1", "objectively_groundable", "grounded", False, False),  # wrong binding
        make_classification("g2", "objectively_groundable", "grounded", True, None),    # over-rejection
        make_classification("a1", "objectively_ambiguous", "ungroundable", False, True),  # unsafe guess (binding "correct" per old metric but wrong policy)
        make_classification("u1", "unsupported_reference_frame", "ungroundable", False, False),  # unsafe guess
    ]
    m = compute_semantic_metrics(classifications)
    assert m["groundable_referential_accuracy"] == 0.0
    assert m["ambiguous_rejection_accuracy"] == 0.0
    assert m["unsupported_reference_rejection_accuracy"] == 0.0
    assert m["referential_policy_score"] == 0.0
    assert m["unsafe_guess_count"] == 2
    assert m["over_rejection_count"] == 1


def test_compute_metrics_mixed():
    classifications = [
        make_classification("g1", "objectively_groundable", "grounded", False, True),   # correct
        make_classification("g2", "objectively_groundable", "grounded", False, False),  # wrong
        make_classification("a1", "objectively_ambiguous", "ungroundable", True, None),   # correct rejection
        make_classification("a2", "objectively_ambiguous", "ungroundable", False, True),  # unsafe guess
        make_classification("u1", "unsupported_reference_frame", "ungroundable", True, None),  # correct rejection
    ]
    m = compute_semantic_metrics(classifications)
    assert m["groundable_referential_accuracy"] == 0.5  # 1/2
    assert m["ambiguous_rejection_accuracy"] == 0.5     # 1/2
    assert m["unsupported_reference_rejection_accuracy"] == 1.0  # 1/1
    assert m["referential_policy_score"] == 0.6          # 3/5
    assert m["unsafe_guess_count"] == 1
    assert m["over_rejection_count"] == 0


def test_compute_metrics_empty():
    m = compute_semantic_metrics([])
    assert m["referential_total"] == 0
    assert m["groundable_referential_accuracy"] == 0.0
    assert m["referential_policy_score"] == 0.0


def test_compute_metrics_type_breakdown():
    classifications = [
        make_classification("g1", "objectively_groundable", "grounded", False, True, "robot_relative_near_far"),
        make_classification("a1", "objectively_ambiguous", "ungroundable", True, None, "on_table"),
        make_classification("u1", "unsupported_reference_frame", "ungroundable", True, None, "visual_order"),
    ]
    m = compute_semantic_metrics(classifications)
    assert "robot_relative_near_far" in m["reference_type_breakdown"]
    assert "on_table" in m["reference_type_breakdown"]
    assert "visual_order" in m["reference_type_breakdown"]
    bd = m["reference_type_breakdown"]["robot_relative_near_far"]
    assert bd["total"] == 1
    assert bd["groundable_correct"] == 1


def test_compute_metrics_non_referential_excluded():
    classifications = [
        make_classification("g1", "objectively_groundable", "grounded", False, True),
        {
            "instruction": "color grounding",
            "has_audit_entry": False,
            "semantic_validity": "non_referential",
            "reference_type": None,
            "recommended_expected_status": None,
            "system_is_ungroundable": False,
            "binding_correct": False,
            "correct_outcome": False,
            "unsafe_guess": False,
            "over_rejection": False,
            "evaluator_mismatch": False,
        },
    ]
    m = compute_semantic_metrics(classifications)
    assert m["referential_total"] == 1
    assert m["non_referential_total"] == 1
    # Non-referential should not influence referential metrics
    assert m["groundable_referential_accuracy"] == 1.0


# ---- Integration test ----

def test_full_pipeline(audit_file, eval_file):
    from vlm_nsm.experiments.eval_referential_semantics import (
        build_audit_lookup,
        classify_result,
        compute_semantic_metrics,
        load_audit,
        load_eval_results,
    )

    eval_data = load_eval_results(eval_file)
    audit_entries = load_audit(audit_file)
    lookup = build_audit_lookup(audit_entries)

    # Set up specific behaviors for each referential instruction
    # Modify the eval results to simulate specific outcomes
    for r in eval_data["results"]:
        audit = lookup.get(r["instruction"])
        if audit is None:
            continue
        iid = audit["instruction_id"]
        if iid == "ref_001":
            # groundable: correct binding
            r["predicted_bindings"] = {"top block": "cube1", "bottom block": "cube2"}
            r["expected_bindings"] = {"top block": "cube1", "bottom block": "cube2"}
        elif iid == "ref_002":
            # ambiguous: correctly reject
            r["status"] = "ungroundable"
            r["failure_type"] = "ungroundable_object"
            r["predicted_bindings"] = None
            r["expected_bindings"] = None
        elif iid == "ref_003":
            # unsupported: unsafe guess
            r["predicted_bindings"] = {"top block": "cube1", "bottom block": "cube2"}
            r["expected_bindings"] = {"top block": "cube1", "bottom block": "cube2"}
        elif iid == "ref_004":
            # unsupported: correctly reject
            r["status"] = "ungroundable"
            r["failure_type"] = "ungroundable_object"
            r["predicted_bindings"] = None
            r["expected_bindings"] = None

    classifications = [classify_result(r, lookup.get(r["instruction"])) for r in eval_data["results"]]
    metrics = compute_semantic_metrics(classifications)

    assert metrics["referential_total"] == 4
    assert metrics["groundable_referential_accuracy"] == 1.0  # ref_001 correct
    assert metrics["ambiguous_rejection_accuracy"] == 1.0     # ref_002 correctly rejected
    assert metrics["unsupported_reference_rejection_accuracy"] == 0.5  # ref_003 guessed, ref_004 rejected
    assert metrics["unsafe_guess_count"] == 1  # ref_003
    assert metrics["over_rejection_count"] == 0
    assert metrics["referential_policy_score"] == 0.75  # 3/4 correct outcomes
