"""Verify simulation_protocol.md contains canonical and deprecated paths."""

from pathlib import Path

DOC = Path("docs/simulation_protocol.md")


def test_doc_exists():
    assert DOC.exists()


def test_mentions_canonical_checkpoint():
    text = DOC.read_text()
    assert "stage_7B-4/dagger_reach_bc/best.pt" in text


def test_mentions_canonical_config():
    text = DOC.read_text()
    assert "robosuite_best_execution_config_stage7a12_final.json" in text


def test_mentions_canonical_eval_command():
    text = DOC.read_text()
    assert "eval_bc_hybrid" in text
    assert "--condition_on_phase" in text
    assert "--bc_phases" in text
    assert "--fixed_zero_action_dims" in text


def test_mentions_bc_phases():
    text = DOC.read_text()
    assert "reach_pick" in text
    assert "none" in text


def test_mentions_failure_types():
    text = DOC.read_text()
    for ft in ("horizon", "preplace", "verify_failed"):
        assert ft in text


def test_mentions_deprecated():
    text = DOC.read_text()
    assert "Deprecated" in text or "废弃" in text
    assert "eval_bc.py" in text
    assert "strict_success" in text or "strict_only" in text
