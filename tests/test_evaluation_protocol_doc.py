"""Verify evaluation_protocol.md contains canonical and deprecated paths."""

from pathlib import Path


DOC = Path("docs/evaluation_protocol.md")


def test_doc_exists():
    assert DOC.exists()


def test_mentions_canonical_eval():
    text = DOC.read_text()
    assert "eval_bc_hybrid" in text
    assert "run_episode" in text or "episode_runner" in text
    assert "shared runner" in text.lower()


def test_mentions_deprecated():
    text = DOC.read_text()
    assert "Deprecated" in text
    assert "eval_bc.py" in text
    assert "strict_success" in text or "strict_only" in text
    assert "MSE" in text or "offline" in text.lower()


def test_mentions_canonical_data():
    text = DOC.read_text()
    assert "strict_only" in text
    assert "stage_7A-14" in text
