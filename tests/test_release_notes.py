"""Verify release notes exist and contain canonical info."""

from pathlib import Path

NOTES = Path("docs/release_notes_v0.1.md")


def test_release_notes_exists():
    assert NOTES.exists()


def test_release_notes_version():
    text = NOTES.read_text()
    assert "v0.1-rc1" in text


def test_release_notes_mentions_reach_pick():
    text = NOTES.read_text()
    assert "reach_pick" in text


def test_release_notes_mentions_92():
    text = NOTES.read_text()
    assert "92.0%" in text


def test_release_notes_mentions_93():
    text = NOTES.read_text()
    assert "93.3%" in text


def test_release_notes_mentions_5_of_5():
    text = NOTES.read_text()
    assert "5/5" in text


def test_release_notes_has_known_limitations():
    text = NOTES.read_text()
    assert "Known Limitations" in text or "known limitations" in text.lower()


def test_release_notes_has_reproduction_commands():
    text = NOTES.read_text()
    assert "run_canonical_demo.sh" in text
    assert "run_canonical_eval_20ep.sh" in text


def test_release_notes_mentions_semantic_constraints():
    text = NOTES.read_text()
    assert "on(top, bottom)" in text or "bottom-to-top" in text


def test_release_notes_mentions_language_batch_status():
    text = NOTES.read_text()
    assert "14/14" in text, "Missing language batch completion status"


def test_release_notes_mentions_984_samples():
    text = NOTES.read_text()
    assert "984" in text, "Missing total sample count"


def test_release_notes_mentions_failure_taxonomy():
    text = NOTES.read_text()
    assert "schema_invalid" in text, "Missing failure taxonomy"
    assert "invalid_object" in text, "Missing failure taxonomy"


def test_release_notes_mentions_556_failures():
    text = NOTES.read_text()
    assert "556" in text, "Missing total failure count"
