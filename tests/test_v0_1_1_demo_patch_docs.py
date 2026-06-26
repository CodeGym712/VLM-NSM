"""Verify v0.1.1 demo patch docs exist and reference frozen canonical state."""

from pathlib import Path

NOTES = Path("docs/release_notes_v0.1.1.md")


def test_v0_1_1_release_notes_exist():
    assert NOTES.exists()


def test_v0_1_1_mentions_v0_1_1_version():
    text = NOTES.read_text()
    assert "v0.1.1" in text


def test_v0_1_1_states_canonical_checkpoint_unchanged():
    text = NOTES.read_text()
    assert "best.pt" in text
    assert "Frozen" in text or "unchanged" in text.lower()


def test_v0_1_1_states_config_unchanged():
    text = NOTES.read_text()
    assert "stage7a12_final.json" in text


def test_v0_1_1_states_reach_pick_unchanged():
    text = NOTES.read_text()
    assert "reach_pick" in text


def test_v0_1_1_mentions_frozen_status():
    text = NOTES.read_text()
    assert "Frozen" in text or "frozen" in text


def test_v0_1_1_has_validation_status():
    text = NOTES.read_text()
    assert "Validation Status" in text or "validation" in text.lower()


def test_v0_1_1_mentions_cached_qwen():
    text = NOTES.read_text()
    assert "cached_qwen" in text or "Cached Qwen" in text
