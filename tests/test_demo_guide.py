"""Verify demo guide exists and references canonical scripts."""

from pathlib import Path

GUIDE = Path("docs/demo_guide.md")


def test_demo_guide_exists():
    assert GUIDE.exists()


def test_demo_guide_mentions_canonical_demo_script():
    text = GUIDE.read_text()
    assert "run_canonical_demo.sh" in text


def test_demo_guide_mentions_canonical_eval_script():
    text = GUIDE.read_text()
    assert "run_canonical_eval_20ep.sh" in text


def test_demo_guide_mentions_reach_pick():
    text = GUIDE.read_text()
    assert "reach_pick" in text


def test_demo_guide_mentions_expected_5_of_5():
    text = GUIDE.read_text()
    assert "5/5" in text


def test_demo_guide_mentions_92():
    text = GUIDE.read_text()
    assert "92.0%" in text


def test_demo_guide_has_troubleshooting():
    text = GUIDE.read_text()
    assert "Troubleshooting" in text or "troubleshooting" in text.lower()


def test_demo_guide_mentions_cached_qwen():
    text = GUIDE.read_text()
    assert "cached_qwen" in text or "cached Qwen" in text
