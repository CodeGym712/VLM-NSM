"""Verify demo media manifest exists and demo guide references v0.1.1."""

from pathlib import Path

MANIFEST = Path("assets/demo/v0_1_1/demo_media_manifest.md")
DEMO_GUIDE = Path("docs/demo_guide.md")
ASSETS_README = Path("assets/demo/README.md")


def test_demo_media_manifest_exists():
    assert MANIFEST.exists()


def test_demo_media_manifest_has_status_table():
    text = MANIFEST.read_text()
    assert "Status" in text or "status" in text.lower()
    assert "Pending" in text or "pending" in text.lower()


def test_demo_media_manifest_has_source_results():
    text = MANIFEST.read_text()
    assert "mock" in text.lower()
    assert "cached_qwen" in text.lower() or "qwen" in text.lower()


def test_demo_guide_mentions_v0_1_1():
    text = DEMO_GUIDE.read_text()
    assert "v0.1.1" in text


def test_demo_guide_mentions_v0_1_1_media_path():
    text = DEMO_GUIDE.read_text()
    assert "v0_1_1" in text


def test_assets_readme_mentions_v0_1_1():
    text = ASSETS_README.read_text()
    assert "v0.1.1" in text


def test_assets_readme_mentions_demo_media_manifest():
    text = ASSETS_README.read_text()
    assert "demo_media_manifest" in text
