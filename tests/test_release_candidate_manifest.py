"""Verify release candidate manifest exists and has canonical fields."""

from pathlib import Path

MANIFEST_PATH = Path("docs/release_candidate_manifest.md")


def test_manifest_exists():
    assert MANIFEST_PATH.exists(), f"{MANIFEST_PATH} not found"


def test_manifest_mentions_canonical_checkpoint():
    text = MANIFEST_PATH.read_text()
    assert "best.pt" in text, "Manifest missing canonical checkpoint reference"


def test_manifest_mentions_canonical_config():
    text = MANIFEST_PATH.read_text()
    assert "stage7a12_final.json" in text, "Manifest missing canonical config reference"


def test_manifest_mentions_reach_pick():
    text = MANIFEST_PATH.read_text()
    assert "reach_pick" in text, "Manifest missing reach_pick policy reference"


def test_manifest_mentions_known_limitations():
    text = MANIFEST_PATH.read_text()
    assert "Known Limitations" in text or "known limitations" in text.lower(), \
        "Manifest missing known limitations section"


def test_manifest_mentions_performance():
    text = MANIFEST_PATH.read_text()
    assert "92.0%" in text or "93.3%" in text, "Manifest missing performance numbers"
    assert "5/5" in text or "100%" in text, "Manifest missing E2E demo results"


def test_manifest_mentions_language_artifacts():
    text = MANIFEST_PATH.read_text()
    assert "language_variation_summary" in text, "Manifest missing language eval summary"
    assert "qwen_failure_cases" in text, "Manifest missing failure cases"
    assert "qwen_alignment_repair_stats" in text, "Manifest missing repair stats"


def test_manifest_mentions_report_tables():
    text = MANIFEST_PATH.read_text()
    assert "report_table_model_comparison" in text, "Manifest missing model comparison table"
    assert "report_table_prompt_comparison" in text, "Manifest missing prompt comparison table"
    assert "report_table_alignment_benefit" in text, "Manifest missing alignment benefit table"
    assert "report_table_by_category" in text, "Manifest missing by-category table"
