"""Verify artifact_manifest.md and key referenced files."""

from pathlib import Path


DOC = Path("docs/artifact_manifest.md")


def test_doc_exists():
    assert DOC.exists()


def test_lists_checkpoints():
    text = DOC.read_text()
    assert "stage_7B-4/dagger_reach_bc/best.pt" in text
    assert "stage_7B-3/bc_main/best.pt" in text


def test_lists_language_results():
    text = DOC.read_text()
    assert "language_variation_summary.json" in text
    assert "qwen_failure_cases.json" in text


def test_lists_final_tables():
    text = DOC.read_text()
    assert "report_tables/final" in text


def test_lists_documentation():
    text = DOC.read_text()
    assert "evaluation_protocol.md" in text
    assert "final_empirical_report.md" in text
    assert "project_summary.md" in text


def test_lists_tests():
    text = DOC.read_text()
    assert "test_final_report_tables" in text
    assert "test_evaluation_protocol_doc" in text
    assert "test_artifact_manifest" in text


def test_references_config():
    text = DOC.read_text()
    assert "robosuite_best_execution_config_stage7a12_final.json" in text
