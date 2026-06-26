"""Verify README / project summary mentions canonical deployment info."""

from pathlib import Path

PROJECT_SUMMARY = Path("docs/project_summary.md")
README = Path("README.md")


def test_project_summary_exists():
    assert PROJECT_SUMMARY.exists()


def test_project_summary_has_deployment_policy():
    text = PROJECT_SUMMARY.read_text()
    assert "reach_pick-only" in text, "Missing canonical policy name"


def test_project_summary_has_92():
    text = PROJECT_SUMMARY.read_text()
    assert "92.0%" in text, "Missing 92.0% result"


def test_project_summary_has_93():
    text = PROJECT_SUMMARY.read_text()
    assert "93.3%" in text, "Missing 93.3% result"


def test_project_summary_has_e2e_5_of_5():
    text = PROJECT_SUMMARY.read_text()
    assert "5/5" in text or "5 demos" in text.lower(), "Missing E2E demo result"


def test_project_summary_has_quickstart():
    text = PROJECT_SUMMARY.read_text()
    assert "run_canonical_demo.sh" in text, "Missing quickstart reference"


def test_project_summary_mentions_deprecated():
    text = PROJECT_SUMMARY.read_text()
    assert "Deprecated" in text or "deprecated" in text, "Missing deprecated paths section"


def test_readme_exists():
    assert README.exists(), "README.md not found"


def test_project_summary_mentions_14_of_14():
    text = PROJECT_SUMMARY.read_text()
    assert "14/14" in text, "Missing language batch status"


def test_project_summary_mentions_984_samples():
    text = PROJECT_SUMMARY.read_text()
    assert "984" in text, "Missing total sample count"


def test_project_summary_mentions_failure_modes():
    text = PROJECT_SUMMARY.read_text()
    assert "schema_invalid" in text, "Missing failure mode analysis"
    assert "invalid_object" in text, "Missing failure mode analysis"


def test_project_summary_mentions_556():
    text = PROJECT_SUMMARY.read_text()
    assert "556" in text, "Missing total failure count"
