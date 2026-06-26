"""Verify final Stage 8 report tables exist and contain expected data."""

from pathlib import Path


TABLES_DIR = Path("generated/report_tables/final")
DOCS_DIR = Path("docs")


def _check_table(name: str, *keywords: str) -> None:
    p = TABLES_DIR / name
    assert p.exists(), f"Missing table: {p}"
    text = p.read_text()
    for kw in keywords:
        assert kw in text, f"'{kw}' not found in {name}"


def test_language_model_comparison_exists():
    _check_table("final_language_model_comparison.md", "qwen25", "Goal Success")


def test_language_failure_taxonomy_exists():
    _check_table("final_language_failure_taxonomy.md", "schema_invalid", "invalid_object")


def test_language_prompt_comparison_exists():
    _check_table("final_language_prompt_comparison.md", "Prompt", "Goal Success")


def test_embodied_policy_comparison_exists():
    _check_table("final_embodied_policy_comparison.md", "DAgger", "bc_main", "Scripted")


def test_dagger_ablation_exists():
    _check_table("final_dagger_ablation.md", "Expert-only", "DAgger-only", "Expert + DAgger")


def test_phase_ablation_exists():
    _check_table("final_phase_ablation.md", "reach_pick", "reach_place")


def test_dataset_summary_exists():
    _check_table("final_dataset_summary.md", "Expert v2", "DAgger")


def test_artifact_manifest_exists():
    _check_table("final_artifact_manifest.md", "Checkpoints", "DAgger Reach BC")


def test_all_required_tables_exist():
    required = [
        "final_language_model_comparison.md",
        "final_language_failure_taxonomy.md",
        "final_language_prompt_comparison.md",
        "final_embodied_policy_comparison.md",
        "final_dagger_ablation.md",
        "final_phase_ablation.md",
        "final_dataset_summary.md",
        "final_artifact_manifest.md",
    ]
    for name in required:
        assert (TABLES_DIR / name).exists(), f"Missing required table: {name}"
