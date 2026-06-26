"""Test dataset_card_v0_2.md for required sections and secret patterns.

No DashScope, Robosuite, or GPU required.
"""

import pytest
from pathlib import Path


CARD_PATH = Path("docs/dataset_card_v0_2.md")

REQUIRED_SECTIONS = [
    "Dataset Overview",
    "Dataset Components",
    "Task Scope",
    "Parser Modes",
    "Standard E2E Summary Schema",
    "Language-to-Plan Fields",
    "E2E Execution Fields",
    "Control Learning Metadata",
    "Known Limitations",
    "Reproducibility Commands",
    "Privacy / API Key Policy",
    "Not Included",
]

SECRET_PATTERNS = [
    "sk-",           # API key prefix (not "sk-your" which is a template)
    "QWEN_API_KEY=",  # explicit key assignment with value
]

REQUIRED_STATEMENTS = [
    "No API keys are included",
    "qwen3.6-plus",
    "2-block stacking",
    "no vision grounding",
    "on(bottom, table)",
    "physical success",
]


@pytest.mark.skipif(not CARD_PATH.exists(), reason="dataset card not found")
class TestDatasetCard:
    def test_card_exists(self):
        assert CARD_PATH.exists(), f"{CARD_PATH} does not exist"

    def test_all_required_sections_present(self):
        text = CARD_PATH.read_text()
        for section in REQUIRED_SECTIONS:
            # Section headings are numbered: "## 1. Dataset Overview"
            # Match either "## SectionName" or "## N. SectionName"
            assert section in text, f"Missing required section: {section}"

    def test_no_secret_patterns(self):
        """Should not contain API keys or secret assignments."""
        text = CARD_PATH.read_text()
        for pattern in SECRET_PATTERNS:
            assert pattern not in text, f"Found potential secret pattern: {pattern}"

    def test_required_statements_present(self):
        text = CARD_PATH.read_text()
        text_lower = text.lower()
        for statement in REQUIRED_STATEMENTS:
            if statement.startswith("no "):
                # Case-insensitive check for "no ..." statements
                assert statement in text_lower, \
                    f"Missing required statement (case-insensitive): {statement}"
            else:
                assert statement in text, \
                    f"Missing required statement: {statement}"

    def test_no_env_file_path_with_key(self):
        """Should not reference the actual env file path in a way that implies it contains keys."""
        text = CARD_PATH.read_text()
        # The file path is OK, but should not say "this file has the key"
        # Check for env mentions in privacy section
        privacy_section = text.split("## Privacy / API Key Policy")[1].split("##")[0] if "## Privacy / API Key Policy" in text else text
        assert "configs/dashscope_qwen.env.example" in privacy_section or "gitignored" in privacy_section.lower(), \
            "env file reference should include gitignore or template note"

    def test_section_ordering(self):
        """Sections should follow expected order."""
        text = CARD_PATH.read_text()
        positions = []
        for section in REQUIRED_SECTIONS:
            # Find either "## N. Section" or "## Section"
            pos = text.find(f"## {section}")
            if pos < 0:
                # Try with numbered prefix
                for line in text.split("\n"):
                    if f"## " in line and section in line and line.strip().endswith(section):
                        pos = text.find(line)
                        break
            assert pos >= 0, f"Section '{section}' not found as heading"
            positions.append(pos)
        # Check positions are in ascending order
        for i in range(1, len(positions)):
            assert positions[i] > positions[i-1], \
                f"Section order violation: '{REQUIRED_SECTIONS[i]}' before '{REQUIRED_SECTIONS[i-1]}'"

    def test_privacy_section_mentions_no_api_keys(self):
        text = CARD_PATH.read_text()
        assert "No API keys are included" in text
        # Should also mention the env example template
        assert "configs/dashscope_qwen.env.example" in text

    def test_limitations_cover_scope(self):
        """Limitations section should mention key constraints."""
        text = CARD_PATH.read_text()
        # Find limitations section by looking for the heading
        for line in text.split("\n"):
            if "##" in line and "Known Limitations" in line:
                limitations_heading = line
                break
        else:
            pytest.fail("Could not find Known Limitations section")
        limitations_section = text.split(limitations_heading)[1].split("\n##")[0]
        assert "2-block" in limitations_section
        assert "vision" in limitations_section.lower()
        assert "100%" in limitations_section
