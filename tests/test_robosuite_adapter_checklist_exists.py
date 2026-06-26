"""Test that robosuite adapter checklist document exists and has required sections."""

from pathlib import Path


def test_checklist_doc_exists():
    doc = Path(__file__).parent.parent / "docs" / "robosuite_adapter_checklist.md"
    assert doc.exists(), "robosuite_adapter_checklist.md not found"


def test_checklist_has_required_sections():
    doc = Path(__file__).parent.parent / "docs" / "robosuite_adapter_checklist.md"
    content = doc.read_text()

    required_sections = [
        "RobosuiteEnvWrapper API",
        "Object Pose Extraction",
        "Cube Name Mapping",
        "OSC_POSE Action Format",
        "ActionGrounder Pose Convention",
        "TerminationFailureMonitor Thresholds",
        "PhysicalPredicateExtractor Thresholds",
        "DemonstrationRecorder Dimensions",
        "Replay Feasibility",
        "Reality Gap Reporting Protocol",
        "Implementation Order",
    ]

    for section in required_sections:
        assert section in content, f"Missing required section: {section}"


def test_checklist_has_status_indicators():
    doc = Path(__file__).parent.parent / "docs" / "robosuite_adapter_checklist.md"
    content = doc.read_text()
    # Should have at least one "Status:" indicator
    assert "**Status:**" in content
