import argparse
import json

from stage20_fake_dataset_utils import write_fake_stage20_dataset
from vlm_nsm.experiments.stage20_q_contact_holding_feasibility_audit import build_markdown_report, main, run_audit


def test_contact_holding_feasibility_audit_runs_on_fake_dataset(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=20)

    audit = run_audit(
        argparse.Namespace(
            dataset_dir=str(dataset),
            split="all",
            contact_distance_threshold=10.0,
            closed_gripper_threshold=10.0,
        )
    )

    assert audit["stage"] == "stage20_q_contact_holding_feasibility"
    assert audit["direct_label_available"] is False
    assert audit["proxy_only"] is True
    assert "holding_phase_separation" in audit["feasibility"]
    assert "recommended_for_training_now" in audit["feasibility"]


def test_contact_holding_feasibility_report_includes_no_hard_fabrication(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=20)
    audit = run_audit(
        argparse.Namespace(
            dataset_dir=str(dataset),
            split="all",
            contact_distance_threshold=10.0,
            closed_gripper_threshold=10.0,
        )
    )

    text = build_markdown_report(audit)

    assert "Stage 20-Q" in text
    assert "no_hard_fabrication" in text
    assert "Direct Label Availability" in text
    assert "recommended_for_training_now" in text


def test_contact_holding_feasibility_cli_writes_outputs(tmp_path, monkeypatch):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=20)
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--dataset_dir", str(dataset),
            "--split", "all",
            "--contact_distance_threshold", "10.0",
            "--closed_gripper_threshold", "10.0",
            "--output_json", str(output_json),
            "--output_md", str(output_md),
        ],
    )

    main()

    assert output_json.exists()
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["proxy_only"] is True
