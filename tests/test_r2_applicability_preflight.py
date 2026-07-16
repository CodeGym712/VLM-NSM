import json
from pathlib import Path

from scripts.audit_r2_applicability_preflight import main


def test_preflight_keeps_provisional_status(tmp_path, monkeypatch):
    domain = tmp_path / "domain.json"
    positive = tmp_path / "positive.jsonl"
    negative = tmp_path / "negative.jsonl"
    output = tmp_path / "out"
    domain.write_text(
        json.dumps(
            [
                {
                    "family": "anonymous_family_000",
                    "effect_signature": [["holding", 1, "add"]],
                    "preconditions": [["handempty"]],
                }
            ]
        ),
        encoding="utf-8",
    )
    positive.write_text(
        json.dumps(
            {
                "file": "demo.hdf5",
                "demo": "demo_0",
                "t_start": 1,
                "before": [["handempty"]],
                "added": [["holding", "cube"]],
                "deleted": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    negative.write_text(
        json.dumps(
            {
                "family": "anonymous_family_000",
                "source_event": "demo.hdf5::demo_0::1",
                "file": "demo.hdf5",
                "demo": "demo_0",
                "frame_t": 2,
                "state": [],
                "missing_preconditions": [["handempty"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_r2_applicability_preflight.py",
            "--domain",
            str(domain),
            "--positive-events",
            str(positive),
            "--negative-candidates",
            str(negative),
            "--output-dir",
            str(output),
        ],
    )
    main()
    report = json.loads((output / "r2_applicability_preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "PROVISIONAL_NO_F1"
    assert report["resolved_candidate_count"] == 1
    assert report["families"][0]["positive_precondition_recall"] == 1.0
    detail = json.loads((output / "r2_applicability_candidate_details.jsonl").read_text(encoding="utf-8"))
    assert detail["event_binding"] == {}
    assert detail["precondition_binding"] == {}
    assert (output / "unresolved_cases.jsonl").read_text(encoding="utf-8") == ""


def test_preflight_reports_missing_precondition_binding_but_not_effect_only_role(tmp_path, monkeypatch):
    domain = tmp_path / "domain.json"
    positive = tmp_path / "positive.jsonl"
    negative = tmp_path / "negative.jsonl"
    output = tmp_path / "out"
    domain.write_text(
        json.dumps(
            [
                {
                    "family": "anonymous_family_020",
                    "effect_signature": [["on", 2, "add"]],
                    "preconditions": [["holding", "?r0"]],
                    "add_effects": [["on", "?r0", "?r1"]],
                    "delete_effects": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    positive.write_text(
        json.dumps(
            {"file": "demo.hdf5", "demo": "demo_0", "t_start": 1, "before": [], "added": [["on", "bowl", "plate"]], "deleted": []}
        )
        + "\n",
        encoding="utf-8",
    )
    negative.write_text(
        json.dumps(
            {"family": "anonymous_family_020", "source_event": "demo.hdf5::demo_0::1", "frame_t": 0, "state": []}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_r2_applicability_preflight.py",
            "--domain", str(domain),
            "--positive-events", str(positive),
            "--negative-candidates", str(negative),
            "--output-dir", str(output),
        ],
    )
    main()
    report = json.loads((output / "r2_applicability_preflight.json").read_text(encoding="utf-8"))
    assert report["resolved_candidate_count"] == 1
    detail = json.loads((output / "r2_applicability_candidate_details.jsonl").read_text(encoding="utf-8"))
    assert detail["precondition_binding"] == {"?r0": "bowl"}
    assert detail["event_binding"] == {"?r0": "bowl", "?r1": "plate"}
