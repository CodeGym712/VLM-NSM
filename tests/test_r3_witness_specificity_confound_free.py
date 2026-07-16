from __future__ import annotations

from scripts.run_r3_witness_specificity_confound_free import run_confound_free_checks


def _domain() -> dict:
    return {
        "operators": [
            {
                "name": "place_in",
                "parameters": [{"name": "?obj"}, {"name": "?container"}],
                "preconditions_pos": [["holding", "?obj"], ["open", "?container"]],
                "preconditions_neg": [],
                "effects_add": [["handempty"], ["in", "?obj", "?container"]],
                "effects_del": [["holding", "?obj"]],
            }
        ]
    }


def test_confound_free_checks_require_open_after_holding_is_satisfied():
    report = run_confound_free_checks(_domain())
    assert report["status"] == "PASS"
    assert all(row["holding_satisfied"] for row in report["rows"])
    rejected = [row for row in report["rows"] if row["expected"] == "REJECT"]
    assert all("missing_positive:('open'" in row["actual_reason"] for row in rejected)
