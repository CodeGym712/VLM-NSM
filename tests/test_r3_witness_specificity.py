from __future__ import annotations

from scripts.audit_r3_witness_specificity import run_specificity


def _domain() -> dict:
    return {
        "operators": [
            {"name": "pick", "parameters": [{"name": "?obj"}], "preconditions_pos": [["handempty"]], "preconditions_neg": [], "effects_add": [["holding", "?obj"]], "effects_del": [["handempty"]]},
            {"name": "place_on", "parameters": [{"name": "?obj"}, {"name": "?target"}], "preconditions_pos": [["holding", "?obj"]], "preconditions_neg": [], "effects_add": [["handempty"], ["on", "?obj", "?target"]], "effects_del": [["holding", "?obj"]]},
            {"name": "place_in", "parameters": [{"name": "?obj"}, {"name": "?container"}], "preconditions_pos": [["holding", "?obj"], ["open", "?container"]], "preconditions_neg": [], "effects_add": [["handempty"], ["in", "?obj", "?container"]], "effects_del": [["holding", "?obj"]]},
            {"name": "open", "parameters": [{"name": "?drawer"}], "preconditions_pos": [["closed", "?drawer"]], "preconditions_neg": [], "effects_add": [["open", "?drawer"]], "effects_del": [["closed", "?drawer"]]},
        ]
    }


def test_specificity_checks_target_binding_and_operator_invariance():
    report = run_specificity(_domain())
    assert report["status"] == "SPECIFICITY_PASS"
    assert all(report["checks"].values())
