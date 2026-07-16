from __future__ import annotations

from vlm_nsm.research.planner_witness_revision import apply_minimal_edit


def test_witness_adds_only_one_precondition():
    domain = {
        "operators": [
            {"name": "place_in", "preconditions_pos": [["holding", "?obj"]], "preconditions_neg": [], "effects_add": [], "effects_del": []},
            {"name": "pick", "preconditions_pos": [], "preconditions_neg": [], "effects_add": [], "effects_del": []},
        ]
    }
    revised, edit = apply_minimal_edit(
        domain,
        {"witness_id": "w1", "operator": "place_in", "literal": ["open", "?container"], "reason": "closed shortcut"},
    )
    assert revised["operators"][0]["preconditions_pos"] == [["holding", "?obj"], ["open", "?container"]]
    assert revised["operators"][1] == domain["operators"][1]
    assert edit["minimality"].startswith("one literal")
