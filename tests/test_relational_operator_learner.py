from __future__ import annotations

from vlm_nsm.research.relational_operator_learner import applicability_binding, canonicalize_event, family_binding, learn_operator_families
from scripts.run_r2_relational_operator_learner import split_by_template


def _event(obj: str, support: str, *, include_support: bool = True) -> dict:
    before = [["handempty"]]
    if include_support:
        before.append(["on", obj, support])
    return {
        "file": "x.hdf5",
        "demo": obj,
        "t_start": 1,
        "before": before,
        "added": [["holding", obj]],
        "deleted": [["handempty"], ["on", obj, support]],
        "operator_type": "pick",
        "task": "must-not-be-read",
    }


def test_alpha_renaming_keeps_same_effect_family():
    a, b = canonicalize_event(_event("bowl_1", "table")), canonicalize_event(_event("bowl_9", "tray"))
    assert a["added"] == b["added"]
    assert a["deleted"] == b["deleted"]
    families = learn_operator_families([_event("bowl_1", "table"), _event("bowl_9", "tray")])
    assert len(families) == 1
    assert families[0]["provenance"]["training_only"] is True


def test_partial_support_is_recorded_as_negative_witness():
    families = learn_operator_families([_event("bowl_1", "table"), _event("bowl_2", "table", include_support=False)], min_support=0.8)
    assert families[0]["negative_witnesses"]
    assert any(w["support"] == 0.5 for w in families[0]["negative_witnesses"])


def test_template_split_keeps_files_disjoint():
    rows = [_event("bowl_1", "table"), _event("bowl_2", "table")]
    rows[1]["file"] = "other.hdf5"
    train, heldout = split_by_template(rows, seed=1, eval_fraction=0.5)
    assert {row["file"] for row in train}.isdisjoint({row["file"] for row in heldout})


def test_family_binding_ignores_unrelated_scene_entities():
    family = learn_operator_families([_event("bowl_1", "table")])[0]
    row = _event("bowl_9", "tray")
    row["before"].insert(0, ["deactivated", "unrelated_device"])
    binding = family_binding(row, family)
    assert binding["?r0"] == "bowl_9"
    assert binding["?r1"] == "tray"


def test_canonicalization_does_not_promote_distractors_to_roles():
    row = _event("bowl_1", "table")
    row["before"].append(["on", "unrelated", "table"])
    canonical = canonicalize_event(row)
    assert canonical["roles"] == [
        {"name": "?r0", "type": "object"},
        {"name": "?r1", "type": "support"},
    ]


def test_applicability_binding_allows_unbound_effect_only_roles():
    family = {
        "preconditions": [["holding", "?r0"]],
        "add_effects": [["on", "?r0", "?r1"]],
        "delete_effects": [],
    }
    row = {"added": [["on", "bowl_1", "plate_1"]], "deleted": []}
    binding, missing_roles = applicability_binding(row, family)
    assert binding == {"?r0": "bowl_1"}
    assert missing_roles == []
