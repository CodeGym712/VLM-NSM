from __future__ import annotations

from vlm_nsm.research.p3_symbolic_induction import (
    build_object_centric_episodes,
    build_counterexample_revision_plan,
    classify_event,
    canonical_goal_arg,
    infer_plan_skeletons,
    infer_operator_clauses_v1,
    normalize_demo_rows,
    raw_atoms_from_predicates,
    replay_goals_against_skeletons,
)


def _row(t: int, predicates: dict):
    return {
        "file": "demo_file.hdf5",
        "ds": "libero_goal",
        "demo": "demo_0",
        "t": t,
        "T": 5,
        "predicates": predicates,
    }


def test_canonicalizes_joint0_but_preserves_instance_id():
    atoms = raw_atoms_from_predicates({"holding": "akita_black_bowl_1_joint0"})
    assert ("holding", "akita_black_bowl_1") in atoms["holding"]


def test_holding_requires_three_consecutive_frames():
    rows = [
        _row(0, {"holding": None, "handempty": True, "on": [], "in": [], "open": [], "closed": [], "activated": [], "deactivated": [], "at_region": []}),
        _row(1, {"holding": "obj_1_joint0", "handempty": False, "on": [], "in": [], "open": [], "closed": [], "activated": [], "deactivated": [], "at_region": []}),
        _row(2, {"holding": "obj_1_joint0", "handempty": False, "on": [], "in": [], "open": [], "closed": [], "activated": [], "deactivated": [], "at_region": []}),
        _row(3, {"holding": None, "handempty": True, "on": [], "in": [], "open": [], "closed": [], "activated": [], "deactivated": [], "at_region": []}),
    ]

    frames = normalize_demo_rows(rows, {"demo_file.hdf5": "toy"}, holding_min_frames=3)

    assert ["holding", "obj_1"] not in [list(atom) for frame in frames for atom in frame.trusted]
    assert any(atom == ("holding", "obj_1") for frame in frames for atom in frame.conditional)


def test_on_table_is_masked_when_specific_support_exists():
    rows = [
        _row(
            0,
            {
                "holding": None,
                "handempty": True,
                "on": [
                    {"top": "cookies_1_joint0", "bot": "table"},
                    {"top": "cookies_1_joint0", "bot": "cookie_box_1_joint0"},
                ],
                "in": [],
                "open": [],
                "closed": [],
                "activated": [],
                "deactivated": [],
                "at_region": [{"obj": "cookies_1_joint0", "region": "bad_region"}],
            },
        )
    ]

    frame = normalize_demo_rows(rows, {"demo_file.hdf5": "toy"})[0]

    assert ("on", "cookies_1", "cookie_box_1") in frame.trusted
    assert ("on", "cookies_1", "table") in frame.conditional
    assert ("at_region", "cookies_1", "bad_region") in frame.excluded


def test_classifies_pick_event():
    operator, bindings, _risk = classify_event(
        {
            "added": [["holding", "bowl_1"]],
            "deleted": [["handempty"]],
            "risk_notes": [],
        }
    )

    assert operator == "pick"
    assert bindings == {"obj": "bowl_1"}


def test_fuses_release_with_recent_relation_into_place():
    events = [
        {
            "file": "demo_file.hdf5",
            "demo": "demo_0",
            "task": "toy",
            "operator_type": "relocate_effect",
            "bindings": {"obj": "bowl_1", "target": "plate_1"},
            "t_start": 10,
            "t_end": 10,
            "confidence": "trusted",
            "operator_risk_notes": [],
        },
        {
            "file": "demo_file.hdf5",
            "demo": "demo_0",
            "task": "toy",
            "operator_type": "release",
            "bindings": {"obj": "bowl_1"},
            "t_start": 15,
            "t_end": 16,
            "confidence": "trusted",
            "operator_risk_notes": [],
        },
    ]
    skeletons, summary = infer_plan_skeletons(
        events,
        {"demo_file.hdf5::demo_0": {"file": "demo_file.hdf5", "demo": "demo_0", "task": "toy", "ds": "toy"}},
        relation_lookback=20,
    )

    assert summary["operator_counts"]["place"] == 1
    assert skeletons[0]["steps"][0]["bindings"] == {
        "obj": "bowl_1",
        "target": "plate_1",
        "relation": "on-or-in",
    }


def test_builds_object_episode_and_operator_clauses():
    skeletons = [
        {
            "demo_id": "demo_file.hdf5::demo_0",
            "file": "demo_file.hdf5",
            "demo": "demo_0",
            "task": "toy",
            "steps": [
                {
                    "operator": "pick",
                    "bindings": {"obj": "bowl_1"},
                    "t_start": 5,
                    "t_end": 6,
                    "confidence": "trusted",
                    "risk_notes": [],
                },
                {
                    "operator": "place",
                    "bindings": {"obj": "bowl_1", "target": "plate_1"},
                    "t_start": 15,
                    "t_end": 16,
                    "confidence": "trusted",
                    "risk_notes": [],
                },
            ],
        }
    ]

    episodes, summary = build_object_centric_episodes(skeletons)
    clauses, _report = infer_operator_clauses_v1(episodes)

    assert summary["episode_type_counts"]["pick_place"] == 1
    assert episodes[0]["duration"] == 10
    assert {clause["name"] for clause in clauses} == {"pick", "place"}


def test_replays_on_goal_against_place_skeleton():
    skeletons = [
        {
            "demo_id": "demo_file.hdf5::demo_0",
            "file": "demo_file.hdf5",
            "demo": "demo_0",
            "task": "toy_place",
            "operator_sequence": ["pick", "place"],
            "steps": [
                {
                    "operator": "place",
                    "bindings": {"obj": "bowl_1", "target": "plate_1", "relation": "on"},
                    "t_start": 10,
                    "t_end": 11,
                }
            ],
        }
    ]

    replay, counterexamples, _md = replay_goals_against_skeletons(
        skeletons,
        {"toy_place": [("on", "bowl_1", "plate_1")]},
    )

    assert replay["summary"]["covered"] == 1
    assert counterexamples == []


def test_at_region_goal_is_blocked_when_excluded_from_p3():
    skeletons = [
        {
            "demo_id": "demo_file.hdf5::demo_0",
            "file": "demo_file.hdf5",
            "demo": "demo_0",
            "task": "toy_push",
            "operator_sequence": [],
            "steps": [],
        }
    ]

    replay, counterexamples, _md = replay_goals_against_skeletons(
        skeletons,
        {"toy_push": [("at_region", "plate_1", "main_table_stove_front_region")]},
    )

    assert replay["summary"]["blocked_excluded_predicate"] == 1
    assert counterexamples[0]["reason"] == "goal_depends_on_at_region_excluded_from_P3"


def test_canonical_goal_arg_maps_drawer_region_to_level():
    assert canonical_goal_arg("wooden_cabinet_1_top_region", "in") == "wooden_cabinet_1_top_level"


def test_builds_counterexample_revision_plan_by_reason():
    plan, md = build_counterexample_revision_plan(
        [
            {"reason": "missing_place_step", "demo_id": "d1", "task": "t1"},
            {"reason": "missing_place_step", "demo_id": "d2", "task": "t1"},
            {"reason": "goal_depends_on_at_region_excluded_from_P3", "demo_id": "d3", "task": "push"},
        ]
    )

    assert plan["reason_counts"]["missing_place_step"] == 2
    assert plan["groups"][0]["reason"] == "missing_place_step"
    assert "Counterexample-driven" in md
