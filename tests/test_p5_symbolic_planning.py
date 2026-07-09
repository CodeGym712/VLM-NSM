from __future__ import annotations

import json

from vlm_nsm.research.p5_symbolic_planning import (
    atom_satisfies_goal_strict,
    apply_action,
    atom_from_list,
    bfs_plan,
    goals_satisfied,
    ground_actions,
    infer_type_universe,
    filter_domain_by_support,
    operator_support_by_task,
    required_movable_goal_symbols,
    run_p5_composition_audit,
    run_p5_loto_planning_audit,
    run_p5_planning_audit,
    support_excluding_task,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _toy_domain():
    return {
        "operators": [
            {
                "name": "pick",
                "parameters": [{"name": "?obj", "type": "object"}],
                "preconditions_pos": [["handempty"]],
                "preconditions_neg": [["holding", "?obj"]],
                "effects_add": [["holding", "?obj"]],
                "effects_del": [["handempty"]],
            },
            {
                "name": "place_on",
                "parameters": [{"name": "?obj", "type": "object"}, {"name": "?target", "type": "support"}],
                "preconditions_pos": [["holding", "?obj"]],
                "preconditions_neg": [],
                "effects_add": [["handempty"], ["on", "?obj", "?target"]],
                "effects_del": [["holding", "?obj"]],
            },
        ]
    }


def test_bfs_finds_pick_place_plan():
    initial = frozenset({("handempty",), ("on", "bowl_1", "table")})
    goals = [{"atom": ("on", "bowl_1", "plate_1"), "arg_aliases": [["bowl_1"], ["plate_1"]]}]
    universe = infer_type_universe(initial, goals)
    actions = ground_actions(_toy_domain(), universe)

    plan, stats = bfs_plan(initial, goals, actions, max_depth=3)

    assert stats["reason"] == "plan_found"
    assert [action.name for action in plan] == ["pick", "place_on"]
    assert goals_satisfied(apply_action(plan[-1], apply_action(plan[0], initial)), goals)


def test_atom_from_list_returns_tuple():
    assert atom_from_list(["holding", "obj"]) == ("holding", "obj")


def test_strict_goal_matching_rejects_wrong_drawer_level():
    goal = {
        "atom": ("in", "bowl_1", "wooden_cabinet_1_top_level"),
        "arg_aliases": [["bowl_1"], ["wooden_cabinet_1", "wooden_cabinet_1_top_level"]],
    }

    assert atom_satisfies_goal_strict(("in", "bowl_1", "wooden_cabinet_1_top_level"), goal)
    assert not atom_satisfies_goal_strict(("in", "bowl_1", "wooden_cabinet_1_bottom_level"), goal)


def test_strict_goal_matching_allows_device_button_alias():
    goal = {
        "atom": ("activated", "flat_stove_1"),
        "arg_aliases": [["flat_stove_1", "flat_stove_1_main"]],
    }

    assert atom_satisfies_goal_strict(("activated", "flat_stove_1_button"), goal)
    assert not atom_satisfies_goal_strict(("activated", "wooden_cabinet_1_top_level"), goal)


def test_run_p5_planning_audit_on_toy_artifacts(tmp_path):
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    package_dir = tmp_path / "package"
    out_dir = tmp_path / "out"

    _write_jsonl(
        p3_dir / "p3_normalized_traces.jsonl",
        [
            {
                "file": "toy.hdf5",
                "demo": "demo_0",
                "task": "toy_place",
                "t": 0,
                "T": 2,
                "trusted_predicates": [["handempty"], ["on", "akita_black_bowl_1", "table"], ["on", "plate_1", "table"]],
            }
        ],
    )
    _write_json(p4_dir / "p4_domain.json", _toy_domain())
    _write_jsonl(
        p4_dir / "p4_plan_skeletons.jsonl",
        [
            {
                "demo_id": "toy.hdf5::demo_0",
                "steps": [
                    {"operator": "pick", "bindings": {"obj": "bowl_1"}},
                    {"operator": "place", "bindings": {"obj": "bowl_1", "target": "plate_1", "relation": "on"}},
                ],
            }
        ],
    )
    _write_json(
        package_dir / "metadata" / "bddl_task_mapping.json",
        {
            "mappings": {
                "toy_place": {
                    "goals": [
                        {
                            "internal_predicate": "on",
                            "bddl_predicate": "On",
                            "bddl_args": ["akita_black_bowl_1", "plate_1"],
                            "grounded_args": [{}, {}],
                        }
                    ]
                }
            }
        },
    )

    summary = run_p5_planning_audit(p3_dir, p4_dir, package_dir, out_dir, max_depth=3)

    assert summary["status"] == "P5_INITIAL_PLANNING_PASS"
    assert summary["solved"] == 1
    assert (out_dir / "p5_initial_planning_results.jsonl").exists()


def test_required_movable_goal_symbols_detects_object_and_movable_target():
    goals = [{"atom": ("on", "cream_cheese_1", "akita_black_bowl_1"), "arg_aliases": []}]

    assert required_movable_goal_symbols(goals) == {"cream_cheese_1", "akita_black_bowl_1"}


def test_run_p5_composition_audit_skips_absent_objects(tmp_path):
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    package_dir = tmp_path / "package"
    out_dir = tmp_path / "out"

    _write_jsonl(
        p3_dir / "p3_normalized_traces.jsonl",
        [
            {
                "file": "toy.hdf5",
                "demo": "demo_0",
                "task": "toy_place",
                "t": 0,
                "T": 2,
                "trusted_predicates": [["handempty"], ["on", "akita_black_bowl_1", "table"], ["on", "plate_1", "table"]],
            }
        ],
    )
    _write_json(p4_dir / "p4_domain.json", _toy_domain())
    _write_json(
        package_dir / "metadata" / "bddl_task_mapping.json",
        {
            "mappings": {
                "toy_place": {
                    "goals": [
                        {
                            "internal_predicate": "on",
                            "bddl_predicate": "On",
                            "bddl_args": ["akita_black_bowl_1", "plate_1"],
                            "grounded_args": [{}, {}],
                        }
                    ]
                },
                "toy_absent": {
                    "goals": [
                        {
                            "internal_predicate": "on",
                            "bddl_predicate": "On",
                            "bddl_args": ["wine_bottle_1", "plate_1"],
                            "grounded_args": [{}, {}],
                        }
                    ]
                },
            }
        },
    )

    summary = run_p5_composition_audit(p3_dir, p4_dir, package_dir, out_dir, max_depth=3)

    assert summary["attempted"] == 1
    assert summary["solved"] == 1
    assert summary["skipped_absent_object"] == 1


def test_support_filter_removes_zero_support_operators():
    domain = {
        "operators": [
            {"name": "pick"},
            {"name": "place_on"},
            {"name": "turn_on"},
        ]
    }

    filtered = filter_domain_by_support(domain, {"pick": 3, "place_on": 2})

    assert [op["name"] for op in filtered["operators"]] == ["pick", "place_on"]
    assert filtered["heldout_removed_operators"] == ["turn_on"]


def test_loto_support_counts_exclude_heldout_task():
    support = operator_support_by_task(
        [
            {"task": "train_task", "steps": [{"operator": "pick"}, {"operator": "place", "bindings": {"relation": "on"}}]},
            {"task": "heldout_task", "steps": [{"operator": "turn_on"}]},
        ]
    )

    counts = support_excluding_task(support, "heldout_task")

    assert counts["pick"] == 1
    assert counts["place_on"] == 1
    assert counts["turn_on"] == 0


def test_run_p5_loto_audit_on_toy_artifacts(tmp_path):
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    package_dir = tmp_path / "package"
    out_dir = tmp_path / "out"

    _write_jsonl(
        p3_dir / "p3_normalized_traces.jsonl",
        [
            {
                "file": "train.hdf5",
                "demo": "demo_0",
                "task": "train_place",
                "t": 0,
                "T": 2,
                "trusted_predicates": [["handempty"], ["on", "akita_black_bowl_1", "table"], ["on", "plate_1", "table"]],
            },
            {
                "file": "heldout.hdf5",
                "demo": "demo_0",
                "task": "heldout_place",
                "t": 0,
                "T": 2,
                "trusted_predicates": [["handempty"], ["on", "akita_black_bowl_1", "table"], ["on", "plate_1", "table"]],
            },
        ],
    )
    _write_json(p4_dir / "p4_domain.json", _toy_domain())
    _write_jsonl(
        p4_dir / "p4_plan_skeletons.jsonl",
        [
            {
                "demo_id": "train.hdf5::demo_0",
                "task": "train_place",
                "steps": [
                    {"operator": "pick", "bindings": {"obj": "akita_black_bowl_1"}},
                    {"operator": "place", "bindings": {"obj": "akita_black_bowl_1", "target": "plate_1", "relation": "on"}},
                ],
            },
            {
                "demo_id": "heldout.hdf5::demo_0",
                "task": "heldout_place",
                "steps": [
                    {"operator": "pick", "bindings": {"obj": "akita_black_bowl_1"}},
                    {"operator": "place", "bindings": {"obj": "akita_black_bowl_1", "target": "plate_1", "relation": "on"}},
                ],
            },
        ],
    )
    _write_json(
        package_dir / "metadata" / "bddl_task_mapping.json",
        {
            "mappings": {
                "train_place": {
                    "goals": [
                        {
                            "internal_predicate": "on",
                            "bddl_predicate": "On",
                            "bddl_args": ["akita_black_bowl_1", "plate_1"],
                            "grounded_args": [{}, {}],
                        }
                    ]
                },
                "heldout_place": {
                    "goals": [
                        {
                            "internal_predicate": "on",
                            "bddl_predicate": "On",
                            "bddl_args": ["akita_black_bowl_1", "plate_1"],
                            "grounded_args": [{}, {}],
                        }
                    ]
                },
            }
        },
    )

    summary = run_p5_loto_planning_audit(p3_dir, p4_dir, package_dir, out_dir, max_depth=3)

    assert summary["solved_splits"] == 2
    assert summary["solved"] == 2
    assert (out_dir / "p5_loto_split_summary.jsonl").exists()
