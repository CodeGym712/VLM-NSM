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
    run_p5_planning_audit,
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
                "trusted_predicates": [["handempty"], ["on", "bowl_1", "table"]],
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
                            "bddl_args": ["bowl_1", "plate_1"],
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
