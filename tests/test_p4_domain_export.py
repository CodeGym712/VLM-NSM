from __future__ import annotations

import json

from vlm_nsm.research.p4_domain_export import (
    canonical_p4_domain,
    merge_server_push_steps,
    operator_name_for_step,
    run_p4_pipeline,
    step_satisfies_goal_p4,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_place_step_is_split_by_relation():
    assert operator_name_for_step({"operator": "place", "bindings": {"relation": "on"}}) == "place_on"
    assert operator_name_for_step({"operator": "place", "bindings": {"relation": "in"}}) == "place_in"
    assert operator_name_for_step({"operator": "place", "bindings": {"relation": "on"}}, split_place=False) == "place"
    assert operator_name_for_step({"operator": "open", "bindings": {}}) == "open"


def test_push_to_region_satisfies_at_region_goal():
    step = {
        "operator": "push_to_region",
        "bindings": {"obj": "plate_1", "region": "main_table_stove_front_region"},
    }
    goal = {
        "atom": ("at_region", "plate_1", "main_table_stove_front_region"),
        "arg_aliases": [["plate_1"], ["main_table_stove_front_region"]],
    }

    assert step_satisfies_goal_p4(step, goal)


def test_server_push_patch_adds_step_to_push_demo():
    skeletons = [
        {
            "demo_id": "push_the_plate_to_the_front_of_the_stove_demo.hdf5::demo_27",
            "file": "push_the_plate_to_the_front_of_the_stove_demo.hdf5",
            "demo": "demo_27",
            "task": "push_the_plate_to_the_front_of_the_stove",
            "steps": [],
        }
    ]
    patched = merge_server_push_steps(
        skeletons,
        {"demos": {"demo_27": {"T": 220, "atr_frames": 53, "final_ok": True}}},
    )

    assert patched[0]["steps"][0]["operator"] == "push_to_region"
    assert patched[0]["p4_operator_sequence"] == ["push_to_region"]


def test_canonical_domain_splits_place_and_includes_push():
    domain = canonical_p4_domain(
        [
            {"name": "place", "relation_counts": {"on": 4, "in": 2}},
            {"name": "pick", "support_count": 3},
        ],
        {"support_count": 1, "confidence": "conditional", "risk_notes": ["server_checked"]},
    )

    ops = {op["name"]: op for op in domain["operators"]}
    assert ops["place_on"]["support_count"] == 4
    assert ops["place_in"]["support_count"] == 2
    assert ops["push_to_region"]["support_count"] == 1


def test_run_p4_pipeline_on_toy_artifacts(tmp_path):
    p3_dir = tmp_path / "p3"
    package_dir = tmp_path / "package"
    server_dir = tmp_path / "server"
    out_dir = tmp_path / "out"

    _write_json(
        p3_dir / "p3_operator_clauses_v2.json",
        [
            {"name": "pick", "support_count": 1, "confidence": "conditional", "risk_notes": []},
            {"name": "place", "relation_counts": {"on": 1, "in": 0}},
            {"name": "open", "support_count": 0, "confidence": "trusted", "risk_notes": []},
            {"name": "turn_on", "support_count": 0, "confidence": "trusted", "risk_notes": []},
        ],
    )
    _write_jsonl(
        p3_dir / "p3_plan_skeletons_v2.jsonl",
        [
            {
                "demo_id": "toy_place.hdf5::demo_0",
                "file": "toy_place.hdf5",
                "demo": "demo_0",
                "task": "toy_place",
                "steps": [
                    {
                        "operator": "place",
                        "bindings": {"obj": "bowl_1", "target": "plate_1", "relation": "on"},
                        "t_start": 10,
                        "t_end": 10,
                    }
                ],
            },
            {
                "demo_id": "push_the_plate_to_the_front_of_the_stove_demo.hdf5::demo_27",
                "file": "push_the_plate_to_the_front_of_the_stove_demo.hdf5",
                "demo": "demo_27",
                "task": "push_the_plate_to_the_front_of_the_stove",
                "steps": [],
            },
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
                },
                "push_the_plate_to_the_front_of_the_stove": {
                    "goals": [
                        {
                            "internal_predicate": "at_region",
                            "bddl_predicate": "AtRegion",
                            "bddl_args": ["plate_1", "main_table_stove_front_region"],
                            "grounded_args": [{}, {}],
                        }
                    ]
                },
            }
        },
    )
    _write_json(
        server_dir / "p3_push_operator_candidate.json",
        {"support_count": 1, "confidence": "conditional", "risk_notes": ["server_checked"]},
    )
    _write_json(
        server_dir / "at_region_replacement_eval.json",
        {"demos": {"demo_27": {"T": 10, "atr_frames": 3, "final_ok": True}}},
    )

    summary = run_p4_pipeline(p3_dir=p3_dir, package_dir=package_dir, server_validation_dir=server_dir, out_dir=out_dir)

    assert summary["status"] == "P4_PASS"
    assert summary["covered"] == 2
    assert summary["ablation_interpretation"]["server_at_region_patch_gain"] == 1
    assert (out_dir / "p4_domain.json").exists()
    assert (out_dir / "p4_domain.pddl").exists()
    assert (out_dir / "p4_ablation_summary.json").exists()
