"""Tests for shared-predicate ambiguity section in scene prompt."""

from __future__ import annotations

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description


def _make_scene(**kwargs) -> SceneInfo:
    objects = []
    for label, attrs in kwargs.items():
        obj = DetectedObject(
            label=label, category="block",
            attributes=attrs,
        )
        objects.append(obj)
    return SceneInfo(objects=objects)


class TestSharedPredicateAmbiguity:
    def test_on_table_shared_triggers_ambiguity(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
            cube2={"on_table": True, "robot_distance": 0.60, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "Shared predicate ambiguity:" in desc
        assert "on_table is shared by cube1, cube2" in desc
        assert "do NOT uniquely identify an object" in desc

    def test_on_table_not_shared_no_ambiguity(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
            cube2={"on_table": False, "robot_distance": 0.60, "near_edges": []},
        )
        desc = build_scene_description(scene)
        if "Shared predicate ambiguity:" in desc:
            # on_table should not appear in shared notes
            ambiguity_section = desc.split("Shared predicate ambiguity:")[1].split("\n\n")[0]
            assert "on_table is shared" not in ambiguity_section

    def test_no_edge_proximity_triggers_edge_ambiguity(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
            cube2={"on_table": True, "robot_distance": 0.60, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "No object is near any table edge" in desc
        assert "closer to the edge" in desc

    def test_edge_proximity_present_suppresses_edge_ambiguity(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": ["front_edge"]},
            cube2={"on_table": True, "robot_distance": 0.60, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "No object is near any table edge" not in desc

    def test_no_predicates_no_ambiguity_section(self):
        scene = _make_scene(
            cube1={},
            cube2={},
        )
        desc = build_scene_description(scene)
        assert "Shared predicate ambiguity:" not in desc

    def test_single_object_no_ambiguity(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "Shared predicate ambiguity:" not in desc

    def test_shared_region_triggers_note(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": ["front_edge"]},
            cube2={"on_table": True, "robot_distance": 0.60, "near_edges": ["front_edge"]},
        )
        desc = build_scene_description(scene)
        assert "front-center" in desc or "Shared predicate ambiguity" in desc


class TestColorReferenceRules:
    def test_color_reference_rules_present(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "Color reference rules:" in desc
        assert "reddish" in desc
        assert "greenish" in desc

    def test_brightness_unsupported_rule_present(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "Do NOT infer brightness" in desc
        assert "lighter/darker/bright/dull" in desc

    def test_shared_ambiguity_section_referenced_in_guidance(self):
        scene = _make_scene(
            cube1={"on_table": True, "robot_distance": 0.40, "near_edges": []},
        )
        desc = build_scene_description(scene)
        assert "shared predicate ambiguity section" in desc.lower()

    def test_no_color_rules_without_predicates(self):
        scene = _make_scene(
            cube1={},
        )
        desc = build_scene_description(scene)
        assert "Color reference rules:" not in desc
