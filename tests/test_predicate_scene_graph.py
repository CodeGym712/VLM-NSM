"""Tests for PredicateSceneGraph — Commit 1."""

import json
import pytest

from vlm_nsm.core.predicate_scene_graph import (
    PredicateSceneGraph,
    RobotState,
    TaskContext,
    build_predicate_scene_graph,
    from_symbolic_scene_graph,
)
from vlm_nsm.symbolic.predicates import (
    SymbolicSceneGraph,
    Predicate,
    on,
    ontable,
    clear,
    handempty,
    graspable,
)
from vlm_nsm.symbolic.goals import StackOrderGoal


class TestPredicateSceneGraph:
    """Core data structure tests."""

    def test_create_empty(self):
        psg = PredicateSceneGraph()
        assert len(psg.objects) == 0
        assert len(psg.predicates) == 0
        assert psg.robot.gripper_empty is True

    def test_add_object_and_relations(self):
        psg = PredicateSceneGraph()
        psg.add_object("cube1", obj_type="cube", color="red", size=0.04)
        psg.add_object("cube2", obj_type="cube", color="green", size=0.05)
        psg.add_relation("on", "cube1", "cube2")
        psg.add_relation("clear", "cube1")
        psg.add_relation("graspable", "cube1")
        psg.add_relation("graspable", "cube2")
        psg.add_relation("handempty", "", "")

        assert "cube1" in psg.objects
        assert "cube2" in psg.objects
        assert psg.has("on", "cube1", "cube2")
        assert psg.has("clear", "cube1")

    def test_set_robot_state(self):
        psg = PredicateSceneGraph()
        psg.set_robot({"x": 0.1, "y": 0.2, "z": 0.9}, gripper_open=1.0)
        assert psg.robot.eef_x == 0.1
        assert psg.robot.eef_y == 0.2
        assert abs(psg.robot.eef_z - 0.9) < 0.001
        assert psg.robot.gripper_empty is True

    def test_set_robot_holding(self):
        psg = PredicateSceneGraph()
        psg.set_robot({"x": 0.0, "y": 0.0, "z": 1.0}, gripper_open=0.0,
                      holding_object="cube1")
        assert psg.robot.holding_object == "cube1"
        assert psg.robot.gripper_empty is False

    def test_add_robot_predicates_holding(self):
        psg = PredicateSceneGraph()
        psg.set_robot({"x": 0, "y": 0, "z": 1.0}, gripper_open=0.0,
                      holding_object="cube1")
        psg.add_robot_predicates()
        assert psg.has("holding", "cube1")

    def test_add_robot_predicates_empty(self):
        psg = PredicateSceneGraph()
        psg.set_robot({"x": 0, "y": 0, "z": 1.0}, gripper_open=1.0)
        psg.add_robot_predicates()
        assert psg.has("handempty")

    def test_to_dict(self):
        psg = PredicateSceneGraph()
        psg.add_object("cube1", color="red", size=0.04)
        psg.add_relation("ontable", "cube1", "")
        psg.set_robot({"x": 0, "y": 0, "z": 1.0})
        d = psg.to_dict()
        assert "objects" in d
        assert "predicates" in d
        assert "robot" in d
        assert "task_context" in d
        assert d["robot"]["gripper_open"] == 1.0

    def test_to_json(self):
        psg = PredicateSceneGraph()
        psg.add_object("cube1", color="red", size=0.04)
        psg.add_relation("ontable", "cube1", "")
        j = psg.to_json()
        d = json.loads(j)
        assert d["robot"]["holding_object"] is None

    def test_predicate_serialization_preserves_observation_provenance(self):
        psg = PredicateSceneGraph()
        psg.add_predicate(Predicate(
            "clear",
            ("cube1",),
            confidence=0.8,
            source="geometry",
            observed_at=123.5,
            freshness_seconds=0.25,
            evidence_id="obs-clear-001",
        ))

        serialized = psg.to_dict()["predicates"][0]

        assert serialized["source"] == "geometry"
        assert serialized["confidence"] == 0.8
        assert serialized["observed_at"] == 123.5
        assert serialized["freshness_seconds"] == 0.25
        assert serialized["evidence_id"] == "obs-clear-001"

    def test_to_prompt_text_2block(self):
        psg = PredicateSceneGraph()
        psg.task_context.task_name = "2-block stacking"
        psg.add_object("cube1", obj_type="cube", color="red", size=0.04)
        psg.add_object("cube2", obj_type="cube", color="green", size=0.05)
        psg.add_predicate(on("cube1", "cube2"))
        psg.add_predicate(ontable("cube2"))
        psg.add_predicate(clear("cube1"))
        psg.add_predicate(graspable("cube1"))
        psg.add_predicate(graspable("cube2"))
        psg.add_predicate(handempty())
        psg.set_robot({"x": -0.11, "y": 0.02, "z": 1.01})
        text = psg.to_prompt_text()
        assert "2-block stacking" in text
        assert "cube1" in text
        assert "cube2" in text
        assert "on" in text
        assert "EEF at" in text

    def test_to_prompt_text_hanoi(self):
        psg = PredicateSceneGraph()
        psg.task_context.task_name = "Hanoi 3-disk"
        psg.add_object("cube1", "cube", "red", 0.04)
        psg.add_object("cube2", "cube", "green", 0.05)
        psg.add_object("cube3", "cube", "blue", 0.04)
        psg.add_predicate(on("cube2", "cube3"))
        psg.add_predicate(on("cube1", "cube2"))
        psg.add_predicate(ontable("cube3"))
        psg.add_predicate(clear("cube1"))
        psg.add_predicate(handempty())
        psg.task_context.known_supports = ["peg1", "peg2", "peg3", "table"]
        text = psg.to_prompt_text()
        assert "Hanoi" in text
        assert "cube1" in text
        assert "cube3" in text

    def test_from_symbolic_scene_graph(self):
        sg = SymbolicSceneGraph()
        from vlm_nsm.symbolic.predicates import ObjectSymbol as OS
        sg.objects["cube1"] = OS(object_id="cube1", color="red", obj_type="cube", size=0.04)
        sg.objects["cube2"] = OS(object_id="cube2", color="green", obj_type="cube", size=0.05)
        sg.predicates.add(on("cube1", "cube2"))

        robot = RobotState(eef_x=0.1, eef_y=0.2, eef_z=1.0)
        ctx = TaskContext(task_name="test")
        psg = from_symbolic_scene_graph(sg, robot, ctx)
        assert psg.has("on", "cube1", "cube2")
        assert psg.robot.eef_x == 0.1
        assert psg.task_context.task_name == "test"


class TestBuilder:
    """Builder from environment tests (need MuJoCo)."""

    @pytest.mark.skip(reason="Needs MuJoCo environment")
    def test_build_2block(self):
        import os; os.environ["MUJOCO_GL"] = "egl"
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

        env = RobosuiteEnvWrapper(
            env_name="Stack", robots="Panda",
            cube_names=["cube1", "cube2"], render_mode="none")
        env.reset()

        psg = build_predicate_scene_graph(env, ["cube1", "cube2"])
        assert len(psg.objects) == 2
        assert psg.robot.gripper_empty is True
        text = psg.to_prompt_text()
        assert "cube1" in text
        assert "cube2" in text
        print(f"\n=== Prompt Text (2-block) ===\n{text}")


class TestTaskContext:
    def test_defaults(self):
        ctx = TaskContext()
        assert "reach" in ctx.available_skills
        assert "table" in ctx.known_supports
        assert ctx.horizon == 1000

    def test_custom(self):
        ctx = TaskContext(
            task_name="hanoi",
            available_skills=["reach", "grasp", "lift", "place"],
            known_supports=["peg1", "peg2", "peg3"],
            forbidden_relations=["on(cube_bottom, cube_top)"],
            max_replans=5,
            horizon=5000,
        )
        d = ctx.to_dict()
        assert d["task_name"] == "hanoi"
        assert len(d["available_skills"]) == 4
        assert "peg1" in d["known_supports"]
