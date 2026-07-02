"""Tests for EmbodiedTransition recording in the MuJoCo coarse runner."""

from types import SimpleNamespace

from vlm_nsm.core.predicate_scene_graph import PredicateSceneGraph
from vlm_nsm.data.embodied_transition import FailureAttribution
from vlm_nsm.embodied.skills.base_skill import SkillResult
from vlm_nsm.experiments import run_coarse_program_mujoco as runner
from vlm_nsm.symbolic.predicates import Predicate


def graph(*predicates: tuple[str, tuple[str, ...]]) -> PredicateSceneGraph:
    scene = PredicateSceneGraph()
    for name, args in predicates:
        scene.add_predicate(Predicate(name, args, source="oracle"))
    return scene


def test_expected_effect_binds_existing_coarse_contracts():
    pick = runner._expected_effect_for("atomic_pick", "cube1", "")
    place = runner._expected_effect_for("atomic_place", "cube1", "cube2")

    assert {item["name"] for item in pick["add"]} == {"holding"}
    assert {item["name"] for item in pick["delete"]} == {"handempty", "ontable", "clear"}
    assert {tuple(item["args"]) for item in place["add"]} == {
        ("cube1", "cube2"), ("cube1",), ()
    }


def test_effect_check_detects_reliable_model_mismatch():
    effect = runner._expected_effect_for("atomic_pick", "cube1", "")
    post_scene = graph(("handempty", ()))

    assert runner._scene_satisfies_effect(post_scene, effect) is False


def test_oracle_scene_injects_grounded_holding_object(monkeypatch):
    base = graph(("handempty", ()), ("clear", ("cube1",)))

    class FakeEnv:
        def get_gripper_state(self):
            return {"has_object": True}

    monkeypatch.setattr(runner, "build_predicate_scene_graph", lambda *args, **kwargs: base)

    scene = runner._build_oracle_scene(
        FakeEnv(), ["cube1"], "stacking_2block", holding_object="cube1"
    )

    assert scene.has("holding", "cube1")
    assert not scene.has("handempty")
    assert not scene.has("clear", "cube1")


def test_oracle_scene_removes_synthetic_ontable_for_lifted_object(monkeypatch):
    base = graph(("ontable", ("cube1",)), ("holding", ("cube1",)))

    class FakeEnv:
        def get_all_object_poses(self):
            return {"cube1": {"x": 0.0, "y": 0.0, "z": 1.0}}

        def get_object_pose(self, name):
            assert name == "table"
            return {"x": 0.0, "y": 0.0, "z": 0.8}

        def get_gripper_state(self):
            return {"has_object": True}

    monkeypatch.setattr(runner, "build_predicate_scene_graph", lambda *args, **kwargs: base)

    scene = runner._build_oracle_scene(
        FakeEnv(), ["cube1"], "stacking_2block", holding_object="cube1"
    )

    assert not scene.has("ontable", "cube1")


def test_run_records_each_atomic_option(monkeypatch):
    class FakeBridge:
        def execution_sequence(self, coarse_steps, objects):
            return [
                {"skill": "atomic_pick", "object": "cube1"},
                {"skill": "atomic_place", "object": "cube1", "target": "cube2"},
            ]

    class FakePick:
        def execute_with_pid(self, env, pid, params):
            return SkillResult(True, 5, diagnostics={"contact_detected": True})

    class FakePlace:
        def execute_with_pid(self, env, pid, params):
            return SkillResult(True, 4)

    scenes = iter([
        graph(("handempty", ()), ("clear", ("cube1",)), ("ontable", ("cube1",))),
        graph(("holding", ("cube1",))),
        graph(("holding", ("cube1",))),
        graph(("on", ("cube1", "cube2")), ("clear", ("cube1",)), ("handempty", ())),
    ])

    class FakeRecorder:
        def __init__(self):
            self.calls = []

        def record_option(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(transition_id=f"run:{kwargs['sequence_index']}")

    recorder = FakeRecorder()
    monkeypatch.setattr(runner, "CoarsePlannerBridge", FakeBridge)
    monkeypatch.setattr(runner, "AtomicPickSkill", FakePick)
    monkeypatch.setattr(runner, "AtomicPlaceSkill", FakePlace)
    monkeypatch.setattr(runner, "_build_oracle_scene", lambda *args, **kwargs: next(scenes))

    result = runner.run_coarse_program(
        env=object(),
        pid=object(),
        coarse_steps=[{"op": "move"}],
        objects=["cube1", "cube2", "table"],
        recorder=recorder,
        scene_objects=["cube1", "cube2"],
        task_name="stacking_2block",
    )

    assert result["success"] is True
    assert result["transition_ids"] == ["run:0", "run:1"]
    assert len(recorder.calls) == 2
    assert recorder.calls[0]["grounded_option"] == {"name": "pick", "arguments": ["cube1"]}
    assert recorder.calls[1]["grounded_option"] == {
        "name": "place", "arguments": ["cube1", "cube2"]
    }
    assert recorder.calls[0]["attribution"] is None


def test_successful_skill_with_unsatisfied_effect_records_model_mismatch(monkeypatch):
    class FakeBridge:
        def execution_sequence(self, coarse_steps, objects):
            return [{"skill": "atomic_pick", "object": "cube1"}]

    class FakePick:
        def execute_with_pid(self, env, pid, params):
            return SkillResult(True, 5)

    class FakeRecorder:
        def __init__(self):
            self.call = None

        def record_option(self, **kwargs):
            self.call = kwargs
            return SimpleNamespace(transition_id="run:0")

    recorder = FakeRecorder()
    scenes = iter([graph(("handempty", ())), graph(("handempty", ()))])
    monkeypatch.setattr(runner, "CoarsePlannerBridge", FakeBridge)
    monkeypatch.setattr(runner, "AtomicPickSkill", FakePick)
    monkeypatch.setattr(runner, "_build_oracle_scene", lambda *args, **kwargs: next(scenes))

    result = runner.run_coarse_program(
        object(), object(), [{"op": "pick"}], ["cube1", "table"],
        recorder=recorder, scene_objects=["cube1"], task_name="stacking_2block",
    )

    assert result["success"] is False
    assert result["failure_type"] == "model_mismatch"
    assert recorder.call["attribution"] is FailureAttribution.MODEL_MISMATCH
    assert recorder.call["result"].success is False
