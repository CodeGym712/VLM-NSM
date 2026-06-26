from vlm_nsm.skills.skill_types import SkillCommand
from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor


class DummyEnv:
    pass


def test_scripted_skill_executor_wraps_pick_place(monkeypatch):
    captured = {}

    def fake_run_episode(**kwargs):
        captured.update(kwargs)
        return {
            "goal_success": True,
            "num_steps": 123,
            "failure_type": None,
            "bc_steps": 10,
            "scripted_steps": 113,
        }

    monkeypatch.setattr(
        "vlm_nsm.experiments.robosuite_episode_runner.run_episode",
        fake_run_episode,
    )

    command = SkillCommand(
        skill_name="pick_place",
        object_id="cube1",
        target_object_id="cube2",
        relation="on",
        source="table",
    )
    result = ScriptedSkillExecutor({"horizon": 999}).execute(
        command,
        DummyEnv(),
        context={"cubes": ["cube1", "cube2"], "override_phases": {0}},
    )

    assert result.success is True
    assert result.steps == 123
    assert captured["actions"][0].name == "move"
    assert captured["actions"][0].args == ["cube1", "table", "cube2"]
    assert captured["goal_predicates"] == [["on", "cube1", "cube2"]]
    assert captured["horizon"] == 999
    assert captured["override_phases"] == {0}


def test_scripted_skill_executor_requires_target():
    command = SkillCommand(skill_name="pick_place", object_id="cube1")

    try:
        ScriptedSkillExecutor().execute(command, DummyEnv())
    except ValueError as exc:
        assert "target_object_id" in str(exc)
    else:
        raise AssertionError("expected missing target to fail")


def test_scripted_skill_executor_rejects_unknown_skill():
    command = SkillCommand(skill_name="dance", object_id="cube1")

    try:
        ScriptedSkillExecutor().execute(command, DummyEnv())
    except NotImplementedError as exc:
        assert "dance" in str(exc)
    else:
        raise AssertionError("expected unknown skill to fail")
