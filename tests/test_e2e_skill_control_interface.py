import ast
from pathlib import Path

from vlm_nsm.core.goal_parser import MockGoalParser
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.experiments.run_e2e_language_to_sim import run_demo


SCRIPT = Path("src/vlm_nsm/experiments/run_e2e_language_to_sim.py")


class DummyEnv:
    pass


def _find_arg(parsed, name):
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            func = getattr(node, "func", None)
            if func and getattr(func, "attr", None) == "add_argument":
                args = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
                if name in args:
                    return True
    return False


def test_control_interface_cli_arg_exists():
    parsed = ast.parse(SCRIPT.read_text())
    assert _find_arg(parsed, "--control_interface")
    assert "skill_executor" in SCRIPT.read_text()


def test_run_demo_skill_executor_path(monkeypatch, tmp_path):
    captured = {}

    def fake_run_episode(**kwargs):
        captured.update(kwargs)
        return {
            "goal_success": True,
            "num_steps": 77,
            "failure_type": None,
            "bc_steps": 0,
            "scripted_steps": 77,
        }

    monkeypatch.setattr(
        "vlm_nsm.experiments.robosuite_episode_runner.run_episode",
        fake_run_episode,
    )

    demo = {
        "instruction": "Put cube1 on cube2.",
        "task_type": "stacking",
        "cubes": ["cube1", "cube2"],
        "target_stack": ["cube2", "cube1"],
    }

    result = run_demo(
        env=DummyEnv(),
        demo=demo,
        planner=PlannerAdapter("internal_stacking"),
        bc_model=None,
        device=None,
        action_mean=None,
        action_std=None,
        exec_config={},
        output_dir=tmp_path,
        use_bc=False,
        parser=MockGoalParser(),
        control_interface="skill_executor",
    )

    assert result["success"] is True
    assert result["control_interface"] == "skill_executor"
    assert result["skill_plan"]["commands"][0]["skill_name"] == "pick_place"
    assert result["skill_results"][0]["steps"] == 77
    assert captured["actions"][0].args == ["cube1", "table", "cube2"]
