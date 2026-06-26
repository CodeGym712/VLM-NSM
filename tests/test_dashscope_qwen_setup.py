from pathlib import Path

from vlm_nsm.core.data_types import GoalParseResult
from vlm_nsm.language.online_qwen_parser import (
    DEFAULT_DASHSCOPE_API_BASE,
    OnlineQwenGoalParser,
)


class _FakeQwenParser(OnlineQwenGoalParser):
    def __init__(self, tmp_path=None, **kwargs):
        super().__init__(**kwargs)
        self._tmp_path = tmp_path
        self.last_prompt = None

    def _call_api(self, prompt: str) -> str:
        self.last_prompt = prompt
        return (
            '{"task_type":"stacking","object_bindings":{"top block":"cube1",'
            '"bottom block":"cube2"},"goal_predicates":[["on","cube1","cube2"],'
            '["on","cube2","table"]]}'
        )


def test_default_dashscope_api_base():
    assert DEFAULT_DASHSCOPE_API_BASE == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_parse_accepts_e2e_kwargs():
    parser = _FakeQwenParser(model="qwen-plus", api_key="dummy")
    result = parser.parse(
        instruction="Put cube1 on cube2.",
        task_type="stacking",
        objects=["cube1", "cube2"],
        n_blocks=2,
        unused_kwarg=True,
    )
    assert result.success is True
    assert result.goal_json["task_type"] == "stacking"


def test_prompt_uses_dynamic_objects():
    parser = _FakeQwenParser(model="qwen-plus", api_key="dummy")
    parser.parse(
        instruction="Stack cube3 on cube2, cube2 on cube1.",
        available_objects=["cube1", "cube2", "cube3"],
        n_blocks=3,
        scene_description="- cube1: red cube\n- cube2: green cube\n- cube3: blue cube",
    )
    prompt = parser.last_prompt
    assert prompt is not None
    assert '"cube3"' in prompt
    assert "Scene inventory:" in prompt
    assert "blue cube" in prompt
    assert "middle block" in prompt


def test_raw_output_path_returned(tmp_path):
    parser = _FakeQwenParser(
        model="qwen-plus",
        api_key="dummy",
        save_raw_outputs=True,
        raw_output_dir=str(tmp_path),
    )
    result = parser.parse("Put cube1 on cube2.", objects=["cube1", "cube2"])
    assert result.raw_output_path is not None
    assert Path(result.raw_output_path).exists()


def test_goal_parse_result_has_raw_output_path_field():
    result = GoalParseResult(success=False, goal_json=None, raw_output_path="x.txt")
    assert result.raw_output_path == "x.txt"


def test_dashscope_scripts_exist():
    for path in [
        "scripts/smoke_dashscope_qwen.sh",
        "scripts/run_dashscope_parse5.sh",
        "scripts/run_dashscope_e2e_demo.sh",
        "configs/dashscope_qwen.env.example",
        "docs/dashscope_qwen_setup.md",
    ]:
        assert Path(path).exists()


def test_smoke_script_uses_no_literal_secret():
    text = Path("configs/dashscope_qwen.env.example").read_text()
    assert "YOUR_DASHSCOPE_API_KEY_HERE" in text
    assert "sk-" not in text
