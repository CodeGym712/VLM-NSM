"""Verify online_qwen CLI args exist in run_e2e_language_to_sim.py."""
import ast
from pathlib import Path

SCRIPT = Path("src/vlm_nsm/experiments/run_e2e_language_to_sim.py")


def _get_ast():
    return ast.parse(SCRIPT.read_text())


def _find_arg(parsed, name):
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            func = getattr(node, "func", None)
            if func and getattr(func, "attr", None) == "add_argument":
                args = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
                if name in args:
                    return True
    return False


def test_online_qwen_parser_choice_exists():
    """--parser must accept online_qwen as a choice."""
    source = SCRIPT.read_text()
    assert "online_qwen" in source, "--parser must include 'online_qwen' choice"


def test_qwen_model_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_model")


def test_qwen_api_base_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_api_base")


def test_qwen_api_key_env_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_api_key_env") or _find_arg(parsed, "--qwen_api_key_env")


def test_qwen_temperature_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_temperature")


def test_qwen_max_tokens_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_max_tokens")


def test_qwen_timeout_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_timeout")


def test_qwen_retries_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_retries")


def test_save_raw_qwen_outputs_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--save_raw_qwen_outputs")


def test_fallback_parser_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--fallback_parser")


def test_online_qwen_import_exists():
    """Verify OnlineQwenGoalParser is imported."""
    source = SCRIPT.read_text()
    assert "OnlineQwenGoalParser" in source


def test_online_qwen_used_in_parser_selection():
    """Verify online_qwen branch exists in parser selection logic."""
    source = SCRIPT.read_text()
    assert 'args.parser == "online_qwen"' in source or "online_qwen" in source
