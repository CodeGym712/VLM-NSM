"""Verify eval_online_qwen_parser.py exists and has expected CLI args."""
import ast
from pathlib import Path

SCRIPT = Path("src/vlm_nsm/experiments/eval_online_qwen_parser.py")


def test_script_exists():
    assert SCRIPT.exists()


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


def test_instructions_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--instructions"), "Missing --instructions arg"


def test_output_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--output"), "Missing --output arg"


def test_model_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--model"), "Missing --model arg"


def test_api_base_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--api_base") or _find_arg(parsed, "--api-base"), "Missing --api_base arg"


def test_qwen_api_key_env_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--qwen_api_key_env") or _find_arg(parsed, "--qwen-api-key-env"), "Missing --qwen_api_key_env arg"


def test_temperature_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--temperature"), "Missing --temperature arg"


def test_max_tokens_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--max_tokens") or _find_arg(parsed, "--max-tokens"), "Missing --max_tokens arg"


def test_timeout_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--timeout"), "Missing --timeout arg"


def test_retries_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--retries"), "Missing --retries arg"


def test_available_objects_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--available_objects") or _find_arg(parsed, "--available-objects"), "Missing --available_objects arg"


def test_save_raw_qwen_outputs_arg():
    parsed = _get_ast()
    assert _find_arg(parsed, "--save_raw_qwen_outputs") or _find_arg(parsed, "--save-raw-qwen-outputs"), "Missing --save_raw_qwen_outputs arg"


def test_load_instructions_function_exists():
    """Verify the load_instructions helper is defined."""
    source = SCRIPT.read_text()
    assert "def load_instructions" in source


def test_main_function_exists():
    source = SCRIPT.read_text()
    assert "def main" in source


def test_online_qwen_parser_imported():
    source = SCRIPT.read_text()
    assert "OnlineQwenGoalParser" in source
