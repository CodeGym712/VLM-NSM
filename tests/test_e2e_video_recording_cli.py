"""Verify --record_video CLI arguments and result fields."""
import ast
import re
from pathlib import Path

SCRIPT = Path("src/vlm_nsm/experiments/run_e2e_language_to_sim.py")


def _get_ast():
    return ast.parse(SCRIPT.read_text())


def _find_arg(parsed, name):
    """Return True if argparse argument with flag `name` is defined."""
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            func = getattr(node, "func", None)
            if func and getattr(func, "attr", None) == "add_argument":
                args = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
                if name in args:
                    return True
    return False


def test_record_video_arg_exists():
    parsed = _get_ast()
    assert _find_arg(parsed, "--record_video"), "--record_video CLI arg not found"


def test_video_dir_arg_exists():
    parsed = _get_ast()
    assert _find_arg(parsed, "--video_dir"), "--video_dir CLI arg not found"


def test_video_fps_arg_exists():
    parsed = _get_ast()
    assert _find_arg(parsed, "--video_fps"), "--video_fps CLI arg not found"


def test_video_max_episodes_arg_exists():
    parsed = _get_ast()
    assert _find_arg(parsed, "--video_max_episodes"), "--video_max_episodes CLI arg not found"


def test_result_dict_has_video_fields():
    """run_demo return dict includes video fields with correct defaults."""
    source = SCRIPT.read_text()
    # Check return dict has video_recorded, video_path, video_error fields
    assert '"video_recorded": False' in source or "'video_recorded': False" in source
    assert '"video_path": None' in source or "'video_path': None" in source
    assert '"video_error": None' in source or "'video_error': None" in source


def test_video_filename_format_defined():
    """Verify the video filename pattern is used."""
    source = SCRIPT.read_text()
    assert "e2e_" in source and "v0_1_1.mp4" in source, "Video filename format not found"
    assert "_outcome_" in source or "outcome" in source, "Video filename includes outcome"


def test_video_recording_does_not_fail_episode():
    """Verify video errors are caught and don't crash the episode."""
    source = SCRIPT.read_text()
    # Look for try/except or graceful handling around video
    assert "video_error" in source
    assert "_write_video_safe" in source, "Expected _write_video_safe helper for graceful failure"
    assert "cv2.VideoWriter" in source, "Expected OpenCV VideoWriter usage"
