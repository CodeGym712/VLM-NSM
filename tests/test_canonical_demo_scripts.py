"""Verify canonical demo/eval scripts exist and reference correct artifacts."""

from pathlib import Path

SCRIPTS = [
    "scripts/run_canonical_demo.sh",
    "scripts/run_canonical_eval_20ep.sh",
]


def _script_text(name):
    return Path(name).read_text()


def test_scripts_exist():
    for s in SCRIPTS:
        assert Path(s).exists(), f"{s} not found"


def test_scripts_executable():
    for s in SCRIPTS:
        assert Path(s).stat().st_mode & 0o111, f"{s} is not executable"


def test_demo_script_uses_e2e_module():
    text = _script_text(SCRIPTS[0])
    assert "run_e2e_language_to_sim" in text


def test_demo_script_uses_reach_pick():
    text = _script_text(SCRIPTS[0])
    assert "reach_pick" in text


def test_demo_script_has_canonical_checkpoint():
    text = _script_text(SCRIPTS[0])
    assert "best.pt" in text
    assert "stage7a12_final.json" in text


def test_eval_script_uses_hybrid_module():
    text = _script_text(SCRIPTS[1])
    assert "eval_bc_hybrid" in text


def test_eval_script_uses_reach_pick():
    text = _script_text(SCRIPTS[1])
    assert "reach_pick" in text


def test_eval_script_has_canonical_checkpoint():
    text = _script_text(SCRIPTS[1])
    assert "best.pt" in text
    assert "stage7a12_final.json" in text


def test_eval_script_has_fixed_zero_dims():
    text = _script_text(SCRIPTS[1])
    assert "3 4 5" in text or "fixed_zero_action_dims" in text
