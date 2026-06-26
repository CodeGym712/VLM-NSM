"""Tests for run_embodied_stacking_mock.py."""

from vlm_nsm.experiments.run_embodied_stacking_mock import run_mock_eval


def test_mock_eval_episode_success_rate():
    """Episode success rate should be >= 0.95 in mock environment."""
    summary = run_mock_eval(
        n_blocks=3,
        num_episodes=10,
        controller_type="scripted",
    )
    assert summary["episode_success_rate"] >= 0.95


def test_mock_eval_step_success_rate():
    """All steps should succeed in mock."""
    summary = run_mock_eval(
        n_blocks=3,
        num_episodes=5,
        controller_type="scripted",
    )
    assert summary["step_success_rate"] == 1.0


def test_mock_eval_goal_success_rate():
    """Goal should always be satisfied in mock."""
    summary = run_mock_eval(
        n_blocks=3,
        num_episodes=5,
        controller_type="scripted",
    )
    assert summary["goal_success_rate"] == 1.0


def test_mock_eval_reality_gap():
    """Reality gap should be 0 in mock environment (no sim-to-real gap)."""
    summary = run_mock_eval(
        n_blocks=3,
        num_episodes=10,
        controller_type="scripted",
    )
    # In mock mode, symbolic = physical success, so reality gap = 0
    assert summary["reality_gap"] == 0.0


def test_mock_eval_failure_stages():
    """No failures should occur in mock mode."""
    summary = run_mock_eval(
        n_blocks=3,
        num_episodes=5,
        controller_type="scripted",
    )
    assert summary["failure_stage_distribution"] == {}


def test_mock_eval_predicate_match():
    """Predicate match rate should be high in mock."""
    summary = run_mock_eval(
        n_blocks=3,
        num_episodes=10,
        controller_type="scripted",
    )
    assert summary["predicate_match_rate"] >= 0.9
