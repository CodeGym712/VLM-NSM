"""Tests for Learned Inverse Dynamics Model."""

import numpy as np
import torch

from vlm_nsm.embodied.controllers.learned_id_model import (
    LearnedIDModel,
    IDModelEnsemble,
    IDPrediction,
    hybrid_pid_learned_action,
)


# ── Model Architecture ───────────────────────────────────────────────────────

def test_model_creation():
    model = LearnedIDModel(input_dim=28, output_dim=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert 20_000 < n_params < 35_000  # ~28K params with 28-dim input


def test_model_forward_shape():
    model = LearnedIDModel()
    x = torch.randn(8, 28)
    y = model(x)
    assert y.shape == (8, 4)


def test_model_predict_numpy_single():
    model = LearnedIDModel()
    model.eval()
    x = np.random.randn(28).astype(np.float32)
    y = model.predict_numpy(x)
    assert y.shape == (4,)
    assert y.dtype == np.float32


def test_model_predict_numpy_batch():
    model = LearnedIDModel()
    model.eval()
    x = np.random.randn(32, 28).astype(np.float32)
    y = model.predict_numpy(x)
    assert y.shape == (32, 4)


# ── Ensemble ─────────────────────────────────────────────────────────────────

def test_ensemble_empty():
    ensemble = IDModelEnsemble()
    pred = ensemble.predict(np.random.randn(28).astype(np.float32))
    assert isinstance(pred, IDPrediction)
    assert pred.confidence == 0.0  # no models loaded


def test_ensemble_from_checkpoints(tmp_path):
    """Test ensemble loading from saved checkpoints."""
    paths = []
    for i in range(3):
        model = LearnedIDModel(input_dim=28, output_dim=4)
        # Use different random init for diversity
        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_dim": 28,
            "output_dim": 4,
            "action_mean": [0.0, 0.0, 0.0, 0.0],
            "action_std": [1.0, 1.0, 1.0, 1.0],
        }
        p = tmp_path / f"model_{i}.pt"
        torch.save(ckpt, p)
        paths.append(str(p))

    ensemble = IDModelEnsemble(model_paths=paths)
    assert len(ensemble.models) == 3

    x = np.zeros(28, dtype=np.float32)
    # Set phase=reach_pick (phase 0)
    x[13] = 1.0
    pred = ensemble.predict(x)
    assert pred.action.shape == (4,)
    assert pred.velocity.shape == (3,)
    assert 0.0 <= pred.confidence <= 1.0


def test_ensemble_confidence(tmp_path):
    """Confidence should be high when models agree, low when they disagree."""
    # Create 3 identical models → high confidence
    paths = []
    for i in range(3):
        model = LearnedIDModel(input_dim=28, output_dim=4)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_dim": 28,
            "output_dim": 4,
            "action_mean": [0.0, 0.0, 0.01, 0.0],
            "action_std": [1.0, 1.0, 1.0, 1.0],
        }
        p = tmp_path / f"model_{i}.pt"
        torch.save(ckpt, p)
        paths.append(str(p))

    ensemble = IDModelEnsemble(model_paths=paths)
    x = np.zeros(28, dtype=np.float32)
    x[13] = 1.0  # reach_pick
    pred = ensemble.predict(x)
    assert pred.confidence > 0.0


# ── Hybrid Fusion ────────────────────────────────────────────────────────────

def test_hybrid_low_confidence_fallback():
    """When confidence < threshold, use pure PID."""
    pid_action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    learned_pred = IDPrediction(
        action=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        gripper=1.0,
        confidence=0.3,  # below threshold
        std=np.ones(4, dtype=np.float32),
    )
    action, source = hybrid_pid_learned_action(
        pid_action, learned_pred, pid_weight=0.3, confidence_threshold=0.7)
    np.testing.assert_array_equal(action, pid_action)
    assert source == "pid_fallback"


def test_hybrid_high_confidence_blend():
    """When confidence >= threshold, blend PID + Learned."""
    pid_action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    learned_pred = IDPrediction(
        action=np.array([0.2, 0.01, -0.01, -1.0], dtype=np.float32),
        velocity=np.array([0.2, 0.01, -0.01], dtype=np.float32),
        gripper=-1.0,
        confidence=0.85,  # above threshold
        std=np.zeros(4, dtype=np.float32),
    )
    action, source = hybrid_pid_learned_action(
        pid_action, learned_pred, pid_weight=0.3, confidence_threshold=0.7)

    # action = 0.3 * pid + 0.7 * learned
    expected = 0.3 * pid_action + 0.7 * np.array(
        [0.2, 0.01, -0.01, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(action, expected)
    assert source == "hybrid"


def test_hybrid_no_learned():
    """When ensemble is empty, use pure PID."""
    pid_action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    learned_pred = IDPrediction(
        action=np.zeros(4, dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        gripper=-1.0,
        confidence=0.0,
        std=np.ones(4, dtype=np.float32),
    )
    action, source = hybrid_pid_learned_action(
        pid_action, learned_pred, pid_weight=0.3, confidence_threshold=0.7)
    np.testing.assert_array_equal(action, pid_action)
    assert source == "pid_fallback"
