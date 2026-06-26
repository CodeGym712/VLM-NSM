"""Stage 20-U6: Tests for Stage20LearnedExecutor wrapper."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from vlm_nsm.embodied.stage20_learned_executor import Stage20LearnedExecutor


class TestLearnedExecutorWrapper:
    def test_init_loads_checkpoint(self):
        """Executor loads the Stage 20-S checkpoint successfully."""
        ckpt = "generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_holding_aux_stage20_s/best.pt"
        ds = "generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled_holding_obs4_stage20_s"
        if not Path(ckpt).exists():
            pytest.skip("Checkpoint not found")
        ex = Stage20LearnedExecutor(ckpt, ds, device="cpu")
        assert ex.input_dim == 110
        assert ex.action_dim == 7
        assert ex.has_holding_head is True

    def test_predict_returns_finite_action(self):
        """predict() returns finite 7-dim action."""
        ckpt = "generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_holding_aux_stage20_s/best.pt"
        ds = "generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled_holding_obs4_stage20_s"
        if not Path(ckpt).exists():
            pytest.skip("Checkpoint not found")
        ex = Stage20LearnedExecutor(ckpt, ds, device="cpu")
        obs = np.array([0.5, 0.0, 0.8, 0.05, 0.0, 0.06, 0.03, 0.83, -0.04, 0.0, 0.835], dtype=np.float32)
        # Need multiple steps to warm up history
        for _ in range(4):
            result = ex.predict(obs, phase_id=0)
        assert result["finite"] is True
        assert len(result["action"]) == 7
        assert "holding_logit" in result

    def test_reset_clears_history(self):
        """reset() clears internal buffers."""
        ckpt = "generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_holding_aux_stage20_s/best.pt"
        ds = "generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled_holding_obs4_stage20_s"
        if not Path(ckpt).exists():
            pytest.skip("Checkpoint not found")
        ex = Stage20LearnedExecutor(ckpt, ds, device="cpu")
        obs = np.zeros(11, dtype=np.float32); obs[3] = 0.05
        ex.predict(obs, phase_id=0)
        ex.predict(obs, phase_id=0)
        assert len(ex._obs_history) == 2
        ex.reset()
        assert len(ex._obs_history) == 0

    def test_is_warm(self):
        """is_warm becomes True after history_len steps."""
        ckpt = "generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_holding_aux_stage20_s/best.pt"
        ds = "generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled_holding_obs4_stage20_s"
        if not Path(ckpt).exists():
            pytest.skip("Checkpoint not found")
        ex = Stage20LearnedExecutor(ckpt, ds, device="cpu")
        obs = np.zeros(11, dtype=np.float32); obs[3] = 0.05
        assert not ex.is_warm
        for _ in range(4):
            ex.predict(obs, phase_id=0)
        assert ex.is_warm

    def test_short_obs_raises(self):
        """obs with < 11 dims raises ValueError."""
        ckpt = "generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_holding_aux_stage20_s/best.pt"
        ds = "generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled_holding_obs4_stage20_s"
        if not Path(ckpt).exists():
            pytest.skip("Checkpoint not found")
        ex = Stage20LearnedExecutor(ckpt, ds, device="cpu")
        with pytest.raises(ValueError):
            ex.predict(np.zeros(3, dtype=np.float32), phase_id=0)
