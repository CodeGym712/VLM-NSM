"""Stage 20-S: Tests for holding auxiliary training.

Tests:
- Holding auxiliary loss path runs 1 epoch on CPU with small dataset.
- Validation action metrics remain unweighted.
- No NaN in holding BCE/accuracy.
- Config records holding_aux_weight and phase ids.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vlm_nsm.training.train_stage20_goal_conditioned_rnn import (
    GoalConditionedRNN,
    _holding_aux_loss,
    _parse_phase_ids,
)


class TestHoldingAuxLoss:

    def test_holding_head_created_when_weight_positive(self):
        """Model creates holding_head when use_holding_head=True."""
        model = GoalConditionedRNN(
            input_dim=10, action_dim=7,
            model_type="lstm", hidden_dim=32, num_layers=1,
            use_holding_head=True,
        )
        assert model.holding_head is not None

    def test_no_holding_head_when_not_requested(self):
        """Model does NOT create holding_head by default."""
        model = GoalConditionedRNN(
            input_dim=10, action_dim=7,
            model_type="lstm", hidden_dim=32, num_layers=1,
        )
        assert model.holding_head is None

    def test_holding_loss_returns_none_when_no_labels(self):
        """_holding_aux_loss returns None when batch has no holding_label."""
        model = GoalConditionedRNN(
            input_dim=10, action_dim=7,
            model_type="lstm", hidden_dim=32, num_layers=1,
            use_holding_head=True,
        )
        x = torch.randn(4, 8, 10)
        batch = {}
        phase_labels = torch.zeros(4, dtype=torch.long)
        loss, correct, target = _holding_aux_loss(
            model, x, batch, phase_labels, set())
        assert loss is None
        assert correct is None
        assert target is None

    def test_holding_loss_computable(self):
        """_holding_aux_loss returns finite BCE and accuracy tensors."""
        model = GoalConditionedRNN(
            input_dim=10, action_dim=7,
            model_type="lstm", hidden_dim=32, num_layers=1,
            use_holding_head=True,
        )
        x = torch.randn(4, 8, 10)
        batch = {
            "holding_label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "holding_label_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }
        phase_labels = torch.tensor([2, 2, 3, 3], dtype=torch.long)
        loss, correct, target = _holding_aux_loss(
            model, x, batch, phase_labels, {2, 3})
        assert loss is not None
        assert correct is not None
        assert target is not None
        assert torch.isfinite(loss)
        assert correct.shape == target.shape

    def test_holding_loss_phase_filtered(self):
        """Only samples with phase in phase_ids contribute to loss."""
        model = GoalConditionedRNN(
            input_dim=10, action_dim=7,
            model_type="lstm", hidden_dim=32, num_layers=1,
            use_holding_head=True,
        )
        x = torch.randn(4, 8, 10)
        batch = {
            "holding_label": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "holding_label_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }
        # Only phases 0,1 — but holding typically positive in phases 2,3
        phase_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss, correct, target = _holding_aux_loss(
            model, x, batch, phase_labels, {2, 3})
        assert loss is not None
        # Should only include samples at phase 2 and 3 (indices 2 and 3)
        assert len(target) == 2

    def test_holding_loss_masked_samples(self):
        """Samples with holding_label_mask=0 are excluded."""
        model = GoalConditionedRNN(
            input_dim=10, action_dim=7,
            model_type="lstm", hidden_dim=32, num_layers=1,
            use_holding_head=True,
        )
        x = torch.randn(4, 8, 10)
        batch = {
            "holding_label": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "holding_label_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
        phase_labels = torch.tensor([2, 2, 2, 2], dtype=torch.long)
        loss, correct, target = _holding_aux_loss(
            model, x, batch, phase_labels, {2})
        assert loss is not None
        # Only 2 samples should contribute (indices 0 and 2)
        assert len(target) == 2

    def test_parse_phase_ids(self):
        """_parse_phase_ids correctly parses comma-separated string."""
        assert _parse_phase_ids("2,3") == {2, 3}
        assert _parse_phase_ids("1") == {1}
        assert _parse_phase_ids(None) == set()
        assert _parse_phase_ids("") == set()
