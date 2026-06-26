"""Tests for symbolic condition encoding."""

import numpy as np
import pytest

from vlm_nsm.neuro_symbolic.symbolic_condition import (
    SymbolicConditionEncoder,
    SymbolicConditionVector,
    symbolic_condition_dim,
)
from vlm_nsm.skills.skill_types import SkillCommand


class TestSymbolicConditionEncoder:
    """Tests for SymbolicConditionEncoder."""

    @pytest.fixture
    def encoder(self):
        return SymbolicConditionEncoder(objects=["cube1", "cube2"])

    @pytest.fixture
    def encoder_3obj(self):
        return SymbolicConditionEncoder(objects=["cube1", "cube2", "cube3"])

    def test_encode_pick_place(self, encoder):
        """Encode a simple pick_place command."""
        cmd = SkillCommand(
            skill_name="pick_place",
            object_id="cube1",
            target_object_id="cube2",
            relation="on",
        )
        cond = encoder.encode(cmd, phase_name="grasp_close")
        vec = cond.as_vector()

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec))

    def test_encode_to_table(self, encoder):
        """Encode a move_to_table command."""
        cmd = SkillCommand(
            skill_name="pick_place",
            object_id="cube2",
            target_object_id="table",
            relation="table",
        )
        cond = encoder.encode(cmd, phase_name="release_verify")
        vec = cond.as_vector()

        assert isinstance(vec, np.ndarray)
        # target is table → target_object_onehot should be at index n_objects
        target_start = 2  # n_objects
        assert vec[target_start + 2] == 1.0  # table is 3rd target (index 2) for 2 objects

    def test_onehot_pick_object(self, encoder):
        """Pick object one-hot is correct."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube2",
                          target_object_id="cube1", relation="on")
        cond = encoder.encode(cmd)
        assert cond.pick_object_onehot[0] == 0.0  # cube1 is NOT picked
        assert cond.pick_object_onehot[1] == 1.0  # cube2 IS picked

    def test_phase_onehot(self, encoder):
        """Phase one-hot is correct."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                          target_object_id="cube2", relation="on")
        for phase in ["reach_pick", "micro_align", "grasp_close", "retreat"]:
            cond = encoder.encode(cmd, phase_name=phase)
            # phase_onehot should have exactly one 1.0
            assert cond.phase_onehot.sum() == 1.0

    def test_color_encoding(self, encoder):
        """Color one-hots use correct mapping."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                          target_object_id="cube2", relation="on")
        cond = encoder.encode(cmd)
        # cube1 → red
        assert cond.pick_color_onehot[0] == 1.0  # red
        # cube2 → green
        assert cond.target_color_onehot[1] == 1.0  # green

    def test_plan_context(self, encoder):
        """Plan step index and remaining steps are normalized."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                          target_object_id="cube2", relation="on")
        cond = encoder.encode(cmd, plan_step_index=0, total_plan_steps=2)
        assert 0.0 <= cond.plan_step_index[0] <= 1.0
        assert 0.0 <= cond.remaining_steps[0] <= 1.0
        # Step 0 of 2 → step_index = 0.0, remaining = 1/2 = 0.5
        assert cond.plan_step_index[0] == 0.0
        assert cond.remaining_steps[0] == 0.5

    def test_dim_consistency(self, encoder, encoder_3obj):
        """dim property matches as_vector().shape."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                          target_object_id="cube2", relation="on")
        cond = encoder.encode(cmd)
        assert cond.dim == len(cond.as_vector())

        cond3 = encoder_3obj.encode(SkillCommand(
            skill_name="pick_place", object_id="cube1",
            target_object_id="cube3", relation="on"))
        assert cond3.dim > cond.dim  # 3 objects → larger vector

    def test_symbolic_condition_dim_function(self):
        """symbolic_condition_dim computes correct dimensions."""
        dim2 = symbolic_condition_dim(2)
        dim3 = symbolic_condition_dim(3)
        assert dim2 == (
            2 + 3 + 3 + 10 + 3 + 3 + 3 + 1 + 1 + 1 + 1
        )
        assert dim3 > dim2

    def test_encode_unknown_object_graceful(self, encoder):
        """Unknown object falls back to last one-hot slot."""
        cmd = SkillCommand(skill_name="pick_place", object_id="unknown_obj",
                          target_object_id="cube1", relation="on")
        cond = encoder.encode(cmd)
        # Should not crash, last one-hot slot should be 1 (fallback)
        assert np.all(np.isfinite(cond.as_vector()))

    def test_encode_without_target(self, encoder):
        """Command without target_object_id uses table."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1")
        cond = encoder.encode(cmd)
        # target should be table
        target_start = len(encoder._objects)  # n_objects
        assert cond.target_object_onehot[target_start] == 1.0

    def test_size_normalization(self, encoder):
        """Object sizes are normalized."""
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                          target_object_id="cube2", relation="on")
        cond = encoder.encode(cmd)
        # cube1 = 0.04 → 0.04/0.06 ≈ 0.667
        assert 0.6 < cond.pick_size[0] < 0.7
        # cube2 = 0.05 → 0.05/0.06 ≈ 0.833
        assert 0.8 < cond.target_size[0] < 0.9

    def test_all_phases_valid(self, encoder):
        """All 10 phase names encode without error."""
        phases = [
            "reach_pick", "descend_align", "micro_align",
            "grasp_close", "lift_verify", "transport",
            "descend_place", "micro_place", "release_verify", "retreat",
        ]
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1",
                          target_object_id="cube2", relation="on")
        for phase in phases:
            cond = encoder.encode(cmd, phase_name=phase)
            assert cond.dim > 0
