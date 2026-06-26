"""Tests for SkillStepDecomposer."""

import pytest

from vlm_nsm.core.data_types import Action
from vlm_nsm.embodied.skill_step_decomposer import (
    SKILL_STEP_NAMES,
    SkillStepDecomposer,
)


def test_decompose_move_six_steps():
    decomposer = SkillStepDecomposer()
    action = Action(name="move", args=["cube1", "table", "cube2"])
    result = decomposer.decompose(action)
    assert result.step_count == 6
    assert result.steps == SKILL_STEP_NAMES
    assert result.action == action


def test_decompose_pick_three_steps():
    decomposer = SkillStepDecomposer()
    action = Action(name="pick", args=["cube1", "table"])
    result = decomposer.decompose(action)
    assert result.step_count == 3
    assert result.steps == ["reach_pick", "grasp", "lift"]


def test_decompose_place_three_steps():
    decomposer = SkillStepDecomposer()
    action = Action(name="place", args=["cube1", "cube2"])
    result = decomposer.decompose(action)
    assert result.step_count == 3
    assert result.steps == ["reach_place", "release", "retreat"]


def test_decompose_unsupported_action():
    decomposer = SkillStepDecomposer()
    action = Action(name="fly", args=["moon"])
    with pytest.raises(ValueError, match="Unsupported action type"):
        decomposer.decompose(action)


def test_decompose_step_order_preserved():
    """Verify the step order is reach_pick first, retreat last."""
    decomposer = SkillStepDecomposer()
    action = Action(name="move", args=["cube1", "table", "cube2"])
    result = decomposer.decompose(action)
    assert result.steps[0] == "reach_pick"
    assert result.steps[-1] == "retreat"
