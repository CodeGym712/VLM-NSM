"""Tests for Transition State Machine."""

from vlm_nsm.embodied.transition_state_machine import (
    TransitionStateMachine,
    PhaseDefinition,
    Substate,
    _default_phases_2block,
)


# ── Phase Definitions ────────────────────────────────────────────────────────

def test_default_phases_have_all_six():
    phases = _default_phases_2block()
    expected = {"reach_pick", "grasp", "lift", "reach_place", "release", "retreat"}
    assert set(phases.keys()) == expected


def test_phase_reach_pick_target():
    phase = PhaseDefinition(
        name="reach_pick",
        desired_substate=Substate.EEF_ABOVE_PICK,
        approach_height=0.10,
    )
    target = phase.get_target({"x": -0.05, "y": 0.0, "z": 0.85})
    assert target["x"] == -0.05
    assert target["y"] == 0.0
    assert abs(target["z"] - 0.95) < 0.01  # obj_z + 0.10


def test_phase_grasp_target():
    phase = PhaseDefinition(
        name="grasp",
        desired_substate=Substate.OBJECT_IN_GRIPPER,
        approach_height=0.005,
    )
    target = phase.get_target({"x": 0.1, "y": 0.2, "z": 0.85})
    assert abs(target["z"] - 0.855) < 0.01  # obj_z + 0.005


def test_phase_place_target():
    phase = PhaseDefinition(
        name="reach_place",
        desired_substate=Substate.EEF_ABOVE_TARGET,
        approach_height=0.15,
    )
    target = phase.get_target(
        {"x": 0.0, "y": 0.0, "z": 0.85},
        target_pose={"x": -0.05, "y": 0.0, "z": 0.90},
    )
    assert target["x"] == -0.05
    assert abs(target["z"] - 1.05) < 0.01  # target_z + 0.15


# ── Transition State Machine ─────────────────────────────────────────────────

def test_tsm_creation():
    tsm = TransitionStateMachine(n_objects=2)
    assert tsm.current_phase == "reach_pick"
    assert len(tsm.phases) == 6


def test_tsm_get_target():
    tsm = TransitionStateMachine()
    target = tsm.get_target("reach_pick", {"x": 0.0, "y": 0.0, "z": 0.85})
    assert "x" in target
    assert "y" in target
    assert "z" in target


def test_tsm_get_target_unknown_phase():
    tsm = TransitionStateMachine()
    target = tsm.get_target("nonexistent", {"x": 0.0, "y": 0.0, "z": 0.85})
    assert target == {"x": 0.0, "y": 0.0, "z": 1.0}  # safe default


def test_tsm_pid_overrides():
    """PID gains now managed solely by GainSchedule. get_pid_overrides returns {}."""
    tsm = TransitionStateMachine()
    overrides = tsm.get_pid_overrides("reach_pick")
    assert overrides == {}  # Gains from GainSchedule, not PhaseDefinition


def test_tsm_learned_id_enabled():
    tsm = TransitionStateMachine()
    assert tsm.should_use_learned_id("grasp") is True
    assert tsm.should_use_learned_id("reach_place") is True
    assert tsm.should_use_learned_id("reach_pick") is False
    assert tsm.should_use_learned_id("release") is False


def test_tsm_holding_check():
    tsm = TransitionStateMachine()
    assert tsm.should_check_holding("reach_place") is True
    assert tsm.should_check_holding("reach_pick") is False


def test_tsm_transition():
    tsm = TransitionStateMachine()
    assert tsm.current_phase == "reach_pick"

    tsm.transition("grasp")
    assert tsm.current_phase == "grasp"
    assert tsm.phase_history == ["reach_pick"]

    tsm.transition("lift")
    assert tsm.current_phase == "lift"
    assert tsm.phase_history == ["reach_pick", "grasp"]


def test_tsm_reset():
    tsm = TransitionStateMachine()
    tsm.transition("grasp")
    tsm.transition("lift")
    tsm.reset()
    assert tsm.current_phase == "reach_pick"
    assert tsm.phase_history == []


def test_tsm_custom_phases():
    custom = {
        "approach": PhaseDefinition(
            name="approach",
            desired_substate=Substate.EEF_ABOVE_PICK,
            approach_height=0.12,
        ),
    }
    tsm = TransitionStateMachine(phases=custom)
    assert len(tsm.phases) == 1
    assert tsm.get_phase_def("approach").approach_height == 0.12
