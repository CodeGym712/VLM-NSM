"""Tests for ReplayController."""

import numpy as np
import pytest

from vlm_nsm.embodied.controllers.replay_controller import ReplayController


def test_replay_load_and_act():
    controller = ReplayController()
    actions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    controller.load(actions)
    assert controller.remaining == 3
    obs = np.zeros(4, dtype=np.float32)
    a1 = controller.act("reach_pick", obs)
    assert np.allclose(a1, [0.1, 0.2])
    assert controller.remaining == 2
    a2 = controller.act("grasp", obs)
    assert np.allclose(a2, [0.3, 0.4])
    assert controller.remaining == 1


def test_replay_exhaustion():
    controller = ReplayController()
    actions = np.array([[1.0, 2.0]], dtype=np.float32)
    controller.load(actions)
    obs = np.zeros(4, dtype=np.float32)
    controller.act("reach_pick", obs)
    with pytest.raises(RuntimeError, match="Replay sequence exhausted"):
        controller.act("grasp", obs)


def test_replay_no_actions_loaded():
    controller = ReplayController()
    obs = np.zeros(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match="No actions loaded"):
        controller.act("reach_pick", obs)


def test_replay_reset():
    controller = ReplayController()
    actions = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    controller.load(actions)
    obs = np.zeros(4, dtype=np.float32)
    controller.act("reach_pick", obs)
    assert controller.remaining == 1
    controller.reset()
    assert controller.remaining == 2


def test_replay_success():
    controller = ReplayController()
    actions = np.array([[1.0, 2.0]], dtype=np.float32)
    controller.load(actions)
    obs = np.zeros(4, dtype=np.float32)
    controller.act("reach_pick", obs)
    assert controller.replay_success is True  # All actions consumed


def test_controller_type():
    controller = ReplayController()
    assert controller.controller_type == "replay"
