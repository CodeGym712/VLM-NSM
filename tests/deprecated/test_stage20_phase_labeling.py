import numpy as np

from vlm_nsm.data.stage20_phase_labeling import phase_names, relabel_composite_place_phases


def test_relabel_composite_place_splits_release_and_retreat():
    phase = np.array([3, 3, 3, 3, 3, 3], dtype=np.int32)
    action = np.zeros((6, 7), dtype=np.float32)
    action[:2, 6] = 1.0
    action[2:4, 6] = -1.0
    action[4:, 6] = -1.0
    action[4:, 2] = 0.05

    out, stats = relabel_composite_place_phases(phase, action)

    assert out.tolist() == [3, 3, 4, 4, 5, 5]
    assert stats.applied
    assert stats.release_steps == 2
    assert stats.retreat_steps == 2


def test_relabel_preserves_existing_release_retreat_labels():
    phase = np.array([3, 4, 5], dtype=np.int32)
    action = np.zeros((3, 7), dtype=np.float32)
    action[:, 6] = -1.0
    action[:, 2] = 0.05

    out, stats = relabel_composite_place_phases(phase, action)

    assert out.tolist() == [5, 4, 5]
    assert stats.release_steps == 0
    assert stats.retreat_steps == 1


def test_phase_names_returns_unknown_for_invalid_id():
    names = phase_names(np.array([0, 4, 5, 99]))

    assert names.tolist() == ["reach_pick", "release", "retreat", "unknown"]
