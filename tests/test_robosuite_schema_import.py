"""Regression tests for torch-free robosuite schema helpers."""

import numpy as np

from vlm_nsm.learning.robosuite_schema import ACTION_DIM, OBS_DIM, build_robosuite_obs_vector


def test_schema_constants():
    assert OBS_DIM == 11
    assert ACTION_DIM == 7


def test_build_obs_vector_without_torch_dependency():
    eef = {"x": 0.5, "y": 0.1, "z": 0.05}
    gripper = {"open": -0.5, "has_object": True}
    poses = {
        "cube1": {"x": 0.4, "y": 0.2, "z": 0.03},
        "cube2": {"x": 0.3, "y": 0.3, "z": 0.8},
    }
    obs = build_robosuite_obs_vector(eef, gripper, poses)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
