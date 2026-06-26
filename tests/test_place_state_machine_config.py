"""Tests for PlaceConfig validation (no robosuite dependency)."""
from __future__ import annotations

from vlm_nsm.embodied.place_state_machine import PlaceConfig


def test_place_config_defaults():
    cfg = PlaceConfig()
    assert cfg.kp == 0.05
    assert cfg.max_delta == 0.02
    assert cfg.preplace_height == 0.15
    assert cfg.descend_offset == 0.02
    assert cfg.xy_align_threshold == 0.01
    assert cfg.release_steps == 30
    assert cfg.hold_steps == 10
    assert cfg.retreat_height == 0.15
    assert cfg.step_limit == 200
    assert cfg.max_retries == 2
    assert cfg.reach_threshold == 0.02
    assert cfg.xy_micro_align_steps == 30


def test_place_config_custom():
    cfg = PlaceConfig(kp=0.03, preplace_height=0.20, release_steps=40, max_retries=3)
    assert cfg.kp == 0.03
    assert cfg.preplace_height == 0.20
    assert cfg.release_steps == 40
    assert cfg.max_retries == 3
    # Defaults unchanged
    assert cfg.descend_offset == 0.02
    assert cfg.hold_steps == 10


def test_place_config_all_params():
    cfg = PlaceConfig(
        kp=0.04, max_delta=0.01, preplace_height=0.12, descend_offset=0.015,
        xy_align_threshold=0.008, release_steps=40, hold_steps=15,
        retreat_height=0.12, step_limit=150, max_retries=3, reach_threshold=0.015,
        xy_micro_align_steps=40, xy_align_kp=0.025,
    )
    assert cfg.kp == 0.04
    assert cfg.xy_micro_align_steps == 40
    assert cfg.xy_align_kp == 0.025
