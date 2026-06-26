from vlm_nsm.experiments.stage19_run_symbolic_rollouts import resolve_scripted_profile


def test_default_profile_is_zero_offsets():
    cfg = resolve_scripted_profile("default", "cube2")

    assert cfg["scripted_profile"] == "default"
    assert cfg["grasp_z_offset"] == 0.0
    assert cfg["place_z_offset"] == 0.0
    assert cfg["release_z_offset"] == 0.0
    assert cfg["stabilization_steps"] == 0
    assert cfg["profile_applied_to_object"] is None


def test_cube2_grasp_profile_only_applies_to_cube2():
    cube2_cfg = resolve_scripted_profile("cube2_grasp_lower", "cube2")
    cube1_cfg = resolve_scripted_profile("cube2_grasp_lower", "cube1")

    assert cube2_cfg["grasp_z_offset"] < 0
    assert cube2_cfg["profile_applied_to_object"] == "cube2"
    assert cube1_cfg["grasp_z_offset"] == 0.0
    assert cube1_cfg["profile_applied_to_object"] is None


def test_green_cube_alias_gets_profile():
    cfg = resolve_scripted_profile("cube2_place_higher", "green_cube")

    assert cfg["place_z_offset"] > 0
    assert cfg["release_z_offset"] > 0
    assert cfg["stabilization_steps"] > 0
    assert cfg["profile_applied_to_object"] == "green_cube"


def test_robust_combines_grasp_and_place_offsets():
    cfg = resolve_scripted_profile("cube2_robust", "cube2")

    assert cfg["grasp_z_offset"] < 0
    assert cfg["place_z_offset"] > 0
    assert cfg["release_z_offset"] > 0
    assert cfg["stabilization_steps"] > 0


def test_unknown_profile_rejected():
    try:
        resolve_scripted_profile("bad_profile", "cube2")
    except ValueError as exc:
        assert "bad_profile" in str(exc)
    else:
        raise AssertionError("expected unknown profile to fail")
