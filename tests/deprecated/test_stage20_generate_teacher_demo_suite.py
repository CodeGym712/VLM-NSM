import json

from vlm_nsm.experiments.stage20_generate_teacher_demo_suite import (
    generate_teacher_demo_suite,
    main,
)


def test_generate_teacher_demo_suite_count_and_fields():
    suite = generate_teacher_demo_suite(
        goals=["red_on_green", "green_on_red"],
        placement_bins=["center", "front"],
        seeds_per_bin=3,
        scripted_profile="cube2_robust",
    )

    assert len(suite["episodes"]) == 12
    first = suite["episodes"][0]
    assert first["goal_id"] == "red_on_green"
    assert first["direction"] == "cube1_on_cube2"
    assert first["placement_bin"] == "center"
    assert first["scripted_profile"] == "cube2_robust"
    assert first["expected_target_stack"] == ["cube2", "cube1"]
    assert first["parser_output"]["goal_predicates"][0] == ["on", "cube1", "cube2"]


def test_generate_teacher_demo_suite_has_both_goals():
    suite = generate_teacher_demo_suite(seeds_per_bin=1)
    goals = {ep["goal_id"] for ep in suite["episodes"]}

    assert goals == {"red_on_green", "green_on_red"}


def test_generate_teacher_demo_suite_deterministic_ordering():
    a = generate_teacher_demo_suite(
        goals=["red_on_green"],
        placement_bins=["center", "left"],
        seeds_per_bin=2,
    )
    b = generate_teacher_demo_suite(
        goals=["red_on_green"],
        placement_bins=["center", "left"],
        seeds_per_bin=2,
    )

    assert [ep["episode_id"] for ep in a["episodes"]] == [
        "stage20_red_on_green_center_s0",
        "stage20_red_on_green_center_s1",
        "stage20_red_on_green_left_s0",
        "stage20_red_on_green_left_s1",
    ]
    assert a == b


def test_generate_teacher_demo_suite_cli_writes_json(tmp_path, monkeypatch):
    out = tmp_path / "suite.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--output", str(out),
            "--seeds_per_bin", "2",
            "--placement_bins", "center,front",
            "--goals", "green_on_red",
            "--scripted_profile", "cube2_robust",
        ],
    )

    main()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["episodes"]) == 4
    assert {ep["goal_id"] for ep in data["episodes"]} == {"green_on_red"}
    assert all(ep["scripted_profile"] == "cube2_robust" for ep in data["episodes"])
