import json

import numpy as np

from vlm_nsm.experiments.collect_skill_demonstrations import (
    add_skill_fields_to_trajectory,
    build_manifest_record,
    default_skill_command,
    episode_kwargs_from_config,
    save_trajectory,
    write_dataset_meta,
    write_manifest,
)
from vlm_nsm.skills.skill_types import SkillExecutionResult


def test_default_skill_command_two_block_stack():
    command = default_skill_command("two_block_stack")

    assert command.skill_name == "pick_place"
    assert command.object_id == "cube1"
    assert command.target_object_id == "cube2"
    assert command.source == "table"
    assert command.relation == "on"


def test_episode_kwargs_from_config_filters_supported_keys():
    cfg = {
        "kp": 0.04,
        "max_delta": 0.02,
        "horizon": 1000,
        "unknown_key": "ignored",
    }

    kwargs = episode_kwargs_from_config(cfg)

    assert kwargs == {"kp": 0.04, "max_delta": 0.02, "horizon": 1000}


def test_add_skill_fields_to_trajectory():
    command = default_skill_command()
    traj = {
        "obs": np.zeros((3, 11), dtype=np.float32),
        "action": np.zeros((3, 7), dtype=np.float32),
        "phase_id": np.array([0, 1, 2], dtype=np.int32),
    }

    labelled = add_skill_fields_to_trajectory(traj, command, episode_index=4, success=True)

    assert labelled["skill_name"].tolist() == ["pick_place"] * 3
    assert labelled["skill_object_id"].tolist() == ["cube1"] * 3
    assert labelled["skill_target_object_id"].tolist() == ["cube2"] * 3
    assert labelled["skill_demo_episode_index"].tolist() == [4, 4, 4]
    assert labelled["skill_demo_success"].tolist() == [True, True, True]


def test_save_trajectory_round_trip(tmp_path):
    command = default_skill_command()
    traj = {
        "obs": np.ones((2, 11), dtype=np.float32),
        "action": np.ones((2, 7), dtype=np.float32),
    }

    out = save_trajectory(tmp_path / "trajectories" / "ep_000001.npz", traj, command, 0, True)

    assert out is not None
    loaded = np.load(out)
    assert loaded["obs"].shape == (2, 11)
    assert loaded["action"].shape == (2, 7)
    assert loaded["skill_name"].tolist() == ["pick_place", "pick_place"]


def test_build_manifest_record_json_safe():
    command = default_skill_command()
    result = SkillExecutionResult(
        success=True,
        skill_command=command,
        steps=123,
        metrics={"raw_result": {"too": "large"}, "executor": "ScriptedSkillExecutor"},
    )

    record = build_manifest_record(
        episode_id="ep_000001",
        command=command,
        result=result,
        trajectory_path="trajectories/ep_000001.npz",
        seed=7,
        initial_scene={"x": np.float32(1.0), "arr": np.array([1, 2])},
        final_scene={},
        teacher_mode="hybrid_reach_pick",
    )

    payload = json.dumps(record)
    assert "raw_result" not in payload
    assert record["episode_id"] == "ep_000001"
    assert record["trajectory_path"] == "trajectories/ep_000001.npz"
    assert record["initial_scene"]["x"] == 1.0
    assert record["initial_scene"]["arr"] == [1, 2]


def test_write_manifest_jsonl(tmp_path):
    command = default_skill_command()
    result = SkillExecutionResult(True, command, steps=1)
    records = [
        build_manifest_record(
            episode_id="ep_000001",
            command=command,
            result=result,
            trajectory_path="trajectories/ep_000001.npz",
            seed=0,
        )
    ]

    path = tmp_path / "manifest.jsonl"
    write_manifest(records, path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["skill_name"] == "pick_place"


def test_write_dataset_meta(tmp_path):
    records = [
        {"success": True},
        {"success": False},
    ]

    meta = write_dataset_meta(
        tmp_path,
        records,
        task="two_block_stack",
        teacher_mode="hybrid_reach_pick",
        seed=0,
        config_path="config.json",
        checkpoint_path="best.pt",
        record_rgb=False,
    )

    assert meta["num_episodes"] == 2
    assert meta["success_count"] == 1
    assert meta["success_rate"] == 0.5
    assert meta["record_rgb"] is False
    assert (tmp_path / "dataset_meta.json").exists()


def test_hybrid_policy_adds_phase_when_checkpoint_obs_dim_requires_it(monkeypatch, tmp_path):
    import sys
    import types

    calls = {}

    class DummyModel:
        def eval(self):
            return None

    fake_eval_bc = types.ModuleType("vlm_nsm.learning.eval_bc")
    fake_eval_bc.load_checkpoint = lambda checkpoint, device: (
        DummyModel(),
        {"obs_dim": 17, "action_dim": 7, "condition_on_phase": False},
    )

    fake_eval_bc_hybrid = types.ModuleType("vlm_nsm.learning.eval_bc_hybrid")

    def fake_get_bc_action(model, obs, device, action_mean, action_std, fixed_zero_dims):
        calls["obs_shape"] = obs.shape
        return np.zeros(7, dtype=np.float32)

    fake_eval_bc_hybrid._get_bc_action = fake_get_bc_action

    monkeypatch.setitem(sys.modules, "vlm_nsm.learning.eval_bc", fake_eval_bc)
    monkeypatch.setitem(sys.modules, "vlm_nsm.learning.eval_bc_hybrid", fake_eval_bc_hybrid)

    from vlm_nsm.experiments.collect_skill_demonstrations import _load_hybrid_policy

    norm_path = tmp_path / "norm.json"
    norm_path.write_text(json.dumps({"action_mean": [0] * 7, "action_std": [1] * 7}))

    context = _load_hybrid_policy("dummy.pt", str(norm_path), "cpu", ["reach_pick"])
    context["policy_override"](np.zeros(11, dtype=np.float32), 0)

    assert calls["obs_shape"] == (17,)


def test_collect_skill_demonstrations_cli_args_present():
    import ast
    from pathlib import Path

    source = Path("src/vlm_nsm/experiments/collect_skill_demonstrations.py").read_text()
    parsed = ast.parse(source)
    args = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call) and getattr(getattr(node, "func", None), "attr", None) == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    args.add(arg.value)

    for expected in [
        "--output_dir",
        "--num_episodes",
        "--task",
        "--executor",
        "--seed",
        "--max_steps",
        "--save_trajectories",
        "--record_rgb",
    ]:
        assert expected in args
