"""Collect VLA demonstration data with RGB recording.

Runs the scripted teacher controller and records per-step:
  - agentview + frontview RGB images (128x128)
  - actions (dx, dy, dz, gripper) — 4-dim
  - proprioception (EEF position + gripper state)
  - phase_id, pick_object_id, target_object_id, relation
  - success/failure labels

Output: HDF5 per episode + manifest.jsonl + stats.json

Usage:
    MUJOCO_GL=egl PYTHON_BIN=nsm/bin/python \\
        python -m vlm_nsm.experiments.collect_vla_demos \\
        --output_dir generated/datasets/vla_demos/v0 \\
        --num_episodes 100 \\
        --task two_block_stack \\
        --executor scripted \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vlm_nsm.skills.skill_types import SkillCommand

# ── Constants ───────────────────────────────────────────────────────────────

_PHASE_NAME_MAP = {
    0: "reach_pick", 1: "grasp", 2: "lift",
    3: "reach_place", 4: "release", 5: "retreat",
    # Extended phases (from phase labeling Stage 20-K and micro_align)
    6: "descend_align", 7: "micro_align", 8: "grasp_close",
    9: "lift_verify", 10: "transport", 11: "descend_place",
    12: "micro_place", 13: "release_verify",
}

# Reverse mapping for phase_name → phase_id lookup
_PHASE_NAME_MAP_INV = {v: k for k, v in _PHASE_NAME_MAP.items()}

_RELATION_MAP = {"on": 0, "table": 1, "none": 2}
_OBJECT_ID_MAP = {"table": 0, "cube1": 1, "cube2": 2, "cube3": 3}

IMAGE_SIZE = 128
ACTION_DIM = 4  # dx, dy, dz, gripper


# ── RGB Recording Wrapper ──────────────────────────────────────────────────

class VLARecordingWrapper:
    """Wraps a Robosuite env to record RGB + proprio at each step.

    Intended for use inside the episode execution loop.
    """

    def __init__(self, env, camera_names=("agentview", "frontview"),
                 image_size=IMAGE_SIZE):
        self._env = env
        self._camera_names = list(camera_names)
        self._image_size = image_size

        # Buffers
        self.agentview_frames: list[np.ndarray] = []
        self.frontview_frames: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.eef_positions: list[np.ndarray] = []
        self.gripper_states: list[np.ndarray] = []
        self.phase_ids: list[int] = []
        self.phase_names: list[str] = []

    def record_step(self, action: np.ndarray, phase_id: int = 0,
                    phase_name: str = ""):
        """Record observation + action for the current step."""
        # Record RGB
        for cam_name in self._camera_names:
            img = self._env.render_rgb(cam_name, self._image_size, self._image_size)
            if cam_name == "agentview":
                self.agentview_frames.append(img)
            elif cam_name == "frontview":
                self.frontview_frames.append(img)

        # Record proprioception
        eef = self._env.get_eef_pose()
        self.eef_positions.append(np.array(
            [eef["x"], eef["y"], eef["z"]], dtype=np.float32))

        grip = self._env.get_gripper_state()
        self.gripper_states.append(np.array(
            [grip.get("open", 0.0)], dtype=np.float32))

        # Record action (convert to 4D)
        action_4d = np.array([
            action[0], action[1], action[2], action[6],
        ], dtype=np.float32)
        self.actions.append(action_4d)

        # Record phase
        self.phase_ids.append(phase_id)
        self.phase_names.append(phase_name or _PHASE_NAME_MAP.get(phase_id, "unknown"))

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    def to_hdf5(self, path: Path, episode_meta: dict[str, Any]) -> None:
        """Write all recorded data to an HDF5 file."""
        T = self.num_steps
        if T == 0:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            # Observations
            obs_g = f.create_group("observations")
            obs_g.create_dataset(
                "agentview",
                data=np.stack(self.agentview_frames, axis=0).astype(np.uint8),
            )
            obs_g.create_dataset(
                "frontview",
                data=np.stack(self.frontview_frames, axis=0).astype(np.uint8),
            )

            # Actions (4D)
            f.create_dataset(
                "actions_4d",
                data=np.stack(self.actions, axis=0).astype(np.float32),
            )

            # Proprioception
            prop_g = f.create_group("proprio")
            prop_g.create_dataset(
                "eef_pos",
                data=np.stack(self.eef_positions, axis=0).astype(np.float32),
            )
            prop_g.create_dataset(
                "gripper_open",
                data=np.stack(self.gripper_states, axis=0).astype(np.float32),
            )

            # Symbolic
            sym_g = f.create_group("symbolic")
            sym_g.create_dataset(
                "phase_id",
                data=np.array(self.phase_ids, dtype=np.int16),
            )
            # Store phase_name as fixed-length strings
            phase_names_bytes = [name.encode("utf-8") for name in self.phase_names]
            sym_g.create_dataset(
                "phase_name",
                data=np.array(phase_names_bytes, dtype=h5py.string_dtype()),
            )
            sym_g.create_dataset(
                "pick_object_id",
                data=np.full((T,), _OBJECT_ID_MAP.get(
                    episode_meta.get("pick_object", "cube1"), 1), dtype=np.int16),
            )
            sym_g.create_dataset(
                "target_object_id",
                data=np.full((T,), _OBJECT_ID_MAP.get(
                    episode_meta.get("target_object", "table"), 0), dtype=np.int16),
            )
            sym_g.create_dataset(
                "relation_id",
                data=np.full((T,), _RELATION_MAP.get(
                    episode_meta.get("relation", "none"), 2), dtype=np.int16),
            )
            sym_g.create_dataset(
                "action_type_id",
                data=np.full((T,), 0 if episode_meta.get("relation") == "on" else 1,
                             dtype=np.int16),
            )
            sym_g.create_dataset(
                "plan_step_index",
                data=np.zeros((T,), dtype=np.int16),
            )

            # Episode metadata
            f.attrs["episode_id"] = episode_meta.get("episode_id", "unknown")
            f.attrs["success"] = episode_meta.get("success", False)
            f.attrs["failure_type"] = episode_meta.get("failure_type", "") or ""
            f.attrs["teacher_config_version"] = episode_meta.get(
                "teacher_config_version", "v0")


# ── Episode Collection ──────────────────────────────────────────────────────

def collect_episodes(
    *,
    output_dir: str | Path,
    num_episodes: int,
    task: str,
    executor: str,
    seed: int,
    device: str = "cuda",
    record_rgb: bool = True,
    camera_names: tuple[str, ...] = ("agentview", "frontview"),
    image_size: int = IMAGE_SIZE,
    max_steps: int = 1000,
    enhanced_place: bool = False,
    enhanced_grasp: bool = False,
    preplace_fast: bool = False,
    high_place: bool = False,
    use_best_config: bool = False,
    **executor_kwargs,
) -> dict[str, Any]:
    """Collect VLA demonstration episodes.

    Args:
        output_dir: Directory to save HDF5 files + manifest.
        num_episodes: Number of episodes to collect.
        task: Task name ("two_block_stack").
        executor: Executor type ("scripted").
        seed: Random seed.
        device: Device for controller inference.
        record_rgb: Must be True for VLA data.
        camera_names: Camera names to record.
        image_size: RGB image size.
        max_steps: Max steps per episode.
        **executor_kwargs: Passed to episode runner.

    Returns:
        Collection summary dict.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Late imports to avoid Robosuite dependency at module level
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    # Determine task config
    if task == "two_block_stack":
        cubes = ["cube1", "cube2"]
        skill_commands = [
            {"skill_name": "pick_place", "object_id": "cube1",
             "target_object_id": "cube2", "relation": "on"},
            {"skill_name": "pick_place", "object_id": "cube2",
             "target_object_id": "cube1", "relation": "on"},
        ]
    else:
        raise ValueError(f"Unsupported task: {task}")

    rng = np.random.RandomState(seed)
    manifest_entries: list[dict[str, Any]] = []
    success_count = 0
    failure_types: dict[str, int] = {}
    total_start = time.time()

    for ep_idx in range(num_episodes):
        # Alternate between red_on_green and green_on_red
        cmd = skill_commands[ep_idx % 2]
        ep_seed = seed + ep_idx

        ep_id = f"ep_{ep_idx:06d}"
        ep_path = episodes_dir / f"{ep_id}.hdf5"

        print(f"[{ep_idx + 1}/{num_episodes}] {ep_id}  "
              f"{cmd['object_id']}→{cmd['target_object_id']}  seed={ep_seed}  ",
              end="", flush=True)

        ep_start = time.time()

        try:
            env = RobosuiteEnvWrapper(
                env_name="Stack",
                robots="Panda",
                cube_names=cubes,
                render_mode="offscreen",
            )
            env.reset()

            # Create recording wrapper
            recorder = VLARecordingWrapper(env, camera_names, image_size)

            # Import episode runner
            from vlm_nsm.experiments.robosuite_episode_runner import run_episode
            from vlm_nsm.core.data_types import Action

            # Build action from skill command
            action = Action(
                "move",
                [cmd["object_id"], "table", cmd["target_object_id"]],
            )

            # We run with a custom step hook that records RGB each step.
            # Strategy: use the existing runner but intercept env.step()
            # via monkey-patching the recording wrapper.
            _orig_step = env.step

            # Phase tracking: read from _RobosuiteRecordingWrapper._phase
            # which is updated by run_episode via env.set_phase()
            def _recording_step(raw_action):
                """Drop-in replacement for env.step() that records RGB."""
                pid = 0
                pname = ""
                # Read phase from env._phase (set by main loop + place_state_machine)
                if hasattr(env, '_phase') and env._phase:
                    pname = env._phase
                    pid = _PHASE_NAME_MAP_INV.get(pname, 0)
                recorder.record_step(raw_action, phase_id=pid, phase_name=pname)
                return _orig_step(raw_action)

            # Override env.step for recording
            env.step = _recording_step

            # Apply per-task enhanced profiles via episode runner kwargs
            place_kwargs = {}
            grasp_kwargs = {}
            if enhanced_place:
                if cmd["object_id"] == "cube2" and cmd["target_object_id"] == "cube1":
                    # green_on_red: use safe place mode (tighter tolerances)
                    place_kwargs = {"place_mode": "safe"}
            if preplace_fast:
                place_kwargs["preplace_fast_mode"] = True
            if enhanced_grasp:
                if cmd["object_id"] == "cube2":
                    # green cube: adaptive close (not aggressive), 2 retries
                    grasp_kwargs = {"grasp_close_steps": 35, "grasp_max_retries": 2,
                                    "close_mode": "adaptive"}

            # Run the episode using the existing scripted controller
            result = run_episode(
                env=env,
                actions=[action],
                cubes=cubes,
                goal_predicates=[
                    ["on", cmd["object_id"], cmd["target_object_id"]],
                ],
                policy_override=None,  # full scripted
                override_phases=set(),  # no BC override
                horizon=max_steps,
                early_stop_on_goal=True,
                recorder=None,  # Don't use external recorder for VLA collection
                **place_kwargs,
                **grasp_kwargs,
                high_place_enabled=high_place,
                # Stage 7A-12 best execution config (original paper)
                kp=0.08 if use_best_config else 0.05,
                max_delta=0.015 if use_best_config else 0.02,
                close_mode="adaptive" if use_best_config else "fixed",
                holding_check_enabled=use_best_config,
                lift_early_exit_enabled=use_best_config,
                grasp_max_retries=3,
                place_max_retries=2,
                skip_retreat_if_goal_satisfied=True,
                horizon_aware=True,
                **executor_kwargs,
            )

            # Back-fill phase info from the phase_trace in the result
            # The _RobosuiteRecordingWrapper records per-phase step counts
            phase_trace = result.get("phase_trace", {})
            if phase_trace:
                # Reconstruct phase sequence from trace
                # phase_trace is {"reach_pick": N, "grasp": M, ...}
                phase_order = ["reach_pick", "grasp", "lift",
                              "reach_place", "release", "retreat"]
                phase_seq = []
                for pname in phase_order:
                    count = phase_trace.get(pname, 0)
                    if count > 0:
                        phase_seq.extend([(pname, _PHASE_NAME_MAP_INV.get(pname, 0))] * count)
                # Override the recorded phases with correct ones
                for i, (pname, pid) in enumerate(phase_seq):
                    if i < len(recorder.phase_ids):
                        recorder.phase_ids[i] = pid
                        recorder.phase_names[i] = pname

            # Restore original step
            env.step = _orig_step

            success = bool(result.get("goal_success", result.get("success", False)))
            failure = result.get("failure_type") if not success else None
            steps = recorder.num_steps

            # Phase 2.0: operator-level effect check
            from vlm_nsm.neuro_symbolic.operator_monitor import OperatorMonitor
            from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
            _monitor = OperatorMonitor()
            _op_check = None
            try:
                _predicates = physical_state_to_predicates(
                    env.get_all_object_poses(),
                    {t: env.get_target_pose(t) for t in cubes},
                    {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05},
                )
                _sc = SkillCommand(
                    skill_name="pick_place",
                    object_id=cmd["object_id"],
                    target_object_id=cmd["target_object_id"],
                    relation=cmd["relation"],
                    preconditions=[
                        {"name": "clear", "args": [cmd["object_id"]]},
                        {"name": "clear", "args": [cmd["target_object_id"]]},
                        {"name": "handempty", "args": []},
                    ],
                    expected_add_effects=[
                        {"name": "on", "args": [cmd["object_id"], cmd["target_object_id"]]},
                        {"name": "holding", "args": [cmd["object_id"]]},
                        {"name": "clear", "args": [cmd["object_id"]]},
                    ],
                )
                _op_check = _monitor.check_after_execution(
                    _sc, _predicates, failure).to_dict()
            except Exception:
                _op_check = {"error": "operator_check_failed"}

            # Write HDF5
            episode_meta = {
                "episode_id": ep_id,
                "success": success,
                "failure_type": failure,
                "pick_object": cmd["object_id"],
                "target_object": cmd["target_object_id"],
                "relation": cmd["relation"],
                "teacher_config_version": "current_scripted_v0",
            }
            recorder.to_hdf5(ep_path, episode_meta)

            # Write manifest entry
            manifest_entries.append({
                "episode_id": ep_id,
                "episode_path": str(ep_path),
                "success": success,
                "failure_type": failure,
                "pick_object": cmd["object_id"],
                "target_object": cmd["target_object_id"],
                "relation": cmd["relation"],
                "num_steps": steps,
                "seed": ep_seed,
                "operator_check": _op_check,
            })

            if success:
                success_count += 1
            else:
                ft = failure or "unknown"
                failure_types[ft] = failure_types.get(ft, 0) + 1

            elapsed = time.time() - ep_start
            print(f"success={success}  steps={steps}  {elapsed:.1f}s")

        except Exception as exc:
            print(f"ERROR: {exc}")
            manifest_entries.append({
                "episode_id": ep_id,
                "episode_path": str(ep_path),
                "success": False,
                "failure_type": "runtime_error",
                "pick_object": cmd["object_id"],
                "target_object": cmd["target_object_id"],
                "relation": cmd["relation"],
                "num_steps": 0,
                "seed": ep_seed,
                "error": str(exc),
            })
            failure_types["runtime_error"] = failure_types.get("runtime_error", 0) + 1
            continue

    total_elapsed = time.time() - total_start

    # Write manifest
    manifest_path = output / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Compute and save normalization stats
    from vlm_nsm.vla.data.vla_dataset import save_normalization_stats
    try:
        save_normalization_stats(output, max_episodes=min(num_episodes, 500))
    except Exception:
        pass  # stats can be computed later if this fails

    summary = {
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_count / max(num_episodes, 1),
        "failure_types": failure_types,
        "total_elapsed_s": total_elapsed,
        "output_dir": str(output),
    }

    # Write summary
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nCollection complete: {success_count}/{num_episodes} successful "
          f"({summary['success_rate']:.1%})  "
          f"elapsed={total_elapsed:.0f}s")
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect VLA demonstration data with RGB recording")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for HDF5 files + manifest")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--task", default="two_block_stack",
                        choices=["two_block_stack"])
    parser.add_argument("--executor", default="scripted",
                        choices=["scripted"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera_names", nargs="+",
                        default=["agentview", "frontview"])
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--enhanced_place", action="store_true",
                        help="Use per-task place profiles (safe for green_on_red)")
    parser.add_argument("--enhanced_grasp", action="store_true",
                        help="Use per-task grasp profiles (conservative for green cube)")
    parser.add_argument("--preplace_fast", action="store_true",
                        help="Use aggressive P-control in preplace (IK-like fast positioning)")
    parser.add_argument("--high_place", action="store_true",
                        help="Use high-place path: lift → XY align → slow descend")
    parser.add_argument("--use_best_config", action="store_true",
                        help="Use Stage 7A-12 best execution config (kp=0.08, adaptive, etc.)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = collect_episodes(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        task=args.task,
        executor=args.executor,
        seed=args.seed,
        device=args.device,
        camera_names=tuple(args.camera_names),
        image_size=args.image_size,
        max_steps=args.max_steps,
        enhanced_place=getattr(args, 'enhanced_place', False),
        enhanced_grasp=getattr(args, 'enhanced_grasp', False),
        preplace_fast=getattr(args, 'preplace_fast', False),
        high_place=getattr(args, 'high_place', False),
        use_best_config=getattr(args, 'use_best_config', False),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
