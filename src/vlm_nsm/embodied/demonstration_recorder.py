"""DemonstrationRecorder.

Records demonstration data from embodied execution into npz format
compatible with downstream BC training (MLP BC, Temporal BC, Diffusion Policy).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class DemonstrationRecorder:
    """Records demonstrations to npz files with standardized schema.

    npz schema:
        obs:          (T, obs_dim)    — Primary observation (filtered, backward compatible)
        filtered_obs: (T, obs_dim)    — Explicit filtered observation
        full_obs:     (T, full_obs_dim) — Full observation (optional)
        action:       (T, action_dim) — Action sequence
        done:         (T,)            — Episode termination flags
        episode_id:   (T,)            — Episode index per timestep
        skill_id:     (T,)            — Skill index per timestep
        step_id:      (T,)            — Step index within episode
        success:      (T,)            — Success flag per timestep

    metadata JSON schema (saved alongside npz):
        See ``metadata_schema()`` for fields.
    """

    def __init__(self, output_dir: str | Path, obs_dim: int, action_dim: int, full_obs_dim: int | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.full_obs_dim = full_obs_dim

        # Buffers for current recording session
        self._obs: list[np.ndarray] = []
        self._filtered_obs: list[np.ndarray] = []
        self._full_obs: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._done: list[bool] = []
        self._episode_id: list[int] = []
        self._skill_id: list[int] = []
        self._phase_id: list[int] = []
        self._step_id: list[int] = []
        self._success: list[bool] = []
        self._current_episode = 0

    def record_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        done: bool,
        skill_id: int,
        step_id: int,
        success: bool,
        filtered_obs: np.ndarray | None = None,
        full_obs: np.ndarray | None = None,
        phase_id: int | None = None,
    ):
        """Record a single step.

        Args:
            obs: Observation vector of shape (obs_dim,). Primary observation (filtered).
            action: Action vector of shape (action_dim,).
            done: Whether the episode terminated.
            skill_id: Skill index (0=reach_pick, 1=grasp, 2=lift, 3=reach_place, 4=release, 5=retreat).
            step_id: Step index within the episode.
            success: Whether the step was successful.
            filtered_obs: Explicit filtered observation (same as obs if None).
            full_obs: Full (unfiltered) observation vector, if available.
            phase_id: Phase index (same as skill_id for 6-phase system). Defaults to skill_id if None.
        """
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._filtered_obs.append(np.asarray(filtered_obs if filtered_obs is not None else obs, dtype=np.float32))
        if full_obs is not None:
            self._full_obs.append(np.asarray(full_obs, dtype=np.float32))
        self._action.append(np.asarray(action, dtype=np.float32))
        self._done.append(done)
        self._episode_id.append(self._current_episode)
        self._skill_id.append(skill_id)
        self._phase_id.append(phase_id if phase_id is not None else skill_id)
        self._step_id.append(step_id)
        self._success.append(success)

    def end_episode(self):
        """Mark the current episode as complete and increment episode counter."""
        self._current_episode += 1

    # Stage 20-R: OBS_INDEX for gripper_has_object in the 11-dim robosuite obs vector.
    _OBS_IDX_GRIPPER_HAS_OBJECT = 4

    def _extract_contact_holding_signals(
        self, filtered_obs_arr: np.ndarray
    ) -> dict[str, np.ndarray | bool]:
        """Extract contact/holding signals from filtered observation array.

        Stage 20-R: The only real per-step signal available is ``gripper_has_object``
        at obs index 4 (heuristic: ``0.005 < gripper_open < 0.035``).
        We do NOT fabricate top_in_gripper / bottom_in_gripper — they require
        per-object contact geom info which is unavailable.
        """
        result: dict[str, np.ndarray | bool] = {
            "contact_signal_available": True,
            "contact_signal_source": "obs_index_4_gripper_has_object_heuristic",
            "holding_signal_available": True,
            "holding_signal_source": "obs_index_4_gripper_has_object_heuristic",
        }

        if filtered_obs_arr.ndim == 2 and filtered_obs_arr.shape[1] > self._OBS_IDX_GRIPPER_HAS_OBJECT:
            gripper_has_object = filtered_obs_arr[:, self._OBS_IDX_GRIPPER_HAS_OBJECT].astype(np.float32)
            result["gripper_has_object"] = gripper_has_object
            # Contact/holding are the same signal — no per-object differentiation available
            result["contact_signal"] = gripper_has_object.copy()
            result["holding_signal"] = gripper_has_object.copy()
        else:
            result["contact_signal_available"] = False
            result["contact_signal_source"] = "unavailable_obs_too_narrow"
            result["holding_signal_available"] = False
            result["holding_signal_source"] = "unavailable_obs_too_narrow"

        # Fields that require per-object contact info — always unavailable
        result["top_in_gripper_available"] = False
        result["top_in_gripper_source"] = "unavailable_no_per_object_contact"
        result["bottom_in_gripper_available"] = False
        result["bottom_in_gripper_source"] = "unavailable_no_per_object_contact"

        return result

    def pop_last_episode_data(self) -> dict[str, np.ndarray] | None:
        """Extract and remove the most recently completed episode's data.

        Returns a dict of arrays suitable for ``np.savez_compressed``,
        or ``None`` if no data is available for the last episode.

        The episode data is removed from internal buffers so subsequent
        calls to ``save()`` will not include it.

        Stage 20-R: Also extracts ``gripper_has_object``, ``contact_signal``,
        and ``holding_signal`` from obs index 4 when available.
        """
        target_ep = self._current_episode - 1
        indices = [i for i, eid in enumerate(self._episode_id) if eid == target_ep]
        if not indices:
            return None

        start, end = indices[0], indices[-1] + 1

        filtered_obs_arr = np.stack(self._filtered_obs[start:end]).astype(np.float32)

        data: dict[str, np.ndarray] = {
            "obs": np.stack(self._obs[start:end]).astype(np.float32),
            "filtered_obs": filtered_obs_arr,
            "action": np.stack(self._action[start:end]).astype(np.float32),
            "done": np.array(self._done[start:end]),
            "episode_id": np.array(self._episode_id[start:end], dtype=np.int32),
            "skill_id": np.array(self._skill_id[start:end], dtype=np.int32),
            "phase_id": np.array(self._phase_id[start:end], dtype=np.int32),
            "step_id": np.array(self._step_id[start:end], dtype=np.int32),
            "success": np.array(self._success[start:end]),
        }
        if self._full_obs:
            data["full_obs"] = np.stack(self._full_obs[start:end]).astype(np.float32)

        # Stage 20-R: extract contact/holding signals from filtered_obs
        contact_holding = self._extract_contact_holding_signals(filtered_obs_arr)
        for key, value in contact_holding.items():
            if isinstance(value, np.ndarray):
                data[key] = value
        # Store metadata flags as scalar arrays for npz compatibility
        data["contact_signal_available_flag"] = np.array(
            contact_holding["contact_signal_available"], dtype=bool
        )
        data["holding_signal_available_flag"] = np.array(
            contact_holding["holding_signal_available"], dtype=bool
        )

        # Remove these entries from all buffers
        for buf in (
            self._obs, self._filtered_obs, self._action, self._done,
            self._episode_id, self._skill_id, self._phase_id,
            self._step_id, self._success,
        ):
            del buf[start:end]
        if self._full_obs:
            del self._full_obs[start:end]

        return data

    def save(
        self,
        filename: str = "demonstrations.npz",
        task: str = "",
        n_blocks: int = 0,
        env_type: str = "mock",
        controller_type: str = "scripted",
        step_names: list[str] | None = None,
        observation_type: str = "symbol_conditioned",
        noise_config: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Save recorded demonstrations to npz and metadata to JSON.

        Args:
            filename: Output filename (should end in .npz).
            task: Task name ("hanoi" or "stacking").
            n_blocks: Number of blocks.
            env_type: Environment type ("mock" or "robosuite").
            controller_type: Controller type ("scripted" or "replay").
            step_names: List of step name strings.
            observation_type: Type of observation ("symbol_conditioned" or "raw").
            noise_config: Noise configuration dict (for metadata).

        Returns:
            Tuple of (npz_path, metadata_path).
        """
        if not filename.endswith(".npz"):
            filename += ".npz"

        obs_arr = np.stack(self._obs) if self._obs else np.array([], dtype=np.float32)
        action_arr = np.stack(self._action) if self._action else np.array([], dtype=np.float32)

        save_dict: dict[str, Any] = {
            "obs": obs_arr,
            "filtered_obs": np.stack(self._filtered_obs) if self._filtered_obs else np.array([], dtype=np.float32),
            "action": action_arr,
            "done": np.array(self._done, dtype=bool),
            "episode_id": np.array(self._episode_id, dtype=np.int32),
            "skill_id": np.array(self._skill_id, dtype=np.int32),
            "phase_id": np.array(self._phase_id, dtype=np.int32),
            "step_id": np.array(self._step_id, dtype=np.int32),
            "success": np.array(self._success, dtype=bool),
        }

        # Stage 20-R: extract contact/holding signals from accumulated filtered_obs
        if self._filtered_obs:
            all_filtered = np.stack(self._filtered_obs).astype(np.float32)
            contact_holding = self._extract_contact_holding_signals(all_filtered)
            for key, value in contact_holding.items():
                if isinstance(value, np.ndarray):
                    save_dict[key] = value
            save_dict["contact_signal_available_flag"] = np.array(
                contact_holding["contact_signal_available"], dtype=bool
            )
            save_dict["holding_signal_available_flag"] = np.array(
                contact_holding["holding_signal_available"], dtype=bool
            )

        if self._full_obs:
            save_dict["full_obs"] = np.stack(self._full_obs)

        npz_path = self.output_dir / filename
        np.savez_compressed(npz_path, **save_dict)

        num_episodes = self._current_episode
        num_success = int(np.sum(self._success)) if self._success else 0

        metadata: dict[str, Any] = {
            "task": task,
            "n_blocks": n_blocks,
            "env_type": env_type,
            "controller_type": controller_type,
            "num_episodes": num_episodes,
            "num_success": num_success,
            "success_rate": float(np.mean(self._success)) if self._success else 0.0,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "step_names": step_names or [],
            "observation_type": observation_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Extended obs fields
        metadata["filtered_obs_dim"] = self.obs_dim
        if self._full_obs and self._full_obs[0] is not None:
            metadata["full_obs_dim"] = int(self._full_obs[0].shape[-1])
        else:
            metadata["full_obs_dim"] = self.full_obs_dim
        metadata["full_obs_available"] = bool(self._full_obs)
        if metadata["full_obs_available"] and metadata.get("full_obs_dim") and self.obs_dim:
            metadata["reduction_ratio"] = 1.0 - (self.obs_dim / metadata["full_obs_dim"])
        else:
            metadata["reduction_ratio"] = None

        if noise_config:
            metadata["noise_config"] = noise_config

        meta_path = npz_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        return npz_path, meta_path

    def clear(self):
        """Clear all recorded buffers."""
        self._obs.clear()
        self._filtered_obs.clear()
        self._full_obs.clear()
        self._action.clear()
        self._done.clear()
        self._episode_id.clear()
        self._skill_id.clear()
        self._phase_id.clear()
        self._step_id.clear()
        self._success.clear()
        self._current_episode = 0

    @property
    def total_steps(self) -> int:
        return len(self._obs)

    @property
    def total_episodes(self) -> int:
        return self._current_episode
