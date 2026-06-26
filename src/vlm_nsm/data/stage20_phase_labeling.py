"""Stage 20 phase labeling helpers.

The robosuite place state machine executes preplace, descend, release, hold,
and retreat inside the outer ``reach_place`` skill. The recorder therefore sees
all of those timesteps as phase 3 unless we split the composite segment after
recording. These helpers are pure numpy utilities and do not run robosuite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHASE_ID_TO_NAME = {
    0: "reach_pick",
    1: "grasp",
    2: "lift",
    3: "reach_place",
    4: "release",
    5: "retreat",
}


@dataclass(frozen=True)
class PhaseRelabelStats:
    """Summary of composite-place phase relabeling."""

    changed_steps: int
    release_steps: int
    retreat_steps: int
    release_chunks: int
    retreat_chunks: int

    @property
    def applied(self) -> bool:
        return self.changed_steps > 0

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "applied": self.applied,
            "changed_steps": self.changed_steps,
            "release_steps": self.release_steps,
            "retreat_steps": self.retreat_steps,
            "release_chunks": self.release_chunks,
            "retreat_chunks": self.retreat_chunks,
        }


def phase_names(phase_ids: np.ndarray) -> np.ndarray:
    """Convert integer phase ids to string names."""
    ids = np.asarray(phase_ids, dtype=np.int64).reshape(-1)
    return np.array([PHASE_ID_TO_NAME.get(int(p), "unknown") for p in ids])


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool).tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def relabel_composite_place_phases(
    phase_ids: np.ndarray,
    actions: np.ndarray,
    *,
    reach_place_phase: int = 3,
    release_phase: int = 4,
    retreat_phase: int = 5,
    gripper_open_threshold: float = -0.5,
    retreat_dz_threshold: float = 0.01,
) -> tuple[np.ndarray, PhaseRelabelStats]:
    """Split composite reach_place timesteps into release and retreat phases.

    The current place state machine uses normalized action dimension 6 for the
    gripper and dimension 2 for z motion. Inside a phase-3 segment:
    - open-gripper timesteps with near-zero z motion are release/hold;
    - open-gripper timesteps with positive z motion are retreat.

    Non-open timesteps remain ``reach_place``. Existing phase 4/5 labels are
    preserved; only phase 3 timesteps are considered.
    """
    phase = np.asarray(phase_ids, dtype=np.int64).reshape(-1)
    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] != phase.shape[0] or action.shape[1] < 7:
        raise ValueError("actions must have shape [T, >=7] and match phase_ids length")

    out = phase.copy()
    release_chunks = 0
    retreat_chunks = 0
    for start, end in _contiguous_true_runs(phase == reach_place_phase):
        segment_actions = action[start:end]
        open_mask = segment_actions[:, 6] <= gripper_open_threshold
        for local_open_start, local_open_end in _contiguous_true_runs(open_mask):
            absolute_open_start = start + local_open_start
            absolute_open_end = start + local_open_end
            dz = action[absolute_open_start:absolute_open_end, 2]
            retreat_local = np.where(dz > retreat_dz_threshold)[0]
            if retreat_local.size:
                retreat_start = absolute_open_start + int(retreat_local[0])
                if retreat_start > absolute_open_start:
                    out[absolute_open_start:retreat_start] = release_phase
                    release_chunks += 1
                out[retreat_start:absolute_open_end] = retreat_phase
                retreat_chunks += 1
            else:
                out[absolute_open_start:absolute_open_end] = release_phase
                release_chunks += 1

    changed = out != phase
    stats = PhaseRelabelStats(
        changed_steps=int(changed.sum()),
        release_steps=int((out == release_phase).sum() - (phase == release_phase).sum()),
        retreat_steps=int((out == retreat_phase).sum() - (phase == retreat_phase).sum()),
        release_chunks=release_chunks,
        retreat_chunks=retreat_chunks,
    )
    return out, stats
