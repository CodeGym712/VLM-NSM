"""Symbolic condition encoder — SkillCommand → structured conditioning vector.

Converts a SkillCommand (from symbolic planning) into a fixed-size numeric
vector that conditions the VLA policy with structured task semantics.

The encoded vector includes:
  - Object identity (pick_object, target_object)
  - Object attributes (color, size)
  - Relation type (on/table/none)
  - Phase identity (current execution phase)
  - Action type (move_on / move_to_table)
  - Plan context (step index, remaining steps)

Designed to be independent of Robosuite — works with any environment
that provides object_ids and an optional object registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vlm_nsm.skills.skill_types import SkillCommand

# ── Constants ───────────────────────────────────────────────────────────────

# Object color mapping (canonical → color name)
_OBJECT_COLOR_MAP: dict[str, str] = {
    "cube1": "red",
    "cube2": "green",
    "cube3": "blue",
    "red_cube": "red",
    "green_cube": "green",
    "blue_cube": "blue",
}

# Object size mapping (canonical → meters)
_OBJECT_SIZE_MAP: dict[str, float] = {
    "cube1": 0.04,   # red, 4cm
    "cube2": 0.05,   # green, 5cm
    "cube3": 0.04,   # blue (assumed 4cm)
}

# Known color set
_KNOWN_COLORS: list[str] = ["red", "green", "blue"]

# Known relations
_KNOWN_RELATIONS: list[str] = ["on", "table", "none"]

# Phase names (from skill_step_decomposer + micro_align extension)
_PHASE_NAMES: list[str] = [
    "reach_pick",
    "descend_align",
    "micro_align",
    "grasp_close",
    "lift_verify",
    "transport",
    "descend_place",
    "micro_place",
    "release_verify",
    "retreat",
]

# Action types
_ACTION_TYPES: list[str] = ["move_on", "move_to_table", "unknown"]


# ── Public API ──────────────────────────────────────────────────────────────

@dataclass
class SymbolicConditionVector:
    """Structured symbolic condition for VLA conditioning.

    All fields are numpy arrays ready for concatenation into a flat vector.
    """

    pick_object_onehot: np.ndarray       # (N_objects,)
    target_object_onehot: np.ndarray     # (N_objects + 1,)  [+1 for table]
    relation_onehot: np.ndarray          # (3,)  on/table/none
    phase_onehot: np.ndarray             # (N_phases,)
    action_type_onehot: np.ndarray       # (3,)  move_on/move_to_table/unknown
    pick_color_onehot: np.ndarray        # (3,)  red/green/blue
    target_color_onehot: np.ndarray      # (3,)
    pick_size: np.ndarray                # (1,)  normalized
    target_size: np.ndarray              # (1,)  normalized
    plan_step_index: np.ndarray          # (1,)  normalized
    remaining_steps: np.ndarray          # (1,)  normalized

    def as_vector(self) -> np.ndarray:
        """Return the flat concatenated symbolic condition vector."""
        return np.concatenate([
            self.pick_object_onehot,
            self.target_object_onehot,
            self.relation_onehot,
            self.phase_onehot,
            self.action_type_onehot,
            self.pick_color_onehot,
            self.target_color_onehot,
            self.pick_size,
            self.target_size,
            self.plan_step_index,
            self.remaining_steps,
        ]).astype(np.float32)

    @property
    def dim(self) -> int:
        """Total dimension of the flat vector."""
        return len(self.as_vector())


class SymbolicConditionEncoder:
    """Encodes SkillCommand + execution context into a SymbolicConditionVector.

    The encoder manages an object registry so that object_ids encountered
    during training/evaluation are consistently mapped to one-hot indices.

    Usage::

        encoder = SymbolicConditionEncoder(objects=["cube1", "cube2"])
        cond = encoder.encode(
            command=skill_command,
            phase_name="grasp_close",
            plan_step_index=0,
            total_plan_steps=1,
        )
        vec = cond.as_vector()   # → np.ndarray of dim ~42
    """

    def __init__(
        self,
        objects: list[str],
        table_name: str = "table",
        color_map: dict[str, str] | None = None,
        size_map: dict[str, float] | None = None,
    ):
        """Initialize encoder with known object set.

        Args:
            objects: Canonical object ids in the scene (e.g. ["cube1","cube2"]).
            table_name: Name used for the table target.
            color_map: Optional override for object→color mapping.
            size_map: Optional override for object→size mapping.
        """
        self._objects = list(objects)
        self._table_name = table_name
        self._color_map = dict(color_map) if color_map else dict(_OBJECT_COLOR_MAP)
        self._size_map = dict(size_map) if size_map else dict(_OBJECT_SIZE_MAP)

        # Build stable index lookups
        self._object_to_idx = {obj: i for i, obj in enumerate(self._objects)}
        # target includes objects + table
        self._target_to_idx = {obj: i for i, obj in enumerate(self._objects)}
        self._target_to_idx[table_name] = len(self._objects)

        self._dim = None  # computed lazily

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def n_objects(self) -> int:
        return len(self._objects)

    @property
    def n_targets(self) -> int:
        return len(self._objects) + 1  # objects + table

    @property
    def dim(self) -> int:
        """Total dimension of the encoded symbolic condition vector."""
        if self._dim is None:
            dummy = self.encode(
                SkillCommand(skill_name="pick_place", object_id=self._objects[0]),
                phase_name="reach_pick",
            )
            self._dim = dummy.dim
        return self._dim

    # ── Main encode method ───────────────────────────────────────────────

    def encode(
        self,
        command: SkillCommand,
        phase_name: str = "reach_pick",
        plan_step_index: int = 0,
        total_plan_steps: int = 1,
        object_roles: dict[str, str] | None = None,
        operator_type: str | None = None,  # Phase 3.0b: "Pick" or "PlaceOn"
    ) -> SymbolicConditionVector:
        """Encode a SkillCommand and execution context.

        Args:
            command: The skill command to encode.
            phase_name: Current execution phase name.
            plan_step_index: 0-based index of this command in the plan.
            total_plan_steps: Total number of skill commands in the plan.
            object_roles: Optional mapping of object_id → role
                          (e.g. {"cube1": "picked", "cube2": "support"}).

        Returns:
            SymbolicConditionVector ready for concatenation.
        """
        pick_id = command.object_id
        target_id = command.target_object_id or self._table_name
        relation = command.relation or "none"
        action_type = self._infer_action_type(command)

        return SymbolicConditionVector(
            pick_object_onehot=self._onehot(pick_id, self._objects),
            target_object_onehot=self._onehot(target_id, list(self._target_to_idx.keys())),
            relation_onehot=self._onehot(relation, _KNOWN_RELATIONS),
            phase_onehot=self._onehot(phase_name, _PHASE_NAMES),
            action_type_onehot=self._onehot(action_type, _ACTION_TYPES),
            pick_color_onehot=self._color_onehot(pick_id),
            target_color_onehot=self._color_onehot(target_id),
            pick_size=self._normalized_size(pick_id),
            target_size=self._normalized_size(target_id),
            plan_step_index=self._normalized_scalar(plan_step_index, max(total_plan_steps, 1)),
            remaining_steps=self._normalized_scalar(
                total_plan_steps - plan_step_index - 1, max(total_plan_steps, 1)
            ),
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _onehot(self, key: str, vocabulary: list[str]) -> np.ndarray:
        idx = vocabulary.index(key) if key in vocabulary else len(vocabulary) - 1
        vec = np.zeros(len(vocabulary), dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _color_onehot(self, object_id: str) -> np.ndarray:
        if object_id == self._table_name:
            return np.zeros(len(_KNOWN_COLORS), dtype=np.float32)
        color = self._color_map.get(object_id, "red")
        return self._onehot(color, _KNOWN_COLORS)

    def _normalized_size(self, object_id: str) -> np.ndarray:
        if object_id == self._table_name:
            return np.array([0.0], dtype=np.float32)
        raw_size = self._size_map.get(object_id, 0.04)
        return np.array([raw_size / 0.06], dtype=np.float32)

    @staticmethod
    def _normalized_scalar(value: int, max_value: int) -> np.ndarray:
        norm = max(float(value), 0.0) / float(max(max_value, 1))
        return np.array([norm], dtype=np.float32)

    @staticmethod
    def _infer_action_type(command: SkillCommand) -> str:
        if command.target_object_id and command.target_object_id != "table":
            return "move_on"
        return "move_to_table"


# ── Convenience: vocabulary sizes for model config ─────────────────────────

def symbolic_condition_dim(n_objects: int) -> int:
    """Return the total dimension given the number of objects.

    Useful for configuring VLA model input_dim.
    """
    return (
        n_objects                       # pick_object_onehot
        + (n_objects + 1)               # target_object_onehot
        + 3                             # relation_onehot
        + len(_PHASE_NAMES)             # phase_onehot (10)
        + 3                             # action_type_onehot
        + 3                             # pick_color_onehot
        + 3                             # target_color_onehot
        + 1                             # pick_size
        + 1                             # target_size
        + 1                             # plan_step_index
        + 1                             # remaining_steps
    )
