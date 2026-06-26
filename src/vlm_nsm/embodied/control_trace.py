"""ControlTrace — per-step execution metrics recorder.

Records detailed per-step diagnostics during episode execution for
failure analysis, phase statistics, and training data quality assessment.

Independent of Robosuite — operates on generic env interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class PhaseStepMetric:
    """Metrics recorded at a single execution step."""
    step: int
    phase: str
    phase_id: int
    eef_pos: list[float]         # [x, y, z]
    obj_pos: list[float]          # pick object position
    target_pos: list[float]       # target position (if applicable)
    action: list[float]           # 7-dim action executed
    eef_obj_xy_error: float       # horizontal distance to object
    eef_obj_z_error: float        # vertical distance to object
    gripper_open: float           # gripper openness [0, 1]
    object_height: float          # object z above table
    holding_confidence: Optional[float] = None
    contact_detected: bool = False
    action_source: str = "scripted"  # "scripted" | "vla" | "hybrid"


@dataclass
class PhaseSummary:
    """Aggregated metrics for one execution phase."""
    phase: str
    phase_id: int
    success: bool
    failure_type: Optional[str]
    steps: int
    retries: int = 0
    min_xy_error: float = float("inf")
    final_xy_error: float = 0.0
    final_z_error: float = 0.0
    mean_gripper_open: float = 0.0
    contact_detected: bool = False


@dataclass
class SkillExecutionTrace:
    """Complete trace for one skill execution."""
    command_id: str
    object_id: str
    target_id: str = ""
    source: str = "oracle"
    success: bool = False
    failure_type: Optional[str] = None
    failure_phase: Optional[str] = None
    total_steps: int = 0
    phase_summaries: list[PhaseSummary] = field(default_factory=list)
    step_metrics: list[PhaseStepMetric] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "object_id": self.object_id,
            "target_id": self.target_id,
            "source": self.source,
            "success": self.success,
            "failure_type": self.failure_type,
            "failure_phase": self.failure_phase,
            "total_steps": self.total_steps,
            "phase_summaries": [
                {
                    "phase": ps.phase,
                    "phase_id": ps.phase_id,
                    "success": ps.success,
                    "failure_type": ps.failure_type,
                    "steps": ps.steps,
                    "retries": ps.retries,
                    "min_xy_error": ps.min_xy_error,
                    "final_xy_error": ps.final_xy_error,
                    "final_z_error": ps.final_z_error,
                    "contact_detected": ps.contact_detected,
                }
                for ps in self.phase_summaries
            ],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ── ControlTrace Recorder ──────────────────────────────────────────────────

class ControlTraceRecorder:
    """Records per-step execution metrics during an episode.

    Usage::

        recorder = ControlTraceRecorder(object_id="cube1", target_id="cube2")
        recorder.start_phase("reach_pick", phase_id=0)

        for step in range(max_steps):
            action = controller.compute_action(obs)
            env.step(action)
            recorder.record_step(
                step=step, phase="reach_pick", phase_id=0,
                eef_pos=eef_pos, obj_pos=obj_pos, action=action,
                gripper_open=gripper_open, table_z=table_z,
            )

        recorder.end_phase(success=True)
        trace = recorder.finalize(success=True)
    """

    # Phase names compatible with existing 6-phase decomposition
    PHASE_NAMES = [
        "reach_pick", "grasp", "lift",
        "reach_place", "release", "retreat",
    ]

    def __init__(
        self,
        object_id: str,
        target_id: str = "",
        source: str = "oracle",
        table_z: float = 0.80,
    ):
        self._object_id = object_id
        self._target_id = target_id
        self._source = source
        self._table_z = table_z

        self._current_phase: str = ""
        self._current_phase_id: int = -1
        self._phase_steps: int = 0
        self._phase_min_xy: float = float("inf")

        self._step_metrics: list[PhaseStepMetric] = []
        self._phase_summaries: list[PhaseSummary] = []

    def start_phase(self, phase: str, phase_id: int = 0) -> None:
        self._current_phase = phase
        self._current_phase_id = phase_id
        self._phase_steps = 0
        self._phase_min_xy = float("inf")

    def record_step(
        self,
        step: int,
        phase: str,
        phase_id: int,
        eef_pos: np.ndarray | list[float],
        obj_pos: np.ndarray | list[float],
        action: np.ndarray | list[float],
        gripper_open: float,
        target_pos: np.ndarray | list[float] | None = None,
        holding_confidence: float | None = None,
        contact_detected: bool = False,
        action_source: str = "scripted",
    ) -> None:
        eef_pos = list(eef_pos) if hasattr(eef_pos, "__iter__") else [0., 0., 0.]
        obj_pos = list(obj_pos) if hasattr(obj_pos, "__iter__") else [0., 0., 0.]
        action = list(action) if hasattr(action, "__iter__") else [0.] * 7
        target_pos = list(target_pos) if target_pos is not None and hasattr(target_pos, "__iter__") else [0., 0., 0.]

        eef_xy = np.array(eef_pos[:2])
        obj_xy = np.array(obj_pos[:2])
        xy_err = float(np.linalg.norm(eef_xy - obj_xy))
        z_err = float(eef_pos[2] - obj_pos[2])

        self._phase_steps += 1
        self._phase_min_xy = min(self._phase_min_xy, xy_err)

        self._step_metrics.append(PhaseStepMetric(
            step=step,
            phase=phase,
            phase_id=phase_id,
            eef_pos=eef_pos,
            obj_pos=obj_pos,
            target_pos=target_pos,
            action=action,
            eef_obj_xy_error=xy_err,
            eef_obj_z_error=z_err,
            gripper_open=gripper_open,
            object_height=float(obj_pos[2]) - self._table_z,
            holding_confidence=holding_confidence,
            contact_detected=contact_detected,
            action_source=action_source,
        ))

    def end_phase(
        self,
        success: bool,
        failure_type: str | None = None,
        retries: int = 0,
        contact_detected: bool = False,
    ) -> None:
        last = self._step_metrics[-1] if self._step_metrics else None
        final_xy = last.eef_obj_xy_error if last else 0.0
        final_z = last.eef_obj_z_error if last else 0.0

        grip_vals = [
            m.gripper_open for m in self._step_metrics[-self._phase_steps:]
        ] if self._phase_steps > 0 else [0.0]

        self._phase_summaries.append(PhaseSummary(
            phase=self._current_phase,
            phase_id=self._current_phase_id,
            success=success,
            failure_type=failure_type,
            steps=self._phase_steps,
            retries=retries,
            min_xy_error=self._phase_min_xy,
            final_xy_error=final_xy,
            final_z_error=final_z,
            mean_gripper_open=float(np.mean(grip_vals)),
            contact_detected=contact_detected,
        ))

    def finalize(
        self,
        success: bool,
        failure_type: str | None = None,
        failure_phase: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillExecutionTrace:
        return SkillExecutionTrace(
            command_id=f"{self._object_id}_{self._target_id}",
            object_id=self._object_id,
            target_id=self._target_id,
            source=self._source,
            success=success,
            failure_type=failure_type,
            failure_phase=failure_phase,
            total_steps=len(self._step_metrics),
            phase_summaries=list(self._phase_summaries),
            step_metrics=list(self._step_metrics),
            metadata=metadata or {},
        )

    @classmethod
    def phase_name(cls, phase_id: int) -> str:
        if 0 <= phase_id < len(cls.PHASE_NAMES):
            return cls.PHASE_NAMES[phase_id]
        return f"phase_{phase_id}"
