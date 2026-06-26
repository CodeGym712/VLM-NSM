"""Neuro-Symbolic Safety Gate — validates VLA actions before execution.

The safety gate combines geometric checks (workspace bounds, action magnitude)
with symbolic constraints (phase semantics) to decide whether to accept a
VLA-proposed action or fall back to the scripted controller.

Design principles:
  - Fast: all checks are O(1) vector operations, no model inference.
  - Conservative: when in doubt, fall back to scripted.
  - Transparent: every rejection logs a machine-readable reason.
  - Independent: does not depend on Robosuite or any specific env.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── Phase semantic constraints ──────────────────────────────────────────────
# Per-phase rules about what gripper actions are safe.
# Positive gripper = close, negative = open.

_PHASE_GRIPPER_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "reach_pick":       {"gripper_allowed": "open_only", "max_open": 0.0},
    "descend_align":    {"gripper_allowed": "open_only", "max_open": 0.0},
    "micro_align":      {"gripper_allowed": "open_only", "max_open": 0.0},
    "grasp_close":      {"gripper_allowed": "close_only", "min_close": -0.5},
    "lift_verify":      {"gripper_allowed": "close_only", "min_close": -0.5},
    "transport":        {"gripper_allowed": "close_only", "min_close": -0.5},
    "descend_place":    {"gripper_allowed": "close_only", "min_close": -0.5},
    "micro_place":      {"gripper_allowed": "close_only", "min_close": -0.5},
    "release_verify":   {"gripper_allowed": "open_only", "max_open": 0.0},
    "retreat":          {"gripper_allowed": "any", "max_open": 0.0},
}


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of safety gate validation."""

    approved: bool
    action: np.ndarray               # The action to execute (VLA or scripted)
    reason: str                      # "approved" or rejection reason code
    vla_confidence: float = 1.0
    deviation_from_scripted: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "vla_confidence": self.vla_confidence,
            "deviation_from_scripted": self.deviation_from_scripted,
        }


@dataclass
class GateConfig:
    """Configuration for the NeuroSymbolicSafetyGate."""

    vla_confidence_threshold: float = 0.6
    use_confidence: bool = False  # disabled by default: VLA confidence not yet calibrated
    max_xyz_deviation: float = 0.02         # max L2 deviation from scripted (meters)
    max_gripper_deviation: float = 0.5      # max absolute gripper delta
    workspace_x_bounds: tuple[float, float] = (-0.5, 0.5)
    workspace_y_bounds: tuple[float, float] = (-0.5, 0.5)
    workspace_z_bounds: tuple[float, float] = (0.75, 1.5)
    table_z: float = 0.80
    enable_phase_constraints: bool = True
    enable_workspace_check: bool = True
    enable_deviation_check: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "vla_confidence_threshold": self.vla_confidence_threshold,
            "max_xyz_deviation": self.max_xyz_deviation,
            "max_gripper_deviation": self.max_gripper_deviation,
            "workspace_x_bounds": list(self.workspace_x_bounds),
            "workspace_y_bounds": list(self.workspace_y_bounds),
            "workspace_z_bounds": list(self.workspace_z_bounds),
            "table_z": self.table_z,
            "enable_phase_constraints": self.enable_phase_constraints,
            "enable_workspace_check": self.enable_workspace_check,
            "enable_deviation_check": self.enable_deviation_check,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateConfig":
        return cls(
            vla_confidence_threshold=float(data.get("vla_confidence_threshold", 0.6)),
            max_xyz_deviation=float(data.get("max_xyz_deviation", 0.02)),
            max_gripper_deviation=float(data.get("max_gripper_deviation", 0.5)),
            workspace_x_bounds=tuple(data.get("workspace_x_bounds", (-0.5, 0.5))),
            workspace_y_bounds=tuple(data.get("workspace_y_bounds", (-0.5, 0.5))),
            workspace_z_bounds=tuple(data.get("workspace_z_bounds", (0.75, 1.5))),
            table_z=float(data.get("table_z", 0.80)),
            enable_phase_constraints=bool(data.get("enable_phase_constraints", True)),
            enable_workspace_check=bool(data.get("enable_workspace_check", True)),
            enable_deviation_check=bool(data.get("enable_deviation_check", True)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "GateConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ── Safety Gate ─────────────────────────────────────────────────────────────

class NeuroSymbolicSafetyGate:
    """Validates VLA-proposed actions against geometric and symbolic constraints.

    Usage::

        gate = NeuroSymbolicSafetyGate(GateConfig())
        result = gate.validate(
            vla_action=vla_action_7d,
            scripted_action=scripted_action_7d,
            vla_confidence=0.85,
            eef_pos=current_eef_xyz,
            phase="grasp_close",
        )
        if result.approved:
            env.step(result.action)
        else:
            env.step(scripted_action)  # fallback already in result.action
    """

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()

    # ── Main validate method ─────────────────────────────────────────────

    def validate(
        self,
        vla_action: np.ndarray,
        scripted_action: np.ndarray,
        vla_confidence: float,
        eef_pos: np.ndarray,
        phase: str,
        symbolic_constraints: dict[str, Any] | None = None,
    ) -> GateResult:
        """Validate a VLA action against safety constraints.

        Args:
            vla_action: VLA-proposed action, shape (7,) [dx,dy,dz,0,0,0,gripper].
            scripted_action: Scripted controller action for fallback, shape (7,).
            vla_confidence: VLA policy confidence score in [0, 1].
            eef_pos: Current end-effector position, shape (3,).
            phase: Current execution phase name.
            symbolic_constraints: Optional additional constraints dict.

        Returns:
            GateResult with approved flag and the action to execute.
        """
        # 1. Non-finite check (always enforced)
        if not np.all(np.isfinite(vla_action)):
            return GateResult(
                approved=False,
                action=self._safe_action(scripted_action),
                reason="non_finite_action",
                vla_confidence=vla_confidence,
            )

        # 2. VLA confidence check (only when enabled and calibrated)
        if self.config.use_confidence and vla_confidence < self.config.vla_confidence_threshold:
            return GateResult(
                approved=False,
                action=self._safe_action(scripted_action),
                reason="vla_low_confidence",
                vla_confidence=vla_confidence,
            )

        # 3. Action deviation check (vs scripted prior)
        xyz_dev = float(np.linalg.norm(vla_action[:3] - scripted_action[:3]))
        grip_dev = float(abs(vla_action[6] - scripted_action[6]))
        if self.config.enable_deviation_check:
            if xyz_dev > self.config.max_xyz_deviation:
                return GateResult(
                    approved=False,
                    action=self._safe_action(scripted_action),
                    reason="action_xyz_too_deviant",
                    vla_confidence=vla_confidence,
                    deviation_from_scripted=xyz_dev,
                )
            if grip_dev > self.config.max_gripper_deviation:
                return GateResult(
                    approved=False,
                    action=self._safe_action(scripted_action),
                    reason="action_gripper_too_deviant",
                    vla_confidence=vla_confidence,
                    deviation_from_scripted=grip_dev,
                )

        # 4. Workspace bounds check
        if self.config.enable_workspace_check:
            new_xyz = eef_pos + vla_action[:3]
            if not self._in_workspace(new_xyz):
                return GateResult(
                    approved=False,
                    action=self._safe_action(scripted_action),
                    reason="workspace_violation",
                    vla_confidence=vla_confidence,
                    deviation_from_scripted=xyz_dev,
                )

        # 5. Phase semantic constraints
        if self.config.enable_phase_constraints:
            constraint = _PHASE_GRIPPER_CONSTRAINTS.get(phase, {})
            if constraint:
                gripper_cmd = float(vla_action[6])
                allowed = constraint.get("gripper_allowed", "any")
                if allowed == "open_only" and gripper_cmd > 0.0:
                    return GateResult(
                        approved=False,
                        action=self._safe_action(scripted_action),
                        reason="phase_semantic_violation_gripper_open_required",
                        vla_confidence=vla_confidence,
                    )
                if allowed == "close_only" and gripper_cmd < 0.0:
                    return GateResult(
                        approved=False,
                        action=self._safe_action(scripted_action),
                        reason="phase_semantic_violation_gripper_close_required",
                        vla_confidence=vla_confidence,
                    )

        # All checks passed
        return GateResult(
            approved=True,
            action=self._safe_action(vla_action),
            reason="approved",
            vla_confidence=vla_confidence,
            deviation_from_scripted=xyz_dev,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _in_workspace(self, xyz: np.ndarray) -> bool:
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return (
            self.config.workspace_x_bounds[0] <= x <= self.config.workspace_x_bounds[1]
            and self.config.workspace_y_bounds[0] <= y <= self.config.workspace_y_bounds[1]
            and self.config.workspace_z_bounds[0] <= z <= self.config.workspace_z_bounds[1]
        )

    @staticmethod
    def _safe_action(action: np.ndarray) -> np.ndarray:
        """Ensure action is safe even if input has issues."""
        action = np.asarray(action, dtype=np.float32).reshape(7)
        action[3:6] = 0.0  # zero rotation
        if not np.all(np.isfinite(action)):
            return np.zeros(7, dtype=np.float32)
        return np.clip(action, -1.0, 1.0)


# ── Gate statistics collector ───────────────────────────────────────────────

class GateStatistics:
    """Collects gate rejection statistics for monitoring VLA performance.

    Usage::

        stats = GateStatistics()
        for step in episode:
            result = gate.validate(...)
            stats.record(result)
        print(stats.summary())
    """

    def __init__(self):
        self.total_checks: int = 0
        self.approved: int = 0
        self.rejected: int = 0
        self.rejection_reasons: dict[str, int] = {}
        self.vla_confidences: list[float] = []
        self.deviations: list[float] = []

    def record(self, result: GateResult) -> None:
        self.total_checks += 1
        if result.approved:
            self.approved += 1
        else:
            self.rejected += 1
            reason = result.reason
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        self.vla_confidences.append(result.vla_confidence)
        self.deviations.append(result.deviation_from_scripted)

    @property
    def acceptance_ratio(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.approved / self.total_checks

    def summary(self) -> dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "approved": self.approved,
            "rejected": self.rejected,
            "acceptance_ratio": self.acceptance_ratio,
            "rejection_reasons": dict(self.rejection_reasons),
            "mean_confidence": float(np.mean(self.vla_confidences)) if self.vla_confidences else 0.0,
            "mean_deviation": float(np.mean(self.deviations)) if self.deviations else 0.0,
        }

    def reset(self) -> None:
        self.__init__()
