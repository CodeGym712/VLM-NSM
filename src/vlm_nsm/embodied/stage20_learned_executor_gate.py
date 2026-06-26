"""Stage 20-U3: Conservative fallback gate for learned executor.

Checks learned actions for safety before allowing execution.
Returns gate decision and reason for every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GateConfig:
    """Configuration for the fallback gate."""
    action_norm_max: float = 3.0          # Max L2 norm of action vector
    action_dim6_min: float = -1.5         # Min gripper action
    action_dim6_max: float = 1.5          # Max gripper action
    action_dim_abs_max: float = 2.0       # Max absolute value of any dim
    require_warm: bool = True             # Require history buffer warm
    require_finite: bool = True           # Require all action dims finite
    # Per-phase overrides
    phase_action_norm_max: dict[int, float] = field(default_factory=lambda: {
        1: 2.5,  # grasp: tighter norm
    })
    phase_dim6_max: dict[int, float] = field(default_factory=lambda: {
        1: 1.2,  # grasp: tighter gripper range
    })
    # High-risk mode: extra checks for direction1/phase1
    high_risk_dim6_abs_max: float = 1.8


@dataclass
class GateResult:
    """Result of gate check for a single step."""
    passed: bool
    reason: str = ""
    action_norm: float = 0.0
    action_dim6: float = 0.0
    action_max_abs: float = 0.0
    is_warm: bool = False
    is_finite: bool = True


class LearnedExecutorGate:
    """Conservative gate for learned executor actions.

    Checks include:
    - History buffer warm (>= history_len observations)
    - All action dimensions finite
    - Action L2 norm within configured max
    - Action dim6 (gripper) within configured range
    - Per-dimension absolute max check
    - Per-phase overrides for tighter constraints
    """

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        self._trigger_counts: dict[str, int] = {}
        self._total_checks = 0

    def check(
        self,
        action: np.ndarray,
        phase_id: int,
        direction_id: int = -1,
        is_warm: bool = True,
    ) -> GateResult:
        """Check whether a learned action passes the safety gate.

        Args:
            action: 7-dim action vector.
            phase_id: Current phase ID (0-5).
            direction_id: Direction ID (0=forward, 1=reverse, -1=unknown).
            is_warm: Whether history buffer has enough observations.

        Returns:
            GateResult with pass/fail and reason.
        """
        self._total_checks += 1
        action = np.asarray(action, dtype=np.float32).ravel()

        action_norm = float(np.linalg.norm(action))
        action_dim6 = float(action[6])
        action_max_abs = float(np.max(np.abs(action)))
        is_finite = bool(np.isfinite(action).all())

        # Check 1: History warm
        if self.config.require_warm and not is_warm:
            return self._fail("history_not_warm", action_norm, action_dim6, action_max_abs, is_warm, is_finite)

        # Check 2: Finite
        if self.config.require_finite and not is_finite:
            return self._fail("non_finite_action", action_norm, action_dim6, action_max_abs, is_warm, is_finite)

        # Check 3: Per-dim absolute max
        if action_max_abs > self.config.action_dim_abs_max:
            return self._fail("action_dim_abs_max_exceeded", action_norm, action_dim6, action_max_abs, is_warm, is_finite)

        # Check 4: Action norm (with phase override)
        norm_max = self.config.phase_action_norm_max.get(phase_id, self.config.action_norm_max)
        if action_norm > norm_max:
            return self._fail("action_norm_max_exceeded", action_norm, action_dim6, action_max_abs, is_warm, is_finite)

        # Check 5: Dim6 range (with phase override)
        dim6_max = self.config.phase_dim6_max.get(phase_id, self.config.action_dim6_max)
        if abs(action_dim6) > dim6_max:
            return self._fail("action_dim6_out_of_range", action_norm, action_dim6, action_max_abs, is_warm, is_finite)

        # Check 6: High-risk mode for direction1/phase1
        if direction_id == 1 and phase_id == 1:
            if abs(action_dim6) > self.config.high_risk_dim6_abs_max:
                return self._fail("high_risk_d1p1_dim6", action_norm, action_dim6, action_max_abs, is_warm, is_finite)

        # All checks passed
        self._trigger_counts["passed"] = self._trigger_counts.get("passed", 0) + 1
        return GateResult(
            passed=True, reason="ok",
            action_norm=action_norm, action_dim6=action_dim6,
            action_max_abs=action_max_abs, is_warm=is_warm, is_finite=is_finite,
        )

    def _fail(self, reason: str, norm: float, dim6: float, max_abs: float,
              warm: bool, finite: bool) -> GateResult:
        self._trigger_counts[reason] = self._trigger_counts.get(reason, 0) + 1
        return GateResult(
            passed=False, reason=reason,
            action_norm=norm, action_dim6=dim6,
            action_max_abs=max_abs, is_warm=warm, is_finite=finite,
        )

    @property
    def trigger_counts(self) -> dict[str, int]:
        return dict(self._trigger_counts)

    @property
    def total_checks(self) -> int:
        return self._total_checks

    def reset_counts(self):
        self._trigger_counts.clear()
        self._total_checks = 0
