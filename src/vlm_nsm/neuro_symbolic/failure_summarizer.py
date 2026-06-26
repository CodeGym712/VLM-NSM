"""Failure Summarizer — converts OptionResult trace into structured diagnosis.

Input: list of SkillResult (with pre/post predicates + failure_type)
Output: FailureSummary { diagnosis, affected_skill, root_cause, evidence }

Used by RepairProposer to decide on recovery strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureSummary:
    """Structured diagnosis from a failed execution trace.

    Attributes:
        diagnosis: Human-readable diagnosis (e.g. "grasp missed object").
        affected_skill: Which skill failed.
        root_cause: Classified root cause (e.g. "xy_misalign", "object_slipped").
        evidence: List of observations supporting the diagnosis.
        failed_contract: Contract name that was violated (if any).
        missing_effects: Expected effects that were not achieved.
        severity: "recoverable" | "likely_recoverable" | "likely_fatal".
        recommended_repair_type: Suggested repair strategy.
    """

    diagnosis: str = ""
    affected_skill: str = ""
    root_cause: str = ""
    evidence: list[str] = field(default_factory=list)
    failed_contract: str = ""
    missing_effects: list[list[str]] = field(default_factory=list)
    severity: str = "recoverable"
    recommended_repair_type: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "diagnosis": self.diagnosis,
            "affected_skill": self.affected_skill,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "failed_contract": self.failed_contract,
            "missing_effects": self.missing_effects,
            "severity": self.severity,
            "recommended_repair_type": self.recommended_repair_type,
            "metadata": self.metadata,
        }


class FailureSummarizer:
    """Analyze execution trace and produce FailureSummary.

    Usage:
        summarizer = FailureSummarizer()
        trace = [skill_result1, skill_result2, ...]  # from BTTrace or TaskRunner
        summary = summarizer.summarize(trace, failed_step_index=2)
    """

    # Known failure patterns and their root causes
    _FAILURE_PATTERNS: dict[str, dict] = {
        "nocontact": {
            "root_cause": "xy_misalign",
            "diagnosis": "Gripper closed without contacting object — likely XY misalignment",
            "severity": "recoverable",
            "repair_type": "retry_grasp",
        },
        "slip": {
            "root_cause": "insufficient_grip",
            "diagnosis": "Object slipped from gripper during lift",
            "severity": "recoverable",
            "repair_type": "regrasp",
        },
        "preplace": {
            "root_cause": "transport_drop",
            "diagnosis": "Object dropped during transport — not holding at preplace",
            "severity": "recoverable",
            "repair_type": "regrasp_and_replan",
        },
        "verify_failed": {
            "root_cause": "placement_inaccurate",
            "diagnosis": "Object placed but not on target — placement inaccurate",
            "severity": "recoverable",
            "repair_type": "micro_place",
        },
        "timeout": {
            "root_cause": "slow_convergence",
            "diagnosis": "Skill timed out — PID did not converge within step limit",
            "severity": "likely_recoverable",
            "repair_type": "retry_with_params",
        },
        "env_error": {
            "root_cause": "env_terminated",
            "diagnosis": "Environment terminated — horizon exhausted or sim error",
            "severity": "likely_fatal",
            "repair_type": "none",
        },
        "reach_failed": {
            "root_cause": "approach_blocked",
            "diagnosis": "Cannot reach target — approach path blocked or too far",
            "severity": "recoverable",
            "repair_type": "xy_realign",
        },
        "grasp_failed": {
            "root_cause": "grasp_error",
            "diagnosis": "Grasp failed — see nested failure_type for details",
            "severity": "recoverable",
            "repair_type": "retry_grasp",
        },
    }

    def summarize(self, trace: list[Any], failed_step_index: int = -1) -> FailureSummary:
        """Analyze trace and produce structured diagnosis.

        Args:
            trace: List of SkillResult from execution.
            failed_step_index: Index of the failed step (default: last).

        Returns:
            FailureSummary with diagnosis and repair recommendation.
        """
        if not trace:
            return FailureSummary(
                diagnosis="Empty trace — no execution data",
                severity="likely_fatal",
            )

        # Find the failed step
        if failed_step_index < 0:
            failed_step_index = len(trace) - 1

        failed = trace[failed_step_index] if 0 <= failed_step_index < len(trace) else None
        if failed is None:
            return FailureSummary(diagnosis="No failed step found", severity="likely_fatal")

        # Extract failure info
        ft = getattr(failed, 'failure_type', None) or "unknown"
        skill_name = getattr(failed, 'skill_name', '') or getattr(failed, 'name', 'unknown')

        # Build evidence
        evidence = [f"Step {failed_step_index} ({skill_name}) failed with: {ft}"]

        # Check effect satisfaction
        effect_ok = getattr(failed, 'effect_satisfied', None)
        if effect_ok is False:
            evidence.append("Expected effects were NOT achieved")
        elif effect_ok is True:
            evidence.append("Expected effects were achieved (failure may be downstream)")

        # Check pre/post predicates
        pre = getattr(failed, 'pre_predicates', []) or []
        post = getattr(failed, 'post_predicates', []) or []
        if pre and post:
            pre_set = {tuple(p) if hasattr(p, '__iter__') else (p,) for p in pre}
            post_set = {tuple(p) if hasattr(p, '__iter__') else (p,) for p in post}
            lost = pre_set - post_set
            gained = post_set - pre_set
            if lost:
                evidence.append(f"Lost predicates: {list(lost)[:3]}")
            if gained:
                evidence.append(f"Gained predicates: {list(gained)[:3]}")

        # Match failure pattern
        pattern = self._FAILURE_PATTERNS.get(ft)
        if pattern is None:
            # Try partial match
            for key, pat in self._FAILURE_PATTERNS.items():
                if key in ft:
                    pattern = pat
                    break

        if pattern is None:
            pattern = {
                "root_cause": "unknown",
                "diagnosis": f"Unclassified failure: {ft}",
                "severity": "likely_recoverable",
                "repair_type": "retry",
            }

        return FailureSummary(
            diagnosis=pattern["diagnosis"],
            affected_skill=skill_name,
            root_cause=pattern["root_cause"],
            evidence=evidence,
            severity=pattern["severity"],
            recommended_repair_type=pattern["repair_type"],
            metadata={
                "failed_step_index": failed_step_index,
                "n_steps": len(trace),
                "raw_failure_type": ft,
            },
        )

    def summarize_from_dict(self, trace_dicts: list[dict],
                            failed_step_index: int = -1) -> FailureSummary:
        """Summarize from dict-format trace (JSON compatible).

        Args:
            trace_dicts: List of dicts with 'success', 'failure_type', 'skill_name'.
            failed_step_index: Index of failed step.

        Returns:
            FailureSummary.
        """
        # Convert to mock SkillResult objects
        class MockResult:
            def __init__(self, d):
                self.success = d.get("success", False)
                self.failure_type = d.get("failure_type")
                self.skill_name = d.get("skill_name", "")
                self.effect_satisfied = d.get("effect_satisfied")
                self.pre_predicates = d.get("pre_predicates", [])
                self.post_predicates = d.get("post_predicates", [])

        trace = [MockResult(d) for d in trace_dicts]
        return self.summarize(trace, failed_step_index)
