"""Repair Proposer — generates structured repair proposals from failure summaries.

Two modes:
  1. Hand-coded baseline: rule-based repair suggestions (deterministic, verifiable)
  2. LLM experimental: Qwen-suggested repairs, filtered through RepairVerifier

Usage:
    proposer = RepairProposer(verifier_bank)
    proposal = proposer.propose(summary, scene_graph, mode="hand_coded")
    # proposal: RepairProposal with modified_option, parameter_patch, retry_budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_nsm.neuro_symbolic.failure_summarizer import FailureSummary


@dataclass
class RepairProposal:
    """Structured repair suggestion.

    Attributes:
        diagnosis: What went wrong.
        repair_type: "retry_grasp" | "regrasp" | "micro_place" | "replan" | "none"
        modified_option: Which skill to modify (e.g. "grasp").
        parameter_patch: Parameter changes (e.g. {"approach_height_delta": 0.02}).
        retry_budget: How many extra retries this repair costs.
        required_verification: Which verifiers must re-run.
        source: "hand_coded" | "llm_proposed".
    """

    diagnosis: str = ""
    repair_type: str = "retry"
    modified_option: str = ""
    parameter_patch: dict[str, Any] = field(default_factory=dict)
    retry_budget: int = 1
    required_verification: list[str] = field(default_factory=list)
    source: str = "hand_coded"

    def to_dict(self) -> dict:
        return {
            "diagnosis": self.diagnosis,
            "repair_type": self.repair_type,
            "modified_option": self.modified_option,
            "parameter_patch": self.parameter_patch,
            "retry_budget": self.retry_budget,
            "required_verification": self.required_verification,
            "source": self.source,
        }


class RepairProposer:
    """Propose repairs from failure summaries.

    Args:
        verifier_bank: VerifierBank for validating LLM-proposed repairs.
        llm_parser: Optional Qwen parser for LLM experimental mode.
    """

    def __init__(self, verifier_bank=None, llm_parser=None):
        self.verifier_bank = verifier_bank
        self.llm_parser = llm_parser

    # ── Hand-coded repair rules ──────────────────────────────────────────

    _HAND_CODED_RULES: dict[str, dict] = {
        # grasp failures
        "nocontact": {
            "repair_type": "retry_grasp",
            "modified_option": "grasp",
            "parameter_patch": {"approach_height_delta": 0.01, "descend_offset_delta": 0.003},
            "retry_budget": 1,
        },
        "xy_misalign": {
            "repair_type": "retry_grasp",
            "modified_option": "grasp",
            "parameter_patch": {"xy_realign": True, "micro_align_steps": 60},
            "retry_budget": 1,
        },
        "slip": {
            "repair_type": "regrasp",
            "modified_option": "grasp",
            "parameter_patch": {"close_steps_extra": 20, "holding_check_required": True},
            "retry_budget": 2,
        },
        "insufficient_grip": {
            "repair_type": "regrasp",
            "modified_option": "grasp",
            "parameter_patch": {"close_steps_extra": 30},
            "retry_budget": 2,
        },
        # place failures
        "transport_drop": {
            "repair_type": "regrasp_and_replan",
            "modified_option": "grasp",
            "parameter_patch": {"holding_check_enabled": True},
            "retry_budget": 3,
        },
        "preplace": {
            "repair_type": "regrasp_and_replan",
            "modified_option": "grasp",
            "parameter_patch": {},
            "retry_budget": 3,
        },
        "placement_inaccurate": {
            "repair_type": "micro_place",
            "modified_option": "release_place",
            "parameter_patch": {"place_mode": "safe", "micro_place_enabled": True},
            "retry_budget": 1,
        },
        "verify_failed": {
            "repair_type": "micro_place",
            "modified_option": "release_place",
            "parameter_patch": {"place_mode": "safe"},
            "retry_budget": 1,
        },
        # reach/approach failures
        "approach_blocked": {
            "repair_type": "xy_realign",
            "modified_option": "reach",
            "parameter_patch": {"approach_height_delta": 0.03, "xy_tolerance": 0.005},
            "retry_budget": 1,
        },
        "reach_failed": {
            "repair_type": "xy_realign",
            "modified_option": "reach",
            "parameter_patch": {"max_steps_extra": 300},
            "retry_budget": 1,
        },
        # timeout
        "slow_convergence": {
            "repair_type": "retry_with_params",
            "modified_option": "",
            "parameter_patch": {"max_steps_extra": 500},
            "retry_budget": 2,
        },
        "timeout": {
            "repair_type": "retry_with_params",
            "modified_option": "",
            "parameter_patch": {"max_steps_extra": 300},
            "retry_budget": 1,
        },
        # catch-all: unknown failure should not be silently retried
        "unknown": {
            "repair_type": "investigate",
            "modified_option": "",
            "parameter_patch": {"log_full_trace": True},
            "retry_budget": 0,
        },
    }

    def propose_hand_coded(self, summary: FailureSummary) -> RepairProposal:
        """Generate repair from hand-coded rules.

        Args:
            summary: FailureSummary from FailureSummarizer.

        Returns:
            RepairProposal with deterministic repair suggestion.
        """
        # Try exact match first, then root_cause match, then catch-all
        rule = (
            self._HAND_CODED_RULES.get(summary.root_cause) or
            self._HAND_CODED_RULES.get(summary.recommended_repair_type) or
            self._HAND_CODED_RULES.get("unknown")
        )
        assert rule is not None  # unknown always exists

        # Apply parameter overrides from evidence
        param_patch = dict(rule["parameter_patch"])
        modified_option = rule["modified_option"] or summary.affected_skill

        return RepairProposal(
            diagnosis=summary.diagnosis,
            repair_type=rule["repair_type"],
            modified_option=modified_option,
            parameter_patch=param_patch,
            retry_budget=rule["retry_budget"],
            required_verification=["RepairVerifier", "OptionContractVerifier"],
            source="hand_coded",
        )

    def propose_llm(self, summary: FailureSummary,
                    scene_graph_text: str = "") -> RepairProposal | None:
        """Generate repair using LLM (experimental path).

        The LLM output is filtered through RepairVerifier before acceptance.

        Args:
            summary: FailureSummary.
            scene_graph_text: PredicateSceneGraph in prompt text format.

        Returns:
            RepairProposal if LLM output passes verification, None otherwise.
        """
        if self.llm_parser is None or self.verifier_bank is None:
            return None

        prompt = self._build_repair_prompt(summary, scene_graph_text)
        try:
            parse_result = self.llm_parser.parse(
                prompt, [], task_type="repair")
        except Exception:
            return None

        if not parse_result.success:
            return None

        try:
            import json
            data = json.loads(parse_result.raw_output) if isinstance(
                parse_result.raw_output, str) else parse_result.raw_output
        except Exception:
            return None

        proposal = RepairProposal(
            diagnosis=data.get("diagnosis", summary.diagnosis),
            repair_type=data.get("repair_type", "retry"),
            modified_option=data.get("modified_option", summary.affected_skill),
            parameter_patch=data.get("parameter_patch", {}),
            retry_budget=data.get("retry_budget", 1),
            required_verification=["RepairVerifier"],
            source="llm_proposed",
        )

        # Filter through verifier
        if self.verifier_bank:
            report = self.verifier_bank.verify_all(
                repair_proposal=proposal.to_dict(),
            )
            if not report.passed:
                return None

        return proposal

    def propose(self, summary: FailureSummary,
                scene_graph_text: str = "",
                mode: str = "hand_coded") -> RepairProposal | None:
        """Generate repair proposal.

        Args:
            summary: FailureSummary.
            scene_graph_text: Scene graph text (needed for LLM path).
            mode: "hand_coded" (default) or "llm" (experimental).

        Returns:
            RepairProposal or None if repair not possible.
        """
        if summary.severity == "likely_fatal":
            return None

        if mode == "llm":
            llm_result = self.propose_llm(summary, scene_graph_text)
            if llm_result is not None:
                return llm_result
            # fall through to hand-coded if LLM fails

        return self.propose_hand_coded(summary)

    def _build_repair_prompt(self, summary: FailureSummary,
                             scene_graph_text: str = "") -> str:
        """Build LLM prompt for repair suggestion."""
        lines = [
            "You are a robot repair planner. Given a failure, propose a repair.",
            "",
            "Failure:",
            f"  Skill: {summary.affected_skill}",
            f"  Root cause: {summary.root_cause}",
            f"  Diagnosis: {summary.diagnosis}",
            f"  Evidence: {', '.join(summary.evidence)}",
            "",
        ]
        if scene_graph_text:
            lines.append(f"Scene:\n{scene_graph_text}\n")

        lines.extend([
            "Output a JSON repair proposal:",
            '{',
            '  "diagnosis": "string",',
            '  "repair_type": "retry_grasp|regrasp|micro_place|replan|xy_realign|retry_with_params",',
            '  "modified_option": "reach|grasp|lift|reach_place|release_place|retreat",',
            '  "parameter_patch": {"key": value},',
            '  "retry_budget": 1-3',
            '}',
            "",
            "Rules:",
            "- repair_type must be one of the listed values",
            "- modified_option must be a valid skill name",
            "- retry_budget must be 1-3",
            "- Only output JSON, no explanation",
        ])
        return "\n".join(lines)
