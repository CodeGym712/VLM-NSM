"""Tests for FailureSummarizer + RepairProposer — Commit 6."""

from vlm_nsm.neuro_symbolic.failure_summarizer import (
    FailureSummarizer, FailureSummary,
)
from vlm_nsm.neuro_symbolic.repair_proposer import (
    RepairProposer, RepairProposal,
)


class TestFailureSummarizer:
    def setup_method(self):
        self.summarizer = FailureSummarizer()

    def test_empty_trace(self):
        s = self.summarizer.summarize([])
        assert s.severity == "likely_fatal"
        assert "Empty trace" in s.diagnosis

    def test_nocontact_failure(self):
        """Grasp failure: gripper closed without contacting object."""
        trace = [type("R", (), {
            "success": False, "failure_type": "nocontact",
            "skill_name": "grasp", "effect_satisfied": False,
            "pre_predicates": [["handempty"], ["clear", "cube1"]],
            "post_predicates": [["handempty"]],  # still empty = didn't grasp
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert s.root_cause == "xy_misalign"
        assert s.affected_skill == "grasp"
        assert s.severity == "recoverable"
        assert s.recommended_repair_type == "retry_grasp"

    def test_slip_failure(self):
        trace = [type("R", (), {
            "success": False, "failure_type": "slip",
            "skill_name": "grasp", "effect_satisfied": False,
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert s.root_cause == "insufficient_grip"

    def test_timeout_failure(self):
        trace = [type("R", (), {
            "success": False, "failure_type": "timeout",
            "skill_name": "reach",
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert s.root_cause == "slow_convergence"
        assert s.severity == "likely_recoverable"

    def test_env_error_is_fatal(self):
        trace = [type("R", (), {
            "success": False, "failure_type": "env_error",
            "skill_name": "reach",
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert s.severity == "likely_fatal"

    def test_transport_drop(self):
        trace = [type("R", (), {
            "success": False, "failure_type": "preplace",
            "skill_name": "reach_place",
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert s.root_cause == "transport_drop"
        assert s.recommended_repair_type == "regrasp_and_replan"

    def test_unknown_failure(self):
        trace = [type("R", (), {
            "success": False, "failure_type": "weird_never_seen_before",
            "skill_name": "unknown_skill",
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert s.root_cause == "unknown"
        assert s.recommended_repair_type == "retry"

    def test_from_dict_trace(self):
        trace_dicts = [
            {"success": True, "failure_type": None, "skill_name": "reach"},
            {"success": False, "failure_type": "nocontact", "skill_name": "grasp"},
        ]
        s = self.summarizer.summarize_from_dict(trace_dicts, 1)
        assert s.root_cause == "xy_misalign"

    def test_effect_satisfied_in_evidence(self):
        """Even if effect OK, downstream skill might have failed."""
        trace = [type("R", (), {
            "success": False, "failure_type": "preplace",
            "skill_name": "reach_place", "effect_satisfied": True,
            "pre_predicates": [["holding", "cube1"]],
            "post_predicates": [["handempty"]],  # lost holding!
        })()]
        s = self.summarizer.summarize(trace, 0)
        assert "Lost predicates" in str(s.evidence) or \
               any("Lost" in e for e in s.evidence)


class TestRepairProposer:
    def setup_method(self):
        self.proposer = RepairProposer()

    def test_hand_coded_nocontact(self):
        summary = FailureSummary(
            diagnosis="Gripper closed without object",
            affected_skill="grasp",
            root_cause="nocontact",
            severity="recoverable",
        )
        p = self.proposer.propose_hand_coded(summary)
        assert p is not None
        assert p.repair_type == "retry_grasp"
        assert p.modified_option == "grasp"
        assert p.source == "hand_coded"
        assert p.retry_budget == 1
        assert "RepairVerifier" in p.required_verification

    def test_hand_coded_transport_drop(self):
        summary = FailureSummary(
            diagnosis="Object dropped during transport",
            affected_skill="reach_place",
            root_cause="transport_drop",
            severity="recoverable",
        )
        p = self.proposer.propose_hand_coded(summary)
        assert p.retry_budget == 3
        assert p.repair_type == "regrasp_and_replan"

    def test_hand_coded_fatal_returns_none(self):
        summary = FailureSummary(
            diagnosis="Env terminated",
            affected_skill="reach",
            root_cause="env_terminated",
            severity="likely_fatal",
        )
        p = self.proposer.propose(summary, mode="hand_coded")
        assert p is None

    def test_hand_coded_unknown(self):
        summary = FailureSummary(
            diagnosis="Weird failure",
            affected_skill="unknown",
            root_cause="unknown",
            severity="recoverable",
        )
        p = self.proposer.propose_hand_coded(summary)
        assert p is not None
        assert p.repair_type == "investigate"  # unknown → investigate, not blind retry
        assert p.retry_budget == 0              # no retry for unknown failures

    def test_repair_proposal_to_dict(self):
        p = RepairProposal(
            diagnosis="test", repair_type="retry_grasp",
            modified_option="grasp", retry_budget=2,
            source="hand_coded",
        )
        d = p.to_dict()
        assert d["repair_type"] == "retry_grasp"
        assert d["source"] == "hand_coded"

    def test_llm_mode_falls_back_to_hand_coded_when_no_parser(self):
        """Without an LLM parser, llm mode should fall back to hand-coded."""
        summary = FailureSummary(
            diagnosis="Grasp failed",
            affected_skill="grasp",
            root_cause="nocontact",
            severity="recoverable",
        )
        p = self.proposer.propose(summary, mode="llm")
        assert p is not None
        assert p.source == "hand_coded"  # fell back
