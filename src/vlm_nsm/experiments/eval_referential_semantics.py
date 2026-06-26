from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_nsm.core.utils import ensure_dir


def load_eval_results(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval results not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in data:
        raise ValueError(f"Eval results missing 'results' key: {path}")
    return data


def load_audit(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audit file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    instructions = data.get("referential_instructions", [])
    if not instructions:
        raise ValueError(f"Audit file missing 'referential_instructions' key: {path}")
    return instructions


def build_audit_lookup(audit_entries: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for entry in audit_entries:
        text = entry.get("instruction_text", "")
        if text:
            lookup[text] = entry
    return lookup


def _bindings_equal(a: dict | None, b: dict | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


def classify_result(result: dict, audit_entry: dict | None) -> dict:
    instruction = result.get("instruction", "")
    predicted = result.get("predicted_bindings")
    expected = result.get("expected_bindings")
    system_status = result.get("status")
    system_is_ungroundable = system_status == "ungroundable" or result.get("failure_type") == "ungroundable_object"

    classification: dict[str, Any] = {
        "instruction": instruction,
        "has_audit_entry": audit_entry is not None,
        "system_is_ungroundable": system_is_ungroundable,
    }

    if audit_entry is None:
        classification["semantic_validity"] = "non_referential"
        classification["reference_type"] = None
        classification["recommended_expected_status"] = None
        classification["binding_correct"] = None if system_is_ungroundable else _bindings_equal(predicted, expected)
        classification["correct_outcome"] = classification["binding_correct"] if not system_is_ungroundable else None
        classification["unsafe_guess"] = False
        classification["over_rejection"] = system_is_ungroundable
        classification["evaluator_mismatch"] = False
        return classification

    sem_validity = audit_entry.get("semantic_validity", "unknown")
    ref_type = audit_entry.get("reference_type", "unknown")
    recommended_status = audit_entry.get("recommended_expected_status", "grounded")
    should_be_groundable = recommended_status == "grounded"

    classification["semantic_validity"] = sem_validity
    classification["reference_type"] = ref_type
    classification["recommended_expected_status"] = recommended_status
    classification["instruction_id"] = audit_entry.get("instruction_id")

    if should_be_groundable:
        # System should produce correct binding
        if system_is_ungroundable:
            classification["binding_correct"] = False
            classification["correct_outcome"] = False
            classification["unsafe_guess"] = False
            classification["over_rejection"] = True
            classification["evaluator_mismatch"] = False
        else:
            binding_ok = _bindings_equal(predicted, expected)
            classification["binding_correct"] = binding_ok
            classification["correct_outcome"] = binding_ok
            classification["unsafe_guess"] = False
            classification["over_rejection"] = False
            classification["evaluator_mismatch"] = False
    else:
        # System should output ungroundable
        if system_is_ungroundable:
            classification["binding_correct"] = None
            classification["correct_outcome"] = True
            classification["unsafe_guess"] = False
            classification["over_rejection"] = False
            classification["evaluator_mismatch"] = False
        else:
            # System attempted a binding when it shouldn't have
            binding_ok = _bindings_equal(predicted, expected)
            classification["binding_correct"] = binding_ok
            classification["correct_outcome"] = False  # should have rejected
            classification["unsafe_guess"] = True
            classification["over_rejection"] = False
            classification["evaluator_mismatch"] = False

            # Check evaluator mismatch: if expected is not None but the ref is ambiguous
            if sem_validity == "evaluator_mismatch":
                classification["evaluator_mismatch"] = True

    return classification


def compute_semantic_metrics(classifications: list[dict]) -> dict:
    referential = [c for c in classifications if c.get("has_audit_entry")]
    non_referential = [c for c in classifications if not c.get("has_audit_entry")]

    groundable = [c for c in referential if c.get("semantic_validity") == "objectively_groundable"]
    ambiguous = [c for c in referential if c.get("semantic_validity") == "objectively_ambiguous"]
    unsupported = [c for c in referential if c.get("semantic_validity") == "unsupported_reference_frame"]
    requires_predicate = [c for c in referential if c.get("semantic_validity") == "requires_missing_predicate"]

    def _safe_rate(num: int, den: int) -> float:
        return round(num / den, 4) if den > 0 else 0.0

    # Raw referential grounding: traditional accuracy treating all refs as groundable
    raw_ref_correct = sum(
        1 for c in referential
        if c.get("binding_correct") is True
    )
    raw_referential_grounding = _safe_rate(raw_ref_correct, len(referential))

    # Groundable referential accuracy: accuracy only on objectively groundable items
    groundable_correct = sum(
        1 for c in groundable
        if c.get("binding_correct") is True
    )
    groundable_referential_accuracy = _safe_rate(groundable_correct, len(groundable))

    # Ambiguous rejection accuracy: how often system correctly rejects ambiguous refs
    ambiguous_rejected = sum(
        1 for c in ambiguous
        if c.get("system_is_ungroundable") is True
    )
    ambiguous_rejection_accuracy = _safe_rate(ambiguous_rejected, len(ambiguous))

    # Unsupported reference rejection accuracy
    unsupported_rejected = sum(
        1 for c in unsupported
        if c.get("system_is_ungroundable") is True
    )
    unsupported_reference_rejection_accuracy = _safe_rate(unsupported_rejected, len(unsupported))

    # Combined: ambiguous + unsupported rejection
    non_groundable = ambiguous + unsupported + requires_predicate
    non_groundable_rejected = sum(
        1 for c in non_groundable
        if c.get("system_is_ungroundable") is True
    )
    non_groundable_rejection_accuracy = _safe_rate(non_groundable_rejected, len(non_groundable))

    # Unsafe guess count: system gave binding on ambiguous/unsupported
    unsafe_guess_count = sum(1 for c in non_groundable if c.get("unsafe_guess") is True)

    # Over-rejection count: system said ungroundable on groundable refs
    over_rejection_count = sum(1 for c in groundable if c.get("over_rejection") is True)

    # Evaluator mismatch count
    evaluator_mismatch_count = sum(1 for c in referential if c.get("evaluator_mismatch") is True)

    # Referential policy score: balanced metric
    # Rewards: correct grounding on groundable + correct rejection on non-groundable
    # Penalizes: unsafe guesses and over-rejections
    total_referential = len(referential)
    correct_outcomes = sum(1 for c in referential if c.get("correct_outcome") is True)
    referential_policy_score = _safe_rate(correct_outcomes, total_referential)

    # Per-type breakdown
    type_breakdown: dict[str, dict] = {}
    for ref_type in sorted(set(c.get("reference_type") for c in referential if c.get("reference_type"))):
        of_type = [c for c in referential if c.get("reference_type") == ref_type]
        type_groundable = [c for c in of_type if c.get("semantic_validity") == "objectively_groundable"]
        type_non_groundable = [c for c in of_type if c.get("semantic_validity") != "objectively_groundable"]

        type_breakdown[ref_type] = {
            "total": len(of_type),
            "groundable_count": len(type_groundable),
            "non_groundable_count": len(type_non_groundable),
            "groundable_correct": sum(1 for c in type_groundable if c.get("binding_correct") is True),
            "non_groundable_rejected": sum(1 for c in type_non_groundable if c.get("system_is_ungroundable") is True),
            "unsafe_guesses": sum(1 for c in type_non_groundable if c.get("unsafe_guess") is True),
        }

    return {
        "total_eval_results": len(classifications),
        "referential_total": len(referential),
        "non_referential_total": len(non_referential),
        "groundable_count": len(groundable),
        "ambiguous_count": len(ambiguous),
        "unsupported_count": len(unsupported),
        "requires_missing_predicate_count": len(requires_predicate),
        "raw_referential_grounding": raw_referential_grounding,
        "raw_referential_correct": raw_ref_correct,
        "groundable_referential_accuracy": groundable_referential_accuracy,
        "groundable_referential_correct": groundable_correct,
        "ambiguous_rejection_accuracy": ambiguous_rejection_accuracy,
        "ambiguous_rejected_count": ambiguous_rejected,
        "unsupported_reference_rejection_accuracy": unsupported_reference_rejection_accuracy,
        "unsupported_rejected_count": unsupported_rejected,
        "non_groundable_rejection_accuracy": non_groundable_rejection_accuracy,
        "non_groundable_rejected_count": non_groundable_rejected,
        "unsafe_guess_count": unsafe_guess_count,
        "over_rejection_count": over_rejection_count,
        "evaluator_mismatch_count": evaluator_mismatch_count,
        "referential_policy_score": referential_policy_score,
        "referential_correct_outcomes": correct_outcomes,
        "reference_type_breakdown": type_breakdown,
    }


def generate_markdown_report(metrics: dict, classifications: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Referential Semantics Evaluation Report")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total eval results | {metrics['total_eval_results']} |")
    lines.append(f"| Referential instructions | {metrics['referential_total']} |")
    lines.append(f"| Non-referential instructions | {metrics['non_referential_total']} |")
    lines.append("")
    lines.append("### Semantic Validity Breakdown")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| Objectively groundable | {metrics['groundable_count']} |")
    lines.append(f"| Objectively ambiguous | {metrics['ambiguous_count']} |")
    lines.append(f"| Unsupported reference frame | {metrics['unsupported_count']} |")
    lines.append(f"| Requires missing predicate | {metrics['requires_missing_predicate_count']} |")
    lines.append("")
    lines.append("### Core Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Raw referential grounding (traditional) | {metrics['raw_referential_grounding']:.2%} ({metrics['raw_referential_correct']}/{metrics['referential_total']}) |")
    lines.append(f"| Groundable referential accuracy | {metrics['groundable_referential_accuracy']:.2%} ({metrics['groundable_referential_correct']}/{metrics['groundable_count']}) |")
    lines.append(f"| Ambiguous rejection accuracy | {metrics['ambiguous_rejection_accuracy']:.2%} ({metrics['ambiguous_rejected_count']}/{metrics['ambiguous_count']}) |")
    lines.append(f"| Unsupported ref rejection accuracy | {metrics['unsupported_reference_rejection_accuracy']:.2%} ({metrics['unsupported_rejected_count']}/{metrics['unsupported_count']}) |")
    lines.append(f"| Non-groundable rejection accuracy | {metrics['non_groundable_rejection_accuracy']:.2%} ({metrics['non_groundable_rejected_count']}/{metrics['ambiguous_count'] + metrics['unsupported_count'] + metrics['requires_missing_predicate_count']}) |")
    lines.append(f"| **Referential policy score** | **{metrics['referential_policy_score']:.2%}** ({metrics['referential_correct_outcomes']}/{metrics['referential_total']}) |")
    lines.append("")
    lines.append("### Error Counts")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Unsafe guesses (guessed on ambiguous/unsupported) | {metrics['unsafe_guess_count']} |")
    lines.append(f"| Over-rejections (rejected on groundable) | {metrics['over_rejection_count']} |")
    lines.append(f"| Evaluator mismatches | {metrics['evaluator_mismatch_count']} |")
    lines.append("")
    lines.append("### Reference Type Breakdown")
    lines.append("")
    lines.append("| Type | Total | Groundable | Non-Groundable | Groundable Correct | Non-Groundable Rejected | Unsafe Guesses |")
    lines.append("|------|-------|------------|----------------|--------------------|-------------------------|----------------|")
    for ref_type, bd in metrics["reference_type_breakdown"].items():
        lines.append(f"| {ref_type} | {bd['total']} | {bd['groundable_count']} | {bd['non_groundable_count']} | {bd['groundable_correct']} | {bd['non_groundable_rejected']} | {bd['unsafe_guesses']} |")
    lines.append("")
    lines.append("## Per-Instruction Detail")
    lines.append("")
    lines.append("| Instruction | Validity | Type | System Output | Expected | Correct? |")
    lines.append("|-------------|----------|------|---------------|----------|----------|")
    for c in classifications:
        if not c.get("has_audit_entry"):
            continue
        inst = c["instruction"][:60]
        validity = c.get("semantic_validity", "?")
        ref_type = c.get("reference_type", "?")
        if c.get("system_is_ungroundable"):
            sys_out = "ungroundable"
        elif c.get("binding_correct") is True:
            sys_out = "correct binding"
        elif c.get("binding_correct") is False:
            sys_out = "wrong binding"
        else:
            sys_out = "?"
        expected_status = c.get("recommended_expected_status", "?")
        correct = "Y" if c.get("correct_outcome") else "N"
        lines.append(f"| {inst} | {validity} | {ref_type} | {sys_out} | {expected_status} | {correct} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic referential instruction evaluator"
    )
    parser.add_argument(
        "--eval_summary", required=True,
        help="Path to eval results JSON (e.g. predicate_inventory_eval.json)"
    )
    parser.add_argument(
        "--audit_file", required=True,
        help="Path to referential instruction audit JSON"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for output semantic metrics JSON"
    )
    parser.add_argument(
        "--markdown_output", default=None,
        help="Optional path for markdown report"
    )
    args = parser.parse_args()

    eval_data = load_eval_results(args.eval_summary)
    audit_entries = load_audit(args.audit_file)
    audit_lookup = build_audit_lookup(audit_entries)

    classifications = []
    for result in eval_data["results"]:
        instruction = result.get("instruction", "")
        audit_entry = audit_lookup.get(instruction)
        classification = classify_result(result, audit_entry)
        classifications.append(classification)

    metrics = compute_semantic_metrics(classifications)

    # Count how many audit entries were matched
    matched_ids = set(
        c.get("instruction_id") for c in classifications
        if c.get("instruction_id")
    )
    all_audit_ids = set(e.get("instruction_id") for e in audit_entries)
    unmatched = all_audit_ids - matched_ids
    metrics["audit_entries_total"] = len(audit_entries)
    metrics["audit_entries_matched"] = len(matched_ids)
    metrics["audit_entries_unmatched"] = list(sorted(unmatched)) if unmatched else []

    output_data = {
        "metrics": metrics,
        "classifications": classifications,
    }

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Semantic metrics written to {output_path}")

    if args.markdown_output:
        md = generate_markdown_report(metrics, classifications)
        md_path = Path(args.markdown_output)
        ensure_dir(md_path.parent)
        md_path.write_text(md, encoding="utf-8")
        print(f"Markdown report written to {md_path}")

    # Print summary
    print(f"\n=== Referential Semantic Evaluation ===")
    print(f"Referential instructions: {metrics['referential_total']}")
    print(f"  Groundable: {metrics['groundable_count']}")
    print(f"  Ambiguous: {metrics['ambiguous_count']}")
    print(f"  Unsupported: {metrics['unsupported_count']}")
    if unmatched:
        print(f"  UNMATCHED audit entries: {unmatched}")
    print(f"Raw referential grounding: {metrics['raw_referential_grounding']:.2%}")
    print(f"Groundable referential accuracy: {metrics['groundable_referential_accuracy']:.2%}")
    print(f"Ambiguous rejection accuracy: {metrics['ambiguous_rejection_accuracy']:.2%}")
    print(f"Unsupported ref rejection: {metrics['unsupported_reference_rejection_accuracy']:.2%}")
    print(f"Referential policy score: {metrics['referential_policy_score']:.2%}")
    print(f"Unsafe guesses: {metrics['unsafe_guess_count']}")
    print(f"Over-rejections: {metrics['over_rejection_count']}")


if __name__ == "__main__":
    main()
