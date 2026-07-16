"""Preflight applicability evaluation for the relational Operator seed.

This script intentionally does *not* report applicability F1.  The negative
pool is generated from traces and is therefore provisional.  It only checks
whether learned preconditions can be grounded against positive events and
trace-derived candidate states, and records the evidence needed for a later
server-side BDDL/state replay audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vlm_nsm.research.relational_operator_learner import applicability_binding, effect_signature, family_binding


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ground(atom: list[str], binding: dict[str, str]) -> tuple[str, ...]:
    return tuple(binding.get(value, value) for value in atom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--positive-events", type=Path, required=True)
    parser.add_argument("--negative-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    families = json.loads(args.domain.read_text(encoding="utf-8"))
    by_family = {str(item["family"]): item for item in families}
    by_signature = {tuple(tuple(part) for part in item["effect_signature"]): item for item in families}
    positives = read_jsonl(args.positive_events)
    candidates = read_jsonl(args.negative_candidates)
    positive_by_source = {
        f"{row.get('file', '')}::{row.get('demo', '')}::{row.get('t_start', '')}": row for row in positives
    }

    positive_counts: Counter[str] = Counter()
    positive_applicable: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    candidate_predicted_negative: Counter[str] = Counter()
    unresolved = 0
    unresolved_cases: list[dict[str, Any]] = []
    candidate_details: list[dict[str, Any]] = []
    positive_failure_details: list[dict[str, Any]] = []

    for row in positives:
        family = by_signature.get(effect_signature(row))
        if family is None:
            continue
        name = str(family["family"])
        positive_counts[name] += 1
        binding, missing_roles = applicability_binding(row, family)
        if missing_roles:
            positive_failure_details.append(
                {
                    "family": name,
                    "source_event": f"{row.get('file', '')}::{row.get('demo', '')}::{row.get('t_start', '')}",
                    "missing_precondition_roles": missing_roles,
                    "provenance": "positive event; effect-only roles are intentionally excluded from this check",
                }
            )
            continue
        required = {ground(atom, binding) for atom in family.get("preconditions", [])}
        before = {tuple(str(value) for value in atom) for atom in row.get("before", [])}
        applicable = required.issubset(before)
        positive_applicable[name] += int(applicable)
        if not applicable:
            positive_failure_details.append(
                {
                    "family": name,
                    "source_event": f"{row.get('file', '')}::{row.get('demo', '')}::{row.get('t_start', '')}",
                    "missing_preconditions": [list(atom) for atom in sorted(required - before)],
                    "before": [list(atom) for atom in sorted(before)],
                    "provenance": "positive event; boundary/state mismatch candidate for R2 repair",
                }
            )

    for candidate in candidates:
        source = str(candidate.get("source_event", ""))
        source_event = positive_by_source.get(source)
        family = by_family.get(str(candidate.get("family", "")))
        if source_event is None:
            unresolved += 1
            unresolved_cases.append(
                {
                    "family": candidate.get("family", ""),
                    "source_event": source,
                    "frame_t": candidate.get("frame_t"),
                    "unresolved_reason": "data_missing",
                    "detail": "source event is absent from the supplied positive-event input",
                }
            )
            continue
        if family is None:
            unresolved += 1
            unresolved_cases.append(
                {
                    "family": candidate.get("family", ""),
                    "source_event": source,
                    "frame_t": candidate.get("frame_t"),
                    "unresolved_reason": "data_missing",
                    "detail": "candidate family is absent from the supplied domain",
                }
            )
            continue
        event_binding = family_binding(source_event, family)
        binding, missing_roles = applicability_binding(source_event, family)
        if missing_roles:
            unresolved += 1
            unresolved_cases.append(
                {
                    "family": family["family"],
                    "source_event": source,
                    "frame_t": candidate.get("frame_t"),
                    "unresolved_reason": "binding_missing",
                    "missing_precondition_roles": missing_roles,
                    "event_binding": event_binding,
                    "detail": "only roles used by preconditions are required; effect-only roles are allowed to remain unbound",
                }
            )
            continue
        required = {ground(atom, binding) for atom in family.get("preconditions", [])}
        state = {tuple(str(value) for value in atom) for atom in candidate.get("state", [])}
        predicted_negative = not required.issubset(state)
        name = str(family["family"])
        candidate_counts[name] += 1
        candidate_predicted_negative[name] += int(predicted_negative)
        candidate_details.append(
            {
                "family": name,
                "source_event": source,
                "file": candidate.get("file", ""),
                "demo": candidate.get("demo", ""),
                "frame_t": candidate.get("frame_t"),
                "predicted_applicable": not predicted_negative,
                "missing_preconditions": candidate.get("missing_preconditions", []),
                "event_binding": event_binding,
                "precondition_binding": binding,
                "provenance": "trace-derived candidate; not ground truth",
            }
        )

    families_report = []
    for name in sorted(set(positive_counts) | set(candidate_counts)):
        p_total = positive_counts[name]
        c_total = candidate_counts[name]
        families_report.append(
            {
                "family": name,
                "positive_events": p_total,
                "positive_applicable": positive_applicable[name],
                "positive_precondition_recall": positive_applicable[name] / p_total if p_total else None,
                "candidate_negatives": c_total,
                "candidate_predicted_negative": candidate_predicted_negative[name],
                "candidate_negative_rate": candidate_predicted_negative[name] / c_total if c_total else None,
            }
        )

    report = {
        "stage": "R2",
        "status": "PROVISIONAL_NO_F1",
        "domain_family_count": len(families),
        "positive_event_count": len(positives),
        "candidate_count": len(candidates),
        "resolved_candidate_count": len(candidate_details),
        "unresolved_candidate_count": unresolved,
        "unresolved_reason_counts": dict(Counter(item["unresolved_reason"] for item in unresolved_cases)),
        "families": families_report,
        "positive_precondition_failures": positive_failure_details,
        "limitations": [
            "Negative examples are trace-derived candidates and are not labels.",
            "Do not report applicability precision/recall/F1 from this file.",
            "Server must validate candidates with BDDL replay or an independent state predicate oracle.",
            "Effect-only roles are not required for applicability binding; unresolved cases distinguish binding and data gaps.",
        ],
        "input_sha256": {
            "domain": sha256(args.domain),
            "positive_events": sha256(args.positive_events),
            "negative_candidates": sha256(args.negative_candidates),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r2_applicability_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "r2_applicability_candidate_details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidate_details),
        encoding="utf-8",
    )
    (args.output_dir / "unresolved_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in unresolved_cases),
        encoding="utf-8",
    )
    files = [p for p in sorted(args.output_dir.iterdir()) if p.is_file() and p.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
