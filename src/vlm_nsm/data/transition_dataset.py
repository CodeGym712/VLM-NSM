"""Validation and leakage-safe dataset helpers for embodied transitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "embodied_transition_v1.schema.json"


class TransitionDatasetError(ValueError):
    """Raised when a transition file cannot be parsed safely."""


@dataclass(frozen=True)
class AuditIssue:
    line_number: int
    transition_id: str | None
    path: str
    message: str


@dataclass(frozen=True)
class AuditReport:
    record_count: int
    valid_count: int
    invalid_count: int
    duplicate_transition_ids: tuple[str, ...]
    attribution_counts: Mapping[str, int]
    run_count: int
    issues: tuple[AuditIssue, ...]

    @property
    def ok(self) -> bool:
        return self.invalid_count == 0 and not self.duplicate_transition_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "record_count": self.record_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "duplicate_transition_ids": list(self.duplicate_transition_ids),
            "attribution_counts": dict(self.attribution_counts),
            "run_count": self.run_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TransitionDatasetError(f"cannot read {source}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TransitionDatasetError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise TransitionDatasetError(f"line {line_number} must contain a JSON object")
        records.append(value)
    return records


def _error_path(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    if error.validator == "required":
        fragments = error.message.split("'")
        if len(fragments) >= 2:
            parts.append(fragments[1])
    return ".".join(parts) or "$"


def audit_records(
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any] | None = None,
) -> AuditReport:
    validator = Draft202012Validator(
        schema or load_schema(),
        format_checker=FormatChecker(),
    )
    issues: list[AuditIssue] = []
    invalid_lines: set[int] = set()
    id_counts: dict[str, int] = {}
    attribution_counts: dict[str, int] = {}
    run_ids: set[str] = set()

    for line_number, record in enumerate(records, start=1):
        transition_id = record.get("transition_id")
        if isinstance(transition_id, str):
            id_counts[transition_id] = id_counts.get(transition_id, 0) + 1
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id:
            run_ids.add(run_id)
        attribution = record.get("attribution")
        if isinstance(attribution, str):
            attribution_counts[attribution] = attribution_counts.get(attribution, 0) + 1

        errors = sorted(validator.iter_errors(record), key=lambda item: list(item.absolute_path))
        if errors:
            invalid_lines.add(line_number)
        for error in errors:
            issues.append(AuditIssue(
                line_number=line_number,
                transition_id=transition_id if isinstance(transition_id, str) else None,
                path=_error_path(error),
                message=error.message,
            ))

    duplicate_ids = tuple(sorted(key for key, count in id_counts.items() if count > 1))
    return AuditReport(
        record_count=len(records),
        valid_count=len(records) - len(invalid_lines),
        invalid_count=len(invalid_lines),
        duplicate_transition_ids=duplicate_ids,
        attribution_counts=attribution_counts,
        run_count=len(run_ids),
        issues=tuple(issues),
    )


def deduplicate_records(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep the first occurrence of each transition identity."""
    seen: set[str] = set()
    unique: list[Mapping[str, Any]] = []
    for record in records:
        transition_id = record.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
            transition_id = f"content:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        if transition_id in seen:
            continue
        seen.add(transition_id)
        unique.append(record)
    return unique


def group_split(
    records: Sequence[Mapping[str, Any]],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 0,
) -> dict[str, list[Mapping[str, Any]]]:
    """Split complete run groups deterministically to prevent trajectory leakage."""
    test_ratio = 1.0 - train_ratio - validation_ratio
    if train_ratio <= 0.0 or validation_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("ratios must reserve non-zero train, validation, and test partitions")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("every record requires a non-empty run_id")
        groups.setdefault(run_id, []).append(record)
    if len(groups) < 3:
        raise ValueError("group split requires at least 3 run groups")

    ordered_ids = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(),
    )
    group_count = len(ordered_ids)
    train_count = max(1, min(group_count - 2, round(group_count * train_ratio)))
    remaining = group_count - train_count
    validation_count = max(1, min(remaining - 1, round(group_count * validation_ratio)))

    assignments = {
        "train": ordered_ids[:train_count],
        "validation": ordered_ids[train_count:train_count + validation_count],
        "test": ordered_ids[train_count + validation_count:],
    }
    return {
        split_name: [record for group_id in group_ids for record in groups[group_id]]
        for split_name, group_ids in assignments.items()
    }
