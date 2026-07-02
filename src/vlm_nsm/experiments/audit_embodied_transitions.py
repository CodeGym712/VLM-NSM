"""Audit and split EmbodiedTransition v1 JSONL datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from vlm_nsm.data.transition_dataset import (
    DEFAULT_SCHEMA_PATH,
    TransitionDatasetError,
    audit_records,
    group_split,
    load_jsonl,
    load_schema,
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="EmbodiedTransition JSONL input")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="JSON Schema path")
    parser.add_argument("--report", required=True, help="Machine-readable audit report path")
    parser.add_argument("--split-dir", help="Optional train/validation/test output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = Path(args.report)
    try:
        records = load_jsonl(args.input)
        schema = load_schema(args.schema)
    except (TransitionDatasetError, OSError, json.JSONDecodeError) as exc:
        _write_json(report_path, {
            "ok": False,
            "record_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "duplicate_transition_ids": [],
            "attribution_counts": {},
            "run_count": 0,
            "issues": [{"line_number": 0, "transition_id": None, "path": "$", "message": str(exc)}],
        })
        print(f"AUDIT FAILED: {exc}")
        return 1

    report = audit_records(records, schema)
    report_data = report.to_dict()
    _write_json(report_path, report_data)

    if not report.ok:
        print(
            "AUDIT FAILED: "
            f"records={report.record_count} invalid={report.invalid_count} "
            f"duplicate_ids={len(report.duplicate_transition_ids)}"
        )
        return 1

    if args.split_dir:
        try:
            splits = group_split(
                records,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
            )
        except ValueError as exc:
            report_data["ok"] = False
            report_data["split_error"] = str(exc)
            _write_json(report_path, report_data)
            print(f"AUDIT FAILED: {exc}")
            return 1
        split_dir = Path(args.split_dir)
        for split_name, split_records in splits.items():
            _write_jsonl(split_dir / f"{split_name}.jsonl", split_records)
        report_data["splits"] = {name: len(items) for name, items in splits.items()}
        _write_json(report_path, report_data)

    print(f"AUDIT OK: records={report.record_count} runs={report.run_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
