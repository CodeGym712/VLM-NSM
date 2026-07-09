"""Run P4 typed domain export and symbolic replay from P3 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from vlm_nsm.research.p4_domain_export import run_p4_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p3-dir",
        default="generated/libero_audit/p3_symbolic_induction",
        help="Directory containing P3 symbolic induction artifacts.",
    )
    parser.add_argument(
        "--package-dir",
        default="generated/libero_audit/p3_start_package",
        help="Directory containing the P3 start package metadata.",
    )
    parser.add_argument(
        "--server-validation-dir",
        default="generated/libero_audit/p3_server_validation",
        help="Directory containing server-side P3 validation patches.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated/libero_audit/p4_domain_export",
        help="Directory for P4 domain export artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server_dir = Path(args.server_validation_dir)
    summary = run_p4_pipeline(
        p3_dir=Path(args.p3_dir),
        package_dir=Path(args.package_dir),
        server_validation_dir=server_dir if server_dir.exists() else None,
        out_dir=Path(args.output_dir),
    )
    print(
        "P4 OK: "
        f"status={summary['status']} operators={summary['domain_operator_count']} "
        f"covered={summary['covered']}/{summary['demo_count']} "
        f"failed={summary['failed']} invalid={summary['invalid_plan']}"
    )
    print(f"Artifacts saved to {summary['out_dir']}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
