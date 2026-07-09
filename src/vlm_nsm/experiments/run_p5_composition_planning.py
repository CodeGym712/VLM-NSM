"""Run P5-B cross-task composition symbolic planning audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from vlm_nsm.research.p5_symbolic_planning import run_p5_composition_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p3-dir",
        default="generated/libero_audit/p3_symbolic_induction",
        help="Directory containing P3 normalized PredicateTrace artifacts.",
    )
    parser.add_argument(
        "--p4-dir",
        default="generated/libero_audit/p4_domain_export",
        help="Directory containing P4 domain export artifacts.",
    )
    parser.add_argument(
        "--package-dir",
        default="generated/libero_audit/p3_start_package",
        help="Directory containing P3/P4 package metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated/libero_audit/p5_composition_planning",
        help="Directory for P5-B composition planning audit artifacts.",
    )
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum BFS plan depth.")
    parser.add_argument("--max-expansions", type=int, default=20000, help="Maximum BFS state expansions per problem.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_p5_composition_audit(
        p3_dir=Path(args.p3_dir),
        p4_dir=Path(args.p4_dir),
        package_dir=Path(args.package_dir),
        out_dir=Path(args.output_dir),
        max_depth=args.max_depth,
        max_expansions=args.max_expansions,
    )
    print(
        "P5-B OK: "
        f"status={summary['status']} attempted={summary['attempted']}/{summary['problem_count']} "
        f"solved={summary['solved']} failed={summary['failed']} invalid={summary['invalid_plan']} "
        f"skipped={summary['skipped_absent_object']}"
    )
    print(f"Artifacts saved to {args.output_dir}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
