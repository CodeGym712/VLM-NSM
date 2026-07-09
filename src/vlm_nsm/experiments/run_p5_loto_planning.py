"""Run P5-C leave-one-task-template-out symbolic planning audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from vlm_nsm.research.p5_symbolic_planning import run_p5_loto_planning_audit


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
        default="generated/libero_audit/p5_loto_planning",
        help="Directory for P5-C leave-one-task-template-out artifacts.",
    )
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum BFS plan depth.")
    parser.add_argument("--max-expansions", type=int, default=20000, help="Maximum BFS state expansions per problem.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_p5_loto_planning_audit(
        p3_dir=Path(args.p3_dir),
        p4_dir=Path(args.p4_dir),
        package_dir=Path(args.package_dir),
        out_dir=Path(args.output_dir),
        max_depth=args.max_depth,
        max_expansions=args.max_expansions,
    )
    print(
        "P5-C OK: "
        f"status={summary['status']} splits={summary['solved_splits']}/{summary['split_count']} "
        f"demos={summary['solved']}/{summary['problem_count']} invalid={summary['invalid_plan']}"
    )
    print(f"Artifacts saved to {args.output_dir}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
