"""Run P3 offline symbolic induction from a P2v4.2 PredicateTrace package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from vlm_nsm.research.p3_symbolic_induction import run_p3_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default="generated/libero_audit/p3_start_package",
        help="Directory containing the P3 start package.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated/libero_audit/p3_symbolic_induction",
        help="Directory for P3 artifacts.",
    )
    parser.add_argument(
        "--holding-min-frames",
        type=int,
        default=3,
        help="Minimum consecutive frames required to trust holding(obj).",
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=3,
        help="Merge predicate changes within this frame gap into one event.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_p3_pipeline(
        package_dir=Path(args.package_dir),
        out_dir=Path(args.output_dir),
        holding_min_frames=args.holding_min_frames,
        merge_gap=args.merge_gap,
    )
    print(
        "P3 OK: "
        f"frames={summary['normalized_frames']} demos={summary['demos']} "
        f"events={summary['events']} covered={summary['covered_demos']}/{summary['demo_count']}"
    )
    print(f"Artifacts saved to {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

