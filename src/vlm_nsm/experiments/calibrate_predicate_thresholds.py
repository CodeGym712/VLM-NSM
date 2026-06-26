"""Calibrate PhysicalPredicateExtractor thresholds for robosuite environment.

Tests combinations of xy_threshold and z tolerance values to find optimal
thresholds for correct predicate extraction in the Stack/StackThree environment.

Usage:
    python -m vlm_nsm.experiments.calibrate_predicate_thresholds
"""

from __future__ import annotations

import json

from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir

# Threshold candidates
XY_CANDIDATES = [0.02, 0.03, 0.05, 0.08, 0.10]
Z_MIN_CANDIDATES = [-0.01, -0.005, 0.0]
Z_MAX_CANDIDATES = [0.03, 0.05, 0.08, 0.10]


def _known_ground_truth() -> list[dict]:
    """Return known object configurations and expected predicates."""
    cases = []

    # Case 1: All cubes on table, nothing stacked
    poses = {
        "cube1": {"x": -0.15, "y": 0.0, "z": 0.02},
        "cube2": {"x": 0.0, "y": 0.0, "z": 0.02},
        "cube3": {"x": 0.15, "y": 0.0, "z": 0.02},
    }
    targets = dict(poses)
    targets["table"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    cases.append({
        "name": "all_on_table",
        "object_poses": poses,
        "targets": targets,
        "expected_on": {("on", "cube1", "table"), ("on", "cube2", "table"), ("on", "cube3", "table")},
        "expected_clear": {("clear", "cube1"), ("clear", "cube2"), ("clear", "cube3")},
    })

    # Case 2: cube3 on cube1 (stack of 2)
    poses = {
        "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
        "cube2": {"x": 0.15, "y": 0.0, "z": 0.02},
        "cube3": {"x": 0.0, "y": 0.0, "z": 0.06},
    }
    targets = dict(poses)
    targets["table"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    cases.append({
        "name": "cube3_on_cube1",
        "object_poses": poses,
        "targets": targets,
        "expected_on": {("on", "cube1", "table"), ("on", "cube2", "table"), ("on", "cube3", "cube1")},
        "expected_clear": {("clear", "cube2"), ("clear", "cube3")},
    })

    # Case 3: Full stack cube3 on cube2 on cube1
    poses = {
        "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
        "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
        "cube3": {"x": 0.0, "y": 0.0, "z": 0.10},
    }
    targets = dict(poses)
    targets["table"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    cases.append({
        "name": "full_stack",
        "object_poses": poses,
        "targets": targets,
        "expected_on": {("on", "cube1", "table"), ("on", "cube2", "cube1"), ("on", "cube3", "cube2")},
        "expected_clear": {("clear", "cube3")},
    })

    # Case 4: Off-center stack (imperfect alignment)
    poses = {
        "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
        "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
        "cube3": {"x": 0.03, "y": 0.02, "z": 0.10},
    }
    targets = dict(poses)
    targets["table"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    cases.append({
        "name": "off_center_stack",
        "object_poses": poses,
        "targets": targets,
        "expected_on": {("on", "cube1", "table"), ("on", "cube2", "cube1")},
        "expected_clear": {("clear", "cube2"), ("clear", "cube3")},
    })

    return cases


def _evaluate_thresholds(cases: list[dict]) -> list[dict]:
    """Test all threshold combinations against known ground truth."""
    results = []

    for xy in XY_CANDIDATES:
        for z_min in Z_MIN_CANDIDATES:
            for z_max in Z_MAX_CANDIDATES:
                if z_min >= z_max:
                    continue

                thresholds = {"xy_threshold": xy, "z_min": z_min, "z_max": z_max}
                correct_on = 0
                correct_clear = 0
                total_on = 0
                total_clear = 0
                case_details = []

                for case in cases:
                    predicted = physical_state_to_predicates(
                        case["object_poses"], case["targets"], thresholds
                    )
                    predicted_on = {p for p in predicted if p[0] == "on"}
                    predicted_clear = {p for p in predicted if p[0] == "clear"}

                    expected_on = case["expected_on"]
                    expected_clear = case["expected_clear"]

                    on_match = len(predicted_on & expected_on)
                    clear_match = len(predicted_clear & expected_clear)

                    correct_on += on_match
                    correct_clear += clear_match
                    total_on += len(expected_on)
                    total_clear += len(expected_clear)

                    case_details.append({
                        "case": case["name"],
                        "expected_on": sorted(str(p) for p in expected_on),
                        "predicted_on": sorted(str(p) for p in predicted_on),
                        "expected_clear": sorted(str(p) for p in expected_clear),
                        "predicted_clear": sorted(str(p) for p in predicted_clear),
                        "on_recall": on_match / max(len(expected_on), 1),
                        "clear_recall": clear_match / max(len(expected_clear), 1),
                    })

                on_recall = correct_on / max(total_on, 1)
                clear_recall = correct_clear / max(total_clear, 1)
                avg_recall = (on_recall + clear_recall) / 2

                results.append({
                    "thresholds": thresholds,
                    "on_recall": on_recall,
                    "clear_recall": clear_recall,
                    "average_recall": avg_recall,
                    "case_details": case_details,
                })

    results.sort(key=lambda r: r["average_recall"], reverse=True)
    return results


def main():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper  # noqa: F401
        HAS_ROBOSUITE = True
    except ImportError:
        HAS_ROBOSUITE = False

    print("Calibrating PhysicalPredicateExtractor thresholds...")
    print(f"Robosuite available: {HAS_ROBOSUITE}")

    cases = _known_ground_truth()
    print(f"Testing {len(cases)} known configurations with threshold grid...")

    all_results = _evaluate_thresholds(cases)

    print(f"\nTop 5 threshold configurations:")
    for i, r in enumerate(all_results[:5]):
        t = r["thresholds"]
        print(f"  {i + 1}. xy={t['xy_threshold']}, z_min={t['z_min']}, z_max={t['z_max']} -> "
              f"avg_recall={r['average_recall']:.3f} (on={r['on_recall']:.3f}, clear={r['clear_recall']:.3f})")

    calibration = {
        "top_configs": all_results[:10],
        "all_results_count": len(all_results),
        "recommended": all_results[0]["thresholds"] if all_results else None,
        "has_robosuite": HAS_ROBOSUITE,
        "threshold_grid": {
            "xy_candidates": XY_CANDIDATES,
            "z_min_candidates": Z_MIN_CANDIDATES,
            "z_max_candidates": Z_MAX_CANDIDATES,
        },
    }

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / "robosuite_predicate_calibration.json"
    output_path.write_text(json.dumps(calibration, indent=2, default=str), encoding="utf-8")
    print(f"\nCalibration results saved to {output_path}")

    full_path = results_dir / "robosuite_predicate_calibration_full.json"
    full_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"Full results saved to {full_path}")


if __name__ == "__main__":
    main()
