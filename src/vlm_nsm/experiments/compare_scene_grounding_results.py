"""Compare scene grounding results with and without oracle scene inventory.

Loads two summary.json files from eval_online_qwen_parser output,
computes per-metric deltas, per-category breakdowns (if input JSON
with categories is provided), and outputs a comparison report.
"""

import argparse
import json
from pathlib import Path


def _load_summary(path: str) -> dict:
    """Load summary.json from directory path or direct file path."""
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text())


def _load_categories(input_path: str | None) -> dict[str, str]:
    """Build {instruction: category} map from input JSON."""
    if not input_path:
        return {}
    try:
        data = json.loads(Path(input_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if isinstance(data, list):
        return {item["instruction"]: item.get("category", "unknown") for item in data}
    if isinstance(data, dict):
        items = data.get("instructions", data)
        if isinstance(items, list):
            return {item["instruction"]: item.get("category", "unknown") for item in items}
    return {}


def _classify_invalid_object(results: list[dict], category_map: dict[str, str]) -> int:
    """Count results where category is adversarial_invalid_object."""
    count = 0
    for r in results:
        cat = category_map.get(r.get("instruction", ""), "")
        if cat == "adversarial_invalid_object":
            count += 1
    return count


def _summarize_results(results: list[dict], tag: str, category_map: dict[str, str]) -> dict:
    """Compute aggregate metrics from a list of result entries."""
    total = len(results)
    parse_ok = sum(1 for r in results if r.get("parse_success"))
    schema_valid = sum(1 for r in results if r.get("schema_valid"))
    alignment_success = sum(1 for r in results if r.get("alignment_success"))
    target_stack_success = sum(1 for r in results if r.get("target_stack_success"))
    planner_success = sum(1 for r in results if r.get("planner_success"))
    fully_successful = sum(
        1 for r in results
        if r.get("parse_success") and r.get("schema_valid")
        and r.get("alignment_success") and r.get("target_stack_success")
        and r.get("planner_success")
    )
    object_grounding_valid = sum(1 for r in results if r.get("object_grounding_valid") is True)
    object_grounding_fail = sum(1 for r in results if r.get("object_grounding_valid") is False)
    object_grounding_none = sum(1 for r in results if r.get("object_grounding_valid") is None)

    total_grounding_errors = sum(
        len(r.get("grounding_errors") or []) for r in results
    )

    # Ungroundable / adversarial rejection metrics
    ungroundable_detected = sum(1 for r in results if r.get("status") == "ungroundable")
    adversarial_rejection_success = sum(
        1 for r in results
        if r.get("category") == "adversarial_invalid_object" and r.get("status") == "ungroundable"
    )
    false_rejection = sum(
        1 for r in results
        if r.get("status") == "ungroundable" and r.get("category") != "adversarial_invalid_object"
    )
    adversarial_total = sum(
        1 for _, cat in category_map.items()
        if cat == "adversarial_invalid_object"
    )

    invalid_object_grounded = 0
    for r in results:
        cat = category_map.get(r.get("instruction", ""), "")
        if cat == "adversarial_invalid_object" and r.get("object_grounding_valid") is not None:
            invalid_object_grounded += 1

    empty_expected = sum(
        1 for r in results
        if r.get("expected_bindings") == {} or r.get("expected_bindings") is None
    )

    failure_taxonomy: dict[str, int] = {}
    for r in results:
        if r.get("status") == "ungroundable":
            key = "ungroundable_object"
        elif not r.get("parse_success"):
            key = "parse_error"
        elif not r.get("schema_valid"):
            key = "schema_invalid"
        elif r.get("alignment_success") is False:
            key = "alignment_failed"
        elif r.get("target_stack_success") is False:
            key = "target_stack_error"
        elif not r.get("planner_success"):
            key = "planner_error"
        elif r.get("object_grounding_valid") is False:
            key = "grounding_failed"
        else:
            continue
        failure_taxonomy[key] = failure_taxonomy.get(key, 0) + 1

    return {
        "tag": tag,
        "total": total,
        "parse_ok": parse_ok,
        "schema_valid": schema_valid,
        "alignment_success": alignment_success,
        "target_stack_success": target_stack_success,
        "planner_success": planner_success,
        "fully_successful": fully_successful,
        "object_grounding_valid": object_grounding_valid,
        "object_grounding_fail": object_grounding_fail,
        "object_grounding_none": object_grounding_none,
        "total_grounding_errors": total_grounding_errors,
        "invalid_object_grounded": invalid_object_grounded,
        "empty_expected_bindings": empty_expected,
        "ungroundable_detected": ungroundable_detected,
        "adversarial_rejection_success": adversarial_rejection_success,
        "adversarial_rejection_rate": round(adversarial_rejection_success / adversarial_total, 3)
        if adversarial_total > 0 else 0,
        "false_rejection": false_rejection,
        "false_rejection_rate": round(false_rejection / total, 3) if total > 0 else 0,
        "failure_taxonomy": failure_taxonomy,
    }


def _per_category(results: list[dict], category_map: dict[str, str]) -> dict:
    """Compute per-category metrics."""
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = category_map.get(r.get("instruction", ""), "unknown")
        categories.setdefault(cat, []).append(r)

    output: dict[str, dict] = {}
    for cat_name, cat_results in sorted(categories.items()):
        total = len(cat_results)
        grounding_valid = sum(1 for r in cat_results if r.get("object_grounding_valid") is True)
        grounding_fail = sum(1 for r in cat_results if r.get("object_grounding_valid") is False)
        grounding_none = sum(1 for r in cat_results if r.get("object_grounding_valid") is None)
        fully_ok = sum(
            1 for r in cat_results
            if r.get("parse_success") and r.get("planner_success")
        )
        ungroundable = sum(1 for r in cat_results if r.get("status") == "ungroundable")
        output[cat_name] = {
            "total": total,
            "fully_successful": fully_ok,
            "object_grounding_valid": grounding_valid,
            "object_grounding_fail": grounding_fail,
            "object_grounding_none": grounding_none,
            "ungroundable_detected": ungroundable,
        }
    return output


def _find_failure_examples(
    results: list[dict],
    max_examples: int = 5,
) -> list[dict]:
    """Extract representative failure entries."""
    failures = [r for r in results if not r.get("planner_success") or r.get("object_grounding_valid") is False or r.get("status") == "ungroundable"]
    return [{"instruction": r["instruction"][:80],
             "error_type": r.get("error_type"),
             "object_grounding_valid": r.get("object_grounding_valid"),
             "grounding_errors": r.get("grounding_errors"),
             "expected_bindings": r.get("expected_bindings"),
             "predicted_bindings": r.get("predicted_bindings")}
            for r in failures[:max_examples]]


def compare_scene_grounding_results(
    without_scene_path: str,
    with_oracle_path: str,
    input_path: str | None = None,
    with_detector_path: str | None = None,
) -> dict:
    """Compare scene grounding results across scene sources.

    Args:
        without_scene_path: Path to summary.json or its dir (no scene).
        with_oracle_path: Path to summary.json or its dir (oracle scene).
        input_path: Optional path to input JSON for category labels.
        with_detector_path: Optional path to rgb_detector summary.json.

    Returns:
        Comparison dict with keys for each condition, deltas, per-category.
    """
    without = _load_summary(without_scene_path)
    with_ = _load_summary(with_oracle_path)
    detector = _load_summary(with_detector_path) if with_detector_path else None

    category_map = _load_categories(input_path)

    without_results = without.get("results", [])
    with_results = with_.get("results", [])
    detector_results = detector.get("results", []) if detector else []

    without_metrics = _summarize_results(without_results, "without_scene", category_map)
    with_metrics = _summarize_results(with_results, "with_oracle", category_map)
    detector_metrics = _summarize_results(detector_results, "rgb_detector", category_map) if detector else {}

    # Compute deltas
    delta_keys = ["total", "parse_ok", "schema_valid", "alignment_success",
                  "target_stack_success", "planner_success", "fully_successful",
                  "object_grounding_valid", "object_grounding_fail", "object_grounding_none",
                  "total_grounding_errors", "invalid_object_grounded",
                  "ungroundable_detected", "adversarial_rejection_success",
                  "false_rejection"]
    delta = {}
    for key in delta_keys:
        delta[key] = with_metrics.get(key, 0) - without_metrics.get(key, 0)
    delta_detector = {}
    if detector_metrics:
        for key in delta_keys:
            delta_detector[key] = detector_metrics.get(key, 0) - without_metrics.get(key, 0)

    # Per-category comparison
    without_cat = _per_category(without_results, category_map)
    with_cat = _per_category(with_results, category_map)
    detector_cat = _per_category(detector_results, category_map) if detector else {}

    per_category: dict[str, dict] = {}
    all_cats = sorted(set(list(without_cat.keys()) + list(with_cat.keys()) + list(detector_cat.keys())))
    for cat in all_cats:
        entry: dict = {
            "without_scene": without_cat.get(cat, {}),
            "with_oracle": with_cat.get(cat, {}),
        }
        if detector_cat:
            entry["rgb_detector"] = detector_cat.get(cat, {})
        per_category[cat] = entry

    invalid_total = sum(
        1 for _, cat in category_map.items()
        if cat == "adversarial_invalid_object"
    )

    result: dict = {
        "input_source": input_path or "none",
        "without_scene": without_metrics,
        "with_oracle": with_metrics,
        "delta": delta,
        "per_category": per_category,
        "adversarial": {
            "total_invalid_instructions": invalid_total,
            "rejected_without_scene": without_metrics.get("adversarial_rejection_success", 0),
            "rejected_with_oracle": with_metrics.get("adversarial_rejection_success", 0),
            "without_ungroundable_detected": without_metrics.get("ungroundable_detected", 0),
            "with_ungroundable_detected": with_metrics.get("ungroundable_detected", 0),
            "without_false_rejection": without_metrics.get("false_rejection", 0),
            "with_false_rejection": with_metrics.get("false_rejection", 0),
        },
        "failure_examples": {
            "without_scene": _find_failure_examples(without_results),
            "with_oracle": _find_failure_examples(with_results),
        },
    }

    if detector_metrics:
        result["rgb_detector"] = detector_metrics
        result["delta_detector_vs_none"] = delta_detector
        result["adversarial"]["rejected_with_detector"] = detector_metrics.get("adversarial_rejection_success", 0)
        result["adversarial"]["with_detector_ungroundable_detected"] = detector_metrics.get("ungroundable_detected", 0)
        result["adversarial"]["with_detector_false_rejection"] = detector_metrics.get("false_rejection", 0)
        result["failure_examples"]["rgb_detector"] = _find_failure_examples(detector_results)

    return result


def _print_summary(result: dict) -> None:
    """Print comparison summary to stdout."""
    w = result["without_scene"]
    o = result["with_oracle"]
    d = result["delta"]
    has_detector = "rgb_detector" in result

    print(f"\n{'=' * 80}")
    print(f"Scene Grounding Comparison")
    print(f"{'=' * 80}")
    headers = ["Without", "Oracle", "Delta_O"]
    fmt = "{:<30s} {:>8s} {:>8s} {:>8s}"
    sep = "-" * 30 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8
    if has_detector:
        headers += ["Detector", "Delta_D"]
        fmt = "{:<30s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
        sep += " " + "-" * 8 + " " + "-" * 8

    print(fmt.format("Metric", *headers))
    print(sep)

    def _v(m, k):
        return m.get(k, 0) if m else 0

    for key in ["total", "parse_ok", "schema_valid", "alignment_success",
                "target_stack_success", "planner_success", "fully_successful",
                "object_grounding_valid", "object_grounding_fail",
                "total_grounding_errors"]:
        row = [key, _v(w, key), _v(o, key), _v(d, key)]
        if has_detector:
            de = result.get("delta_detector_vs_none", {})
            row += [_v(result.get("rgb_detector", {}), key), _v(de, key)]
        print(fmt.format(*[str(x) for x in row]))

    if result["per_category"]:
        print(f"\n{'=' * 80}")
        print(f"Per-Category Object Grounding Validity")
        print(f"{'=' * 80}")
        for cat, data in result["per_category"].items():
            wv = data["without_scene"].get("object_grounding_valid", 0)
            wt = data["without_scene"].get("total", 0)
            ov = data["with_oracle"].get("object_grounding_valid", 0)
            ot = data["with_oracle"].get("total", 0)
            line = f"  {cat:<30s} none: {wv}/{wt}  oracle: {ov}/{ot}"
            if has_detector and "rgb_detector" in data:
                dv = data["rgb_detector"].get("object_grounding_valid", 0)
                dt = data["rgb_detector"].get("total", 0)
                line += f"  detector: {dv}/{dt}"
            print(line)

    adv = result["adversarial"]
    if adv["total_invalid_instructions"] > 0:
        line = (f"\nAdversarial rejection: none={adv['rejected_without_scene']}"
                f"/{adv['total_invalid_instructions']}"
                f"  oracle={adv['rejected_with_oracle']}"
                f"/{adv['total_invalid_instructions']}")
        if has_detector:
            line += (f"  detector={adv.get('rejected_with_detector', 0)}"
                     f"/{adv['total_invalid_instructions']}")
        print(line)
        for tag in ["with_oracle", "rgb_detector"]:
            if tag not in result.get("failure_examples", {}):
                continue
            fr_key = "with_false_rejection" if tag == "with_oracle" else "with_detector_false_rejection"
            fr = adv.get(fr_key, 0)
            if fr > 0:
                print(f"  False rejection ({tag}): {fr}")

    for tag in ["with_oracle", "rgb_detector"]:
        examples = result.get("failure_examples", {}).get(tag, [])
        if examples:
            print(f"\nFailure examples ({tag}):")
            for ex in examples[:3]:
                print(f"  - {ex['instruction'][:60]}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare scene grounding results across scene sources"
    )
    parser.add_argument("--without_scene", required=True,
                        help="Summary.json or dir for without-scene results")
    parser.add_argument("--with_oracle", required=True,
                        help="Summary.json or dir for with-oracle results")
    parser.add_argument("--with_detector", default=None,
                        help="Summary.json or dir for rgb_detector results")
    parser.add_argument("--input", default=None,
                        help="Input JSON (for category labels)")
    parser.add_argument("--output",
                        default="generated/results/stage_17/scene_grounding_comparison.json",
                        help="Output comparison JSON path")
    args = parser.parse_args()

    result = compare_scene_grounding_results(
        without_scene_path=args.without_scene,
        with_oracle_path=args.with_oracle,
        input_path=args.input,
        with_detector_path=args.with_detector,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Comparison written to {output_path}")

    _print_summary(result)


if __name__ == "__main__":
    main()
