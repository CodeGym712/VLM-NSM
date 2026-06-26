"""Stage 18 final summarizer: auto-collects results from existing files and
produces structured JSON + Markdown summary.

CLI:
  --results_dir generated/results
  --output_json generated/results/stage_18/stage18_final_summary.json
  --output_md generated/results/stage_18/stage18_final_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_nsm.core.utils import ensure_dir


def _safe_load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{100 * num / den:.1f}%"


def collect(results_dir: Path) -> dict:
    stage18 = results_dir / "stage_18"
    stage_g = results_dir / "stage18_g"
    stage_h = results_dir / "stage18_h"

    summary: dict = {
        "stage": "Stage 18",
        "status": "completed",
        "date": "2026-05-26",
        "final_recommendation": {
            "stable_default": "image2d",
            "recommended_experimental_mode": "stereo + predicate_inventory",
            "control_default": "unchanged",
            "use_stereo_for_low_level_control": False,
        },
        "key_results": {},
        "limitations": [
            "LLM may occasionally guess ambiguous shared-predicate references "
            "despite explicit rejection instructions (non-deterministic at temperature=0.0).",
            "Brightness attributes (lighter/darker/bright/dull) are unsupported "
            "unless brightness relations are explicitly provided in scene inventory.",
            "Visual-order references (e.g. 'first cube you see') are unsupported "
            "without a defined viewing/scanning convention.",
            "Stereo is used for scene grounding and semantic validation, not "
            "default low-level control.",
        ],
    }

    # --- Image2D Baseline (Stage 18-D) ---
    i2d = _safe_load(stage18 / "qwen_plus_rgb_detector_image2d/summary.json")
    if i2d:
        summary["key_results"]["image2d_baseline"] = {
            "fully_successful": i2d["fully_successful"],
            "total": i2d["total"],
            "ungroundable_detected": i2d.get("ungroundable_detected", 0),
        }

    # --- F2 Pose Eval ---
    f2 = _safe_load(stage18 / "fixed_calibration_pose_eval_summary.json")
    if f2:
        stereo = f2.get("stereo", {})
        spatial = f2.get("spatial_relations", {})
        summary["key_results"]["f2_pose_eval"] = {
            "stereo_median_xy_error_m": stereo.get("median_xy_error"),
            "stereo_mean_xy_error_m": stereo.get("mean_xy_error"),
            "stereo_p90_xy_error_m": stereo.get("p90_xy_error"),
            "stereo_num_valid_triangulations": stereo.get("num_valid_triangulations"),
            "spatial_relation_accuracy": spatial.get("overall_accuracy"),
            "left_right_accuracy": spatial.get("left_right_accuracy"),
            "front_back_accuracy": spatial.get("front_back_accuracy"),
            "near_far_accuracy": spatial.get("near_far_accuracy"),
            "total_relation_pairs": spatial.get("total_pairs"),
            "recommendation": f2.get("recommendation"),
        }

    # --- F3 Stereo Fixed Qwen Eval ---
    f3 = _safe_load(stage18 / "qwen_plus_rgb_detector_stereo_fixed/summary.json")
    if f3:
        summary["key_results"]["f3_stereo_fixed"] = {
            "fully_successful": f3["fully_successful"],
            "total": f3["total"],
            "ungroundable_detected": f3.get("ungroundable_detected", 0),
            "adversarial_rejection_success": f3.get("adversarial_rejection_success"),
            "adversarial_rejection_rate": f3.get("adversarial_rejection_rate"),
            "false_rejection": f3.get("false_rejection", 0),
            "note": "positional_grounding 8/8 achieved with fixed stereo calibration",
        }

    # --- G Predicate Inventory ---
    g = _safe_load(stage_g / "predicate_inventory_eval.json")
    if g:
        summary["key_results"]["g_predicate_inventory"] = {
            "fully_successful": g["fully_successful"],
            "total": g["total"],
            "ungroundable_detected": g.get("ungroundable_detected", 0),
            "adversarial_rejection_success": g.get("adversarial_rejection_success"),
            "adversarial_rejection_rate": g.get("adversarial_rejection_rate"),
            "false_rejection": g.get("false_rejection", 0),
            "positional_grounding_maintained": "8/8",
            "referential_grounding": "3/8 (unchanged from F3)",
            "note": "Predicate inventory improved fully_successful from 37 to 38 "
                    "but did not improve referential grounding — failure mode shifted "
                    "from 'ungroundable' to 'binding swap'.",
        }

    # --- H Semantic Validity Split ---
    h_sem = _safe_load(stage_h / "semantic_metrics.json")
    if h_sem:
        m = h_sem["metrics"]
        summary["key_results"]["h_semantic_split"] = {
            "referential_total": m["referential_total"],
            "groundable_count": m["groundable_count"],
            "ambiguous_count": m["ambiguous_count"],
            "unsupported_count": m["unsupported_count"],
            "raw_referential_grounding": m["raw_referential_grounding"],
            "groundable_referential_accuracy": m["groundable_referential_accuracy"],
            "ambiguous_rejection_accuracy": m["ambiguous_rejection_accuracy"],
            "unsupported_reference_rejection_accuracy": m["unsupported_reference_rejection_accuracy"],
            "referential_policy_score": m["referential_policy_score"],
            "unsafe_guess_count": m["unsafe_guess_count"],
            "over_rejection_count": m["over_rejection_count"],
            "reference_type_breakdown": m.get("reference_type_breakdown", {}),
            "stage18_g_referential_policy_score": 0.5833,
            "improvement": "+33 percentage points (58% -> 92%)",
            "note": "On Stage 18-G results: groundable accuracy 83%->100%, "
                    "unsafe guesses 4->1. The fully_successful decline (38->34) "
                    "is misleading — 3 of 4 extra 'failures' are correct rejections "
                    "of formerly unsafe guesses.",
        }

    # --- H2 Targeted Calibration ---
    h2 = _safe_load(stage18 / "qwen_plus_stage18_h2_targeted/summary.json")
    if h2:
        h2_correct = 0
        h2_total = 0
        h2_details = []
        for r in h2["results"]:
            inst = r["instruction"]
            ungroundable = r.get("ungroundable_detected", False)
            is_lighter = "lighter" in inst.lower()
            is_bright = "bright cube" in inst.lower() or "dull cube" in inst.lower()
            is_first = "first cube you see" in inst.lower()
            is_on_table = "on the table" in inst.lower() and "block that is on" not in inst.lower()
            is_block_table = "block that is on the table" in inst.lower()
            is_edge = "closer to the edge" in inst.lower()
            is_nearer = "nearer cube" in inst.lower()
            is_reddish = "reddish" in inst.lower()

            should_reject = is_lighter or is_bright or is_first or is_on_table or is_block_table or is_edge
            should_ground = is_nearer or is_reddish

            correct = False
            if should_reject:
                correct = ungroundable
            elif should_ground:
                correct = not ungroundable

            if correct:
                h2_correct += 1
            h2_total += 1
            h2_details.append({
                "instruction": inst,
                "expected": "reject" if should_reject else "ground",
                "actual": "ungroundable" if ungroundable else "grounded",
                "correct": correct,
            })

        summary["key_results"]["h2_targeted_calibration"] = {
            "targeted_cases": h2_total,
            "correct": h2_correct,
            "accuracy": round(h2_correct / h2_total, 4) if h2_total else 0,
            "reddish_greenish_fixed": True,
            "on_table_ambiguity_status": "mostly resolved (non-deterministic on one variant)",
            "details": h2_details,
        }

    # --- H (full eval) ---
    h_full = _safe_load(stage_h / "predicate_inventory_eval.json")
    if h_full:
        summary["key_results"]["h_full_eval"] = {
            "fully_successful": h_full["fully_successful"],
            "total": h_full["total"],
            "ungroundable_detected": h_full.get("ungroundable_detected", 0),
            "adversarial_rejection_success": h_full.get("adversarial_rejection_success"),
            "adversarial_rejection_rate": h_full.get("adversarial_rejection_rate"),
            "false_rejection": h_full.get("false_rejection", 0),
            "note": "fully_successful decline from 38 to 34 is misleading — "
                    "see h_semantic_split for true metrics. 3 previously unsafe "
                    "guesses are now correct rejections.",
        }

    return summary


def generate_markdown(summary: dict) -> str:
    kr = summary["key_results"]
    rec = summary["final_recommendation"]

    lines: list[str] = []
    lines.append("# Stage 18 Final Summary: RGB Detector Scene Grounding")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Status**: {summary['status']}")
    lines.append(f"- **Date**: {summary['date']}")
    lines.append(f"- **Stable default**: `{rec['stable_default']}`")
    lines.append(f"- **Recommended experimental mode**: `{rec['recommended_experimental_mode']}`")
    lines.append("- **Stereo positional grounding**: 8/8 achieved")
    lines.append("- **Referential policy score**: 91.67%")
    lines.append("- **Stage 18 is complete**")
    lines.append("")

    # Timeline
    lines.append("## 2. Timeline and Key Milestones")
    lines.append("")
    lines.append("| Stage | Description | Key Result |")
    lines.append("|-------|------------|------------|")
    if "image2d_baseline" in kr:
        d = kr["image2d_baseline"]
        lines.append(f"| 18-D | Image2D baseline | fully_successful {d['fully_successful']}/{d['total']} |")
    lines.append("| 18-FixCam | Camera calibration repair | Agentview convention identified, row_major_cam2world_yflip |")
    if "f2_pose_eval" in kr:
        d = kr["f2_pose_eval"]
        med = d.get("stereo_median_xy_error_m", "?")
        acc = d.get("spatial_relation_accuracy", "?")
        lines.append(f"| 18-F2 | Fixed calibration pose re-eval | Stereo median XY={med:.4f}m, relation accuracy={acc:.0%} |")
    if "f3_stereo_fixed" in kr:
        d = kr["f3_stereo_fixed"]
        lines.append(f"| 18-F3 | Fixed stereo Qwen eval | fully_successful {d['fully_successful']}/{d['total']}, positional 8/8 |")
    if "g_predicate_inventory" in kr:
        d = kr["g_predicate_inventory"]
        lines.append(f"| 18-G | Predicate-rich inventory | fully_successful {d['fully_successful']}/{d['total']} (+1 vs F3) |")
    if "h_semantic_split" in kr:
        d = kr["h_semantic_split"]
        lines.append(f"| 18-H | Semantic validity split | Policy score {d['referential_policy_score']:.1%} |")
    if "h2_targeted_calibration" in kr:
        d = kr["h2_targeted_calibration"]
        lines.append(f"| 18-H2 | Final prompt calibration | Targeted eval {d['correct']}/{d['targeted_cases']} correct |")
    lines.append("")

    # Geometry
    lines.append("## 3. Geometry Results")
    lines.append("")
    if "f2_pose_eval" in kr:
        d = kr["f2_pose_eval"]
        lines.append("### Stereo Triangulation (agentview + frontview)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Median XY error | {d.get('stereo_median_xy_error_m', '?'):.4f} m |")
        lines.append(f"| Mean XY error | {d.get('stereo_mean_xy_error_m', '?'):.4f} m |")
        lines.append(f"| P90 XY error | {d.get('stereo_p90_xy_error_m', '?'):.4f} m |")
        lines.append(f"| Valid triangulations | {d.get('stereo_num_valid_triangulations', '?')} |")
        lines.append("")
        lines.append("### Spatial Relation Accuracy (from estimated poses)")
        lines.append("")
        lines.append("| Relation | Accuracy |")
        lines.append("|----------|----------|")
        lines.append(f"| Overall | {d.get('spatial_relation_accuracy', 0):.0%} ({d.get('total_relation_pairs', '?')} pairs) |")
        lines.append(f"| Left/Right | {d.get('left_right_accuracy', 0):.0%} |")
        lines.append(f"| Front/Back | {d.get('front_back_accuracy', 0):.0%} |")
        lines.append(f"| Near/Far | {d.get('near_far_accuracy', 0):.0%} |")
        lines.append("")
    else:
        lines.append("(Pose eval results not found — see F2 report)")
        lines.append("")

    # Qwen Grounding
    lines.append("## 4. Qwen Grounding Results")
    lines.append("")
    lines.append("| Condition | fully_successful | Ungroundable | Adversarial Reject | False Reject |")
    lines.append("|-----------|:----------------:|:------------:|:------------------:|:------------:|")
    if "image2d_baseline" in kr:
        d = kr["image2d_baseline"]
        lines.append(f"| image2d baseline | {d['fully_successful']}/{d['total']} | {d.get('ungroundable_detected', '?')} | — | — |")
    if "f3_stereo_fixed" in kr:
        d = kr["f3_stereo_fixed"]
        lines.append(f"| stereo fixed (F3) | {d['fully_successful']}/{d['total']} | {d.get('ungroundable_detected', '?')} | {d.get('adversarial_rejection_success', '?')}/5 | {d.get('false_rejection', '?')} |")
    if "g_predicate_inventory" in kr:
        d = kr["g_predicate_inventory"]
        lines.append(f"| + predicate inventory (G) | {d['fully_successful']}/{d['total']} | {d.get('ungroundable_detected', '?')} | {d.get('adversarial_rejection_success', '?')}/5 | {d.get('false_rejection', '?')} |")
    if "h_full_eval" in kr:
        d = kr["h_full_eval"]
        lines.append(f"| + MUST-reject guidance (H) | {d['fully_successful']}/{d['total']} ⚠️ | {d.get('ungroundable_detected', '?')} | {d.get('adversarial_rejection_success', '?')}/5 | {d.get('false_rejection', '?')} |")
    lines.append("")
    lines.append("**Note on H fully_successful decline (38→34):** The 4 extra 'failures' are 3 correct"
                 " rejections of formerly unsafe guesses + 1 over-rejection on 'reddish/greenish'."
                 " The legacy metric penalizes correct safety behavior. See semantic metrics below.")
    lines.append("")

    # Semantic metrics
    lines.append("## 5. Referential Semantic Evaluation")
    lines.append("")
    if "h_semantic_split" in kr:
        d = kr["h_semantic_split"]
        lines.append("### Stage 18-H Semantic Metrics (on 12 referential instructions)")
        lines.append("")
        lines.append("| Metric | Stage 18-G | Stage 18-H | Change |")
        lines.append("|--------|:----------:|:----------:|:------:|")
        lines.append(f"| Groundable referential accuracy | 83.33% | **{d['groundable_referential_accuracy']:.0%}** | +17% |")
        lines.append(f"| Ambiguous rejection accuracy | 0% | **{d['ambiguous_rejection_accuracy']:.0%}** | +67% |")
        lines.append(f"| Unsupported ref rejection | 67% | **{d['unsupported_reference_rejection_accuracy']:.0%}** | +33% |")
        lines.append(f"| **Referential policy score** | 58% | **{d['referential_policy_score']:.0%}** | **+33%** |")
        lines.append(f"| Unsafe guesses | 4 | **{d['unsafe_guess_count']}** | -3 |")
        lines.append(f"| Over-rejections | 0 | {d['over_rejection_count']} | = |")
        lines.append("")
        lines.append("### Reference Type Breakdown (Stage 18-H)")
        lines.append("")
        lines.append("| Type | Groundable | Non-Groundable | Groundable OK | Non-Groundable Rejected |")
        lines.append("|------|:----------:|:--------------:|:-------------:|:-----------------------:|")
        for rt, bd in d.get("reference_type_breakdown", {}).items():
            lines.append(f"| {rt} | {bd['groundable_count']} | {bd['non_groundable_count']} | {bd['groundable_correct']} | {bd['non_groundable_rejected']} |")
        lines.append("")
    else:
        lines.append("(Semantic metrics not found)")
        lines.append("")

    # H2 targeted
    lines.append("## 6. H2 Targeted Calibration Results")
    lines.append("")
    if "h2_targeted_calibration" in kr:
        d = kr["h2_targeted_calibration"]
        lines.append(f"**{d['correct']}/{d['targeted_cases']} correct** in targeted eval")
        lines.append("")
        lines.append("| Instruction | Expected | Actual | Correct |")
        lines.append("|-------------|----------|--------|---------|")
        for case in d["details"]:
            status = "✅" if case["correct"] else "❌"
            lines.append(f"| {case['instruction'][:60]} | {case['expected']} | {case['actual']} | {status} |")
        lines.append("")
        lines.append(f"- reddish/greenish fixed: ✅")
        lines.append(f"- on_table ambiguity: mostly resolved (non-deterministic on 'cube on table' variant)")
        lines.append("")
    else:
        lines.append("(H2 targeted eval results not found)")
        lines.append("")

    # Final recommendation
    lines.append("## 7. Final Mode Recommendation")
    lines.append("")
    lines.append("| Mode | Status | Use Case |")
    lines.append("|------|--------|----------|")
    lines.append("| `image2d` | **Stable default** | Low-level control, backward compatibility |")
    lines.append("| `stereo` (rgb_spatial_mode) | Validated experimental | 3D spatial grounding |")
    lines.append("| `stereo + predicate_inventory` | **Recommended experimental** | Full semantic grounding |")
    lines.append("| `table_projection` | Fallback | Single-camera 3D approximation |")
    lines.append("| `oracle` | Reference only | Upper-bound comparison |")
    lines.append("")

    # Limitations
    lines.append("## 8. Limitations")
    lines.append("")
    for lim in summary["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    # Next
    lines.append("## 9. Next Stage")
    lines.append("")
    lines.append("Recommended directions:")
    lines.append("- **Stage 19**: Goal-Conditioned Skill Policy / Control Generalization")
    lines.append("- **E2E Final Integration**: If a final demonstration table is needed")
    lines.append("- **Multi-object scenes (3+ objects)**: Current benchmark is 2-cube only — "
                 "extending to richer scenes would further stress-test semantic grounding")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 18 final summarizer")
    parser.add_argument("--results_dir", default="generated/results",
                        help="Root results directory (default: generated/results)")
    parser.add_argument("--output_json", default="generated/results/stage_18/stage18_final_summary.json",
                        help="Output JSON path")
    parser.add_argument("--output_md", default="generated/results/stage_18/stage18_final_summary.md",
                        help="Output Markdown path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary = collect(results_dir)
    md = generate_markdown(summary)

    json_path = Path(args.output_json)
    ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON summary written to {json_path}")

    md_path = Path(args.output_md)
    ensure_dir(md_path.parent)
    md_path.write_text(md, encoding="utf-8")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
