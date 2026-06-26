#!/usr/bin/env python3
"""Summarize Stage 18-F2 fixed calibration pose re-evaluation results.

Loads single-camera table projection metrics, dual-camera stereo metrics,
and spatial relation accuracy. Produces JSON summary and markdown report.

Usage:
  python src/vlm_nsm/experiments/summarize_fixed_calibration_pose_eval.py \
    --results_dir generated/results/stage_18 \
    --output generated/results/stage_18/fixed_calibration_pose_eval_summary.json \
    --markdown_output generated/results/stage_18/fixed_calibration_pose_eval_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-camera summary
# ---------------------------------------------------------------------------


def _summarize_single_camera(results_dir: Path) -> dict[str, Any]:
    """Aggregate single-camera table projection results."""
    cameras = ["agentview", "frontview", "birdview"]
    modes = ["center", "bottom_center", "weighted_bottom"]

    summary: dict[str, Any] = {}

    for camera in cameras:
        cam_summary = {}
        for mode in modes:
            path = results_dir / f"table_projection_pose_error_{camera}_{mode}_fixed_calib.json"
            data = _load_json(path)
            if data:
                cam_summary[mode] = {
                    "mean_xy_error": data.get("mean_xy_error"),
                    "median_xy_error": data.get("median_xy_error"),
                    "p90_xy_error": data.get("p90_xy_error"),
                    "max_xy_error": data.get("max_xy_error"),
                    "num_objects": data.get("num_objects"),
                    "failure_count": data.get("failure_count", 0),
                }
            else:
                cam_summary[mode] = {"error": "no_data"}

        # Determine best mode
        best_mode = None
        best_median = float("inf")
        for mode, s in cam_summary.items():
            m = s.get("median_xy_error")
            if m is not None and m < best_median:
                best_median = m
                best_mode = mode

        # Determine status
        best_data = cam_summary.get(best_mode or "", {})
        med = best_data.get("median_xy_error")
        mean_ = best_data.get("mean_xy_error")
        p90 = best_data.get("p90_xy_error")
        if med is not None:
            if med <= 0.02 and (p90 or 0) <= 0.03:
                status = "STRONG_PASS"
            elif med <= 0.03 and (mean_ or 0) <= 0.04:
                status = "PASS"
            else:
                status = "FAIL"
        else:
            status = "NO_DATA"

        summary[camera] = {
            "modes": cam_summary,
            "best_mode": best_mode,
            "best_median_xy_error": best_data.get("median_xy_error"),
            "best_mean_xy_error": best_data.get("mean_xy_error"),
            "best_p90_xy_error": best_data.get("p90_xy_error"),
            "status": status,
        }

    return summary


# ---------------------------------------------------------------------------
# Stereo summary
# ---------------------------------------------------------------------------


def _summarize_stereo(results_dir: Path) -> dict[str, Any]:
    """Load stereo triangulation results."""
    path = results_dir / "dual_camera_pose_error_agentview_frontview_fixed_calib.json"
    data = _load_json(path)
    if data is None:
        return {"error": "no_data"}

    med_xy = data.get("median_xy_error")
    p90_xy = data.get("p90_xy_error")
    res_med = data.get("triangulation_residual_median")

    if med_xy is not None and p90_xy is not None and res_med is not None:
        if med_xy <= 0.02 and p90_xy <= 0.03 and res_med <= 0.01:
            status = "PASS"
        else:
            status = "FAIL"
    else:
        status = "NO_DATA"

    return {
        "camera_pair": f"{data.get('camera_a', '?')}+{data.get('camera_b', '?')}",
        "num_frames": data.get("num_frames"),
        "num_valid_triangulations": data.get("num_valid_triangulations"),
        "median_xy_error": med_xy,
        "mean_xy_error": data.get("mean_xy_error"),
        "p90_xy_error": p90_xy,
        "triangulation_residual_median": res_med,
        "triangulation_residual_mean": data.get("triangulation_residual_mean"),
        "per_axis_abs_error_median": data.get("per_axis_abs_error_median"),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Spatial relation summary
# ---------------------------------------------------------------------------


def _summarize_spatial(results_dir: Path) -> dict[str, Any]:
    """Load spatial relation accuracy results."""
    path = results_dir / "spatial_relation_accuracy_fixed_calib.json"
    data = _load_json(path)
    if data is None:
        return {"error": "no_data"}
    return {
        "overall_accuracy": data.get("overall_relation_accuracy"),
        "left_right_accuracy": data.get("mean_left_right_accuracy"),
        "front_back_accuracy": data.get("mean_front_back_accuracy"),
        "near_far_accuracy": data.get("mean_near_far_accuracy"),
        "total_pairs": data.get("total_comparable_pairs"),
        "total_wrong_left_right": data.get("total_wrong_left_right"),
        "total_wrong_front_back": data.get("total_wrong_front_back"),
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _recommend(single: dict, stereo: dict, spatial: dict) -> str:
    stereo_status = stereo.get("status", "NO_DATA")
    single_statuses = [
        single[c].get("status", "NO_DATA")
        for c in ["agentview", "frontview"]
        if c in single
    ]

    if stereo_status == "PASS":
        return "PROCEED_STEREO_QWEN_EVAL"
    elif any(s in ("STRONG_PASS", "PASS") for s in single_statuses):
        return "ENABLE_TABLE_PROJECTION_EXPERIMENTAL"
    return "KEEP_IMAGE2D"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _generate_markdown(summary: dict) -> str:
    lines = [
        "# Stage 18-F2: Fixed Calibration Pose Re-Evaluation",
        "",
        f"**Date**: 2026-05-25",
        "",
        "## Single-Camera Table Projection (Fixed Calibration)",
        "",
        "| Camera | Mode | Mean XY (m) | Median XY (m) | P90 XY (m) | Max XY (m) | Status |",
        "|:-------|:-----|:-----------:|:-------------:|:----------:|:----------:|:------:|",
    ]

    sc = summary.get("single_camera", {})
    for camera in ["agentview", "frontview", "birdview"]:
        cam_data = sc.get(camera, {})
        modes = cam_data.get("modes", {})
        for mode in ["center", "bottom_center", "weighted_bottom"]:
            md = modes.get(mode, {})
            if "error" in md:
                lines.append(f"| {camera} | {mode} | — | — | — | — | — |")
                continue
            mean_s = f"{md.get('mean_xy_error', 0):.4f}" if md.get("mean_xy_error") is not None else "—"
            med_s = f"{md.get('median_xy_error', 0):.4f}" if md.get("median_xy_error") is not None else "—"
            p90_s = f"{md.get('p90_xy_error', 0):.4f}" if md.get("p90_xy_error") is not None else "—"
            max_s = f"{md.get('max_xy_error', 0):.4f}" if md.get("max_xy_error") is not None else "—"
            # Only show status on best mode row
            best = cam_data.get("best_mode")
            if mode == best:
                lines.append(f"| {camera} | **{mode}** | {mean_s} | {med_s} | {p90_s} | {max_s} | **{cam_data.get('status', '—')}** |")
            else:
                lines.append(f"| {camera} | {mode} | {mean_s} | {med_s} | {p90_s} | {max_s} | |")

    # Best mode summary
    lines.append("")
    lines.append("### Best Mode Per Camera")
    lines.append("")
    for camera in ["agentview", "frontview", "birdview"]:
        cam_data = sc.get(camera, {})
        best = cam_data.get("best_mode", "—")
        med = cam_data.get("best_median_xy_error")
        status = cam_data.get("status", "—")
        med_s = f"{med:.4f}m" if med is not None else "—"
        lines.append(f"- **{camera}**: `{best}` — median XY = {med_s} [{status}]")

    # Stereo
    lines.append("")
    lines.append("## Dual-Camera Stereo Triangulation (Fixed Calibration)")
    lines.append("")
    stereo = summary.get("stereo", {})
    if "error" in stereo:
        lines.append(f"**{stereo['error']}**")
    else:
        lines.append(f"- Pair: {stereo.get('camera_pair', '—')}")
        lines.append(f"- Frames: {stereo.get('num_frames', '—')}")
        lines.append(f"- Valid triangulations: {stereo.get('num_valid_triangulations', '—')}")
        med_xy = stereo.get("median_xy_error")
        mean_xy = stereo.get("mean_xy_error")
        p90_xy = stereo.get("p90_xy_error")
        res_med = stereo.get("triangulation_residual_median")
        lines.append(f"- Median XY error: {f'{med_xy:.4f}m' if med_xy else '—'}")
        lines.append(f"- Mean XY error: {f'{mean_xy:.4f}m' if mean_xy else '—'}")
        lines.append(f"- P90 XY error: {f'{p90_xy:.4f}m' if p90_xy else '—'}")
        lines.append(f"- Residual median: {f'{res_med:.6f}m' if res_med else '—'}")
        lines.append(f"- Status: **{stereo.get('status', '—')}**")

    # Spatial relations
    lines.append("")
    lines.append("## Spatial Relation Accuracy (Fixed Calibration)")
    lines.append("")
    spat = summary.get("spatial_relations", {})
    if "error" in spat:
        lines.append(f"**{spat['error']}**")
    else:
        lines.append(f"- Overall accuracy: {spat.get('overall_accuracy', '—')}")
        lines.append(f"- Left/Right: {spat.get('left_right_accuracy', '—')}")
        lines.append(f"- Front/Behind: {spat.get('front_back_accuracy', '—')}")
        lines.append(f"- Total pairs evaluated: {spat.get('total_pairs', '—')}")

    # Recommendation
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**{summary.get('recommendation', 'KEEP_IMAGE2D')}**")
    lines.append("")
    lines.append("`--rgb_spatial_mode image2d` remains the stable default.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summarize_fixed_calib(
    results_dir: str,
    output: str,
    markdown_output: str | None = None,
) -> dict:
    rd = Path(results_dir)

    single = _summarize_single_camera(rd)
    stereo = _summarize_stereo(rd)
    spatial = _summarize_spatial(rd)

    recommendation = _recommend(single, stereo, spatial)

    summary = {
        "single_camera": single,
        "stereo": stereo,
        "spatial_relations": spatial,
        "recommendation": recommendation,
        "results_dir": str(rd),
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to {out}")

    if markdown_output:
        md = _generate_markdown(summary)
        md_path = Path(markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md)
        print(f"Markdown report written to {md_path}")

    print(f"\nRecommendation: {recommendation}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Stage 18-F2 fixed calibration pose re-eval",
    )
    parser.add_argument("--results_dir", default="generated/results/stage_18")
    parser.add_argument("--output", default="generated/results/stage_18/fixed_calibration_pose_eval_summary.json")
    parser.add_argument("--markdown_output", default="generated/results/stage_18/fixed_calibration_pose_eval_summary.md")
    args = parser.parse_args()

    summarize_fixed_calib(
        results_dir=args.results_dir,
        output=args.output,
        markdown_output=args.markdown_output,
    )


if __name__ == "__main__":
    main()
