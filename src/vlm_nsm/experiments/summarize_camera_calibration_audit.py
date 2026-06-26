#!/usr/bin/env python3
"""Summarize Stage 18-FixCam camera calibration audit results.

Loads world geometry inspection, projection consistency, and convention
search results. Produces a JSON summary and a markdown report.

Usage:
  python src/vlm_nsm/experiments/summarize_camera_calibration_audit.py \
    --results_dir generated/results/stage_18 \
    --output generated/results/stage_18/camera_calibration_audit_summary.json \
    --markdown_output generated/results/stage_18/camera_calibration_audit_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-camera analysis
# ---------------------------------------------------------------------------


def _get_vanilla_pixel_errors(proj_consistency: dict | None,
                              camera_name: str) -> dict[str, float | None]:
    """Extract vanilla convention pixel errors from projection consistency."""
    result: dict[str, float | None] = {
        "center_pixel_median": None,
        "center_pixel_mean": None,
        "bottom_pixel_median": None,
        "bottom_pixel_mean": None,
    }
    if proj_consistency is None:
        return result
    for cm in proj_consistency.get("cameras", []):
        if cm.get("camera") == camera_name:
            cerr = cm.get("center_pixel_error", {})
            berr = cm.get("bottom_pixel_error", {})
            result["center_pixel_median"] = cerr.get("median")
            result["center_pixel_mean"] = cerr.get("mean")
            result["bottom_pixel_median"] = berr.get("median")
            result["bottom_pixel_mean"] = berr.get("mean")
            result["num_valid"] = cm.get("num_valid_projections")
            result["num_detected"] = cm.get("num_detected")
    return result


def _get_best_convention(convention_search: dict | None) -> dict[str, Any]:
    """Extract the best convention candidate."""
    if convention_search is None:
        return {"error": "no_data"}
    top = convention_search.get("top_candidates", [])
    if not top:
        return {"error": "no_candidates"}
    best = top[0]
    return {
        "name": best.get("candidate"),
        "reshape_mode": best.get("reshape_mode"),
        "R_direction": best.get("R_direction"),
        "y_flip": best.get("y_flip"),
        "z_flip": best.get("z_flip"),
        "center_pixel_median": best.get("center_pixel_error_median"),
        "center_pixel_mean": best.get("center_pixel_error_mean"),
        "center_pixel_p90": best.get("center_pixel_error_p90"),
        "bottom_pixel_median": best.get("bottom_pixel_error_median"),
        "bottom_pixel_mean": best.get("bottom_pixel_error_mean"),
        "bottom_pixel_p90": best.get("bottom_pixel_error_p90"),
        "valid_projections": best.get("valid_projections"),
    }


def _get_vanilla_rank(convention_search: dict | None) -> int | None:
    """Find the rank of the vanilla convention."""
    if convention_search is None:
        return None
    for i, c in enumerate(convention_search.get("all_candidates", [])):
        if "vanilla" in c.get("candidate", "") and "yflip" not in c.get("candidate", ""):
            return i + 1
    return None


def _get_improvement(best: dict, vanilla_rank: int | None) -> float | None:
    """Compute improvement factor of best vs vanilla."""
    if vanilla_rank is None or vanilla_rank <= 1:
        return None
    # Actually let me compute from the values
    return None  # computed at summary level


def _determine_status(best: dict, vanilla_errors: dict) -> str:
    """Determine PASS / MARGINAL / FAIL for a camera."""
    bottom_median = best.get("bottom_pixel_median")
    center_median = best.get("center_pixel_median")

    # Check both center and bottom
    px_ok = (center_median is not None and center_median <= 20) or \
            (bottom_median is not None and bottom_median <= 20)

    if px_ok:
        return "PASS"
    elif (center_median is not None and center_median <= 50) or \
         (bottom_median is not None and bottom_median <= 50):
        return "MARGINAL"
    return "FAIL"


# ---------------------------------------------------------------------------
# Global diagnosis
# ---------------------------------------------------------------------------


def _diagnose(cameras: dict[str, dict]) -> dict[str, Any]:
    """Produce global diagnosis and recommendation."""
    diagnoses: list[str] = []
    recommendation = "KEEP_IMAGE2D"

    # Check if all cameras share the same best convention
    conventions = {}
    for cam_name, cam_data in cameras.items():
        best = cam_data.get("best_convention", {})
        name = best.get("name")
        if name and "error" not in best:
            conventions[cam_name] = name

    # Check consistent convention
    unique_convs = set(conventions.values())
    if len(unique_convs) == 1 and len(conventions) >= 2:
        diagnoses.append(f"CONSISTENT_CONVENTION_FOUND: {list(unique_convs)[0]} across {list(conventions.keys())}")

    # Check per-camera status
    agentview_status = cameras.get("agentview", {}).get("status", "FAIL")
    frontview_status = cameras.get("frontview", {}).get("status", "FAIL")
    birdview_status = cameras.get("birdview", {}).get("status", "FAIL")

    if agentview_status in ("PASS", "MARGINAL") and frontview_status == "FAIL":
        diagnoses.append("AGENTVIEW_ONLY_VALID")

    # Check if convention search improves significantly
    for cam_name, cam_data in cameras.items():
        improvement = cam_data.get("improvement_factor")
        if improvement is not None and improvement > 5:
            diagnoses.append(f"CAMERA_CONVENTION_SUSPECT_{cam_name}_improved_{improvement:.0f}x")

    # Check if all conventions fail
    all_fail = all(
        cameras.get(c, {}).get("status") == "FAIL"
        for c in ["agentview", "frontview", "birdview"]
        if c in cameras
    )
    if all_fail:
        diagnoses.append("DATASET_CAMERA_MISMATCH")

    # Check if table_z is suspect
    world_geom = cameras.get("_world_geometry", {})
    table_body_z = world_geom.get("table_body_z")
    if table_body_z is not None and abs(table_body_z - 0.8) > 0.05:
        diagnoses.append(f"TABLE_Z_SUSPECT: body_z={table_body_z}, expected ~0.8")

    # Determine recommendation
    pass_count = sum(1 for s in [agentview_status, frontview_status, birdview_status] if s == "PASS")
    marginal_count = sum(1 for s in [agentview_status, frontview_status, birdview_status] if s == "MARGINAL")
    unique_cameras_with_data = [c for c in ["agentview", "frontview", "birdview"]
                                if c in cameras and "error" not in cameras[c].get("best_convention", {})]

    if len(unique_convs) == 1 and pass_count + marginal_count >= 2:
        recommendation = "PROCEED_STEREO_REPAIR"
    elif pass_count + marginal_count >= 1:
        recommendation = "FIX_EXTRINSICS"
    elif all_fail:
        recommendation = "KEEP_IMAGE2D"

    if not diagnoses:
        diagnoses.append("KEEP_IMAGE2D")

    return {
        "diagnoses": diagnoses,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _generate_markdown(summary: dict) -> str:
    """Generate a compact markdown report from the summary dict."""
    lines = [
        "# Stage 18-FixCam-R: Camera Calibration Audit Summary",
        "",
        f"**Date**: 2026-05-25",
        "",
        "## Per-Camera Results",
        "",
        "| Camera | Best Convention | Center px (med) | Bottom px (med) | Status | Vanilla Rank |",
        "|:-------|:----------------|:----------------:|:----------------:|:------:|:------------:|",
    ]

    for cam_name in ["agentview", "frontview", "birdview"]:
        cam_data = summary.get("cameras", {}).get(cam_name, {})
        best = cam_data.get("best_convention", {})
        if "error" in best:
            lines.append(f"| {cam_name} | {best.get('error', 'N/A')} | — | — | — | — |")
            continue
        name = best.get("name", "N/A")
        cm = best.get("center_pixel_median", "—")
        bm = best.get("bottom_pixel_median", "—")
        cm_str = f"{cm:.1f}" if isinstance(cm, (int, float)) else str(cm)
        bm_str = f"{bm:.1f}" if isinstance(bm, (int, float)) else str(bm)
        status = cam_data.get("status", "—")
        vr = cam_data.get("vanilla_rank", "—")
        lines.append(f"| {cam_name} | {name} | {cm_str} | {bm_str} | **{status}** | {vr} |")

    # Improvement details
    lines.append("")
    lines.append("## Convention Improvement")
    lines.append("")
    for cam_name in ["agentview", "frontview", "birdview"]:
        cam_data = summary.get("cameras", {}).get(cam_name, {})
        improvement = cam_data.get("improvement_factor")
        vanilla = cam_data.get("vanilla_center_median")
        best_med = cam_data.get("best_convention", {}).get("center_pixel_median")
        if improvement is not None and vanilla is not None and best_med is not None:
            lines.append(f"- **{cam_name}**: {vanilla:.0f} px → {best_med:.1f} px "
                         f"({improvement:.0f}x improvement)")

    # World geometry
    wg = summary.get("world_geometry", {})
    if wg:
        lines.append("")
        lines.append("## World Geometry")
        lines.append("")
        lines.append(f"- Table body Z: {wg.get('table_body_z', 'N/A')}")
        lines.append(f"- Table surface Z: {wg.get('table_surface_z', 'N/A')}")

    # Diagnosis
    lines.append("")
    lines.append("## Global Diagnosis")
    lines.append("")
    for d in summary.get("global_diagnosis", []):
        lines.append(f"- {d}")

    lines.append("")
    lines.append(f"**Recommendation**: `{summary.get('recommendation', 'KEEP_IMAGE2D')}`")

    lines.append("")
    lines.append("## Stereo Status")
    lines.append("")
    if summary.get("recommendation") == "PROCEED_STEREO_REPAIR":
        lines.append("Stereo triangulation can proceed with fixed calibrations.")
        lines.append("All cameras with data share the same best convention.")
    else:
        lines.append("Stereo should remain **disabled** until calibration is resolved.")

    lines.append("")
    lines.append("## Current Default")
    lines.append("")
    lines.append("`--rgb_spatial_mode image2d` remains the stable default.")
    lines.append("`table_projection` and `stereo` are experimental until Qwen eval is re-run after geometry repair.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summarize_audit(
    results_dir: str,
    output: str,
    markdown_output: str | None = None,
) -> dict:
    """Load all audit outputs and produce a summary."""
    rd = Path(results_dir)

    # Load inputs
    world_geom = _load_json(rd / "robosuite_world_geometry_inspection.json")
    proj_agent = _load_json(rd / "projection_consistency_agentview.json")
    proj_front = _load_json(rd / "projection_consistency_frontview.json")
    proj_bird = _load_json(rd / "projection_consistency_birdview.json")
    conv_agent = _load_json(rd / "camera_convention_search_agentview.json")
    conv_front = _load_json(rd / "camera_convention_search_frontview.json")
    conv_bird = _load_json(rd / "camera_convention_search_birdview.json")

    proj_by_cam = {
        "agentview": proj_agent,
        "frontview": proj_front,
        "birdview": proj_bird,
    }
    conv_by_cam = {
        "agentview": conv_agent,
        "frontview": conv_front,
        "birdview": conv_bird,
    }

    cameras: dict[str, dict] = {}

    # --- Per-camera analysis ---
    for cam_name in ["agentview", "frontview", "birdview"]:
        proj = proj_by_cam.get(cam_name)
        conv = conv_by_cam.get(cam_name)

        best = _get_best_convention(conv)
        vanilla_errors = _get_vanilla_pixel_errors(proj, cam_name)
        vanilla_rank = _get_vanilla_rank(conv)
        status = _determine_status(best, vanilla_errors)

        # Improvement factor: best median vs vanilla median
        improvement_factor = None
        vanilla_med = vanilla_errors.get("center_pixel_median")
        best_med = best.get("center_pixel_median")
        if vanilla_med and best_med and best_med > 0:
            improvement_factor = vanilla_med / best_med

        cameras[cam_name] = {
            "best_convention": best,
            "vanilla_center_median": vanilla_errors.get("center_pixel_median"),
            "vanilla_center_mean": vanilla_errors.get("center_pixel_mean"),
            "vanilla_bottom_median": vanilla_errors.get("bottom_pixel_median"),
            "vanilla_bottom_mean": vanilla_errors.get("bottom_pixel_mean"),
            "vanilla_rank": vanilla_rank,
            "improvement_factor": improvement_factor,
            "status": status,
        }

    # --- World geometry summary ---
    wg_summary = {}
    if world_geom:
        table = world_geom.get("table", {})
        wg_summary["table_body_z"] = table.get("position", {}).get("z")
        table_geoms = table.get("geoms", [])
        if table_geoms and table_geoms[0].get("size"):
            half_z = table_geoms[0]["size"][2] if len(table_geoms[0]["size"]) > 2 else 0.025
            wg_summary["table_surface_z"] = (wg_summary.get("table_body_z") or 0) + half_z
        cubes = world_geom.get("cubes", {})
        for cn, cd in cubes.items():
            if "position" in cd:
                wg_summary[f"{cn}_z"] = cd["position"]["z"]
        cameras["_world_geometry"] = wg_summary

    # --- Global diagnosis ---
    diag = _diagnose(cameras)

    summary = {
        "cameras": {k: v for k, v in cameras.items() if not k.startswith("_")},
        "world_geometry": wg_summary,
        "global_diagnosis": diag["diagnoses"],
        "recommendation": diag["recommendation"],
        "results_dir": str(rd),
    }

    # --- Write JSON ---
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to {out}")

    # --- Write Markdown ---
    if markdown_output:
        md = _generate_markdown(summary)
        md_path = Path(markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md)
        print(f"Markdown report written to {md_path}")

    # Print to stdout
    print()
    for cam_name in ["agentview", "frontview", "birdview"]:
        cd = cameras.get(cam_name, {})
        best = cd.get("best_convention", {})
        status = cd.get("status", "—")
        imp = cd.get("improvement_factor")
        imp_str = f" ({imp:.0f}x improvement)" if imp else ""
        if "error" in best:
            print(f"{cam_name}: NO DATA — {best['error']}")
        else:
            print(f"{cam_name}: {best.get('name', 'N/A')} "
                  f"center={best.get('center_pixel_median', 'N/A')} px "
                  f"bottom={best.get('bottom_pixel_median', 'N/A')} px "
                  f"[{status}]{imp_str}")

    print(f"\nDiagnosis: {'; '.join(diag['diagnoses'])}")
    print(f"Recommendation: {diag['recommendation']}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Stage 18-FixCam camera calibration audit",
    )
    parser.add_argument("--results_dir", default="generated/results/stage_18")
    parser.add_argument("--output", default="generated/results/stage_18/camera_calibration_audit_summary.json")
    parser.add_argument("--markdown_output", default="generated/results/stage_18/camera_calibration_audit_summary.md")
    args = parser.parse_args()

    summarize_audit(
        results_dir=args.results_dir,
        output=args.output,
        markdown_output=args.markdown_output,
    )


if __name__ == "__main__":
    main()
