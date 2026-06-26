#!/usr/bin/env python3
"""Search for the correct cam_xmat convention by enumerating interpretations.

Enumerates combinations of:
- reshape: row_major / col_major
- direction: cam2world / world2cam (transpose)
- y_flip: flip camera Y axis
- z_flip: flip camera Z axis

For each candidate convention, builds T_camera_world, forward-projects oracle
3D positions to pixels, and compares against detected bbox centres. Candidates
are ranked by median center pixel error.

Output: sorted candidates in JSON and per-candidate debug JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from vlm_nsm.perception.camera_geometry import (
    intrinsic_from_fov,
    project_world_point_to_pixel,
)
from vlm_nsm.perception.robosuite_camera_utils import CameraInfo


# ---------------------------------------------------------------------------
# Convention candidates
# ---------------------------------------------------------------------------


@dataclass
class ConventionCandidate:
    """A specific interpretation of cam_xmat."""

    name: str
    reshape_mode: str  # "row_major" | "col_major"
    R_direction: str   # "cam2world" | "world2cam"
    y_flip: bool
    z_flip: bool

    def build_T(self, cam_xpos, cam_xmat_flat) -> np.ndarray:
        """Build 4x4 T_camera_world from raw MuJoCo arrays."""
        if self.reshape_mode == "row_major":
            R = cam_xmat_flat.reshape(3, 3).copy()
        else:
            R = cam_xmat_flat.reshape(3, 3, order="F").copy()

        # Axis flips before direction decision
        if self.y_flip:
            R[:, 1] *= -1.0  # flip camera Y axis
        if self.z_flip:
            R[:, 2] *= -1.0  # flip camera Z axis

        if self.R_direction == "world2cam":
            R = R.T

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = cam_xpos
        return T


def generate_candidates() -> list[ConventionCandidate]:
    """Generate the full search space of convention candidates."""
    candidates = []
    for reshape_mode in ("row_major", "col_major"):
        for direction in ("cam2world", "world2cam"):
            for y_flip in (False, True):
                for z_flip in (False, True):
                    name = f"{reshape_mode}_{direction}"
                    if y_flip:
                        name += "_yflip"
                    if z_flip:
                        name += "_zflip"
                    if not y_flip and not z_flip:
                        name += "_vanilla"
                    candidates.append(ConventionCandidate(
                        name=name,
                        reshape_mode=reshape_mode,
                        R_direction=direction,
                        y_flip=y_flip,
                        z_flip=z_flip,
                    ))
    return candidates


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


def _get_oracle_pixel_pairs(
    eval_input_path: str,
    camera_name: str,
    detector_model_path: str,
    detector_conf_threshold: float = 0.15,
    max_images: int | None = None,
) -> list[dict]:
    """Extract (oracle_xyz, detected_pixel) pairs from eval data.

    Runs YOLO detector on images and matches detections to oracle objects
    by color. Returns list of dicts with oracle position, detected center
    and bottom-center pixels.
    """
    with open(eval_input_path) as f:
        instructions = json.load(f)
    if max_images:
        instructions = instructions[:max_images]

    # Load detector
    from vlm_nsm.perception.detector_perception import RGBDetectorPerception
    detector_path = Path(detector_model_path)
    if not detector_path.exists():
        print(f"WARNING: Detector not found at {detector_path}")
        return []
    detector = RGBDetectorPerception(
        model_path=detector_path, conf_threshold=detector_conf_threshold,
    )
    print(f"Detector loaded: {detector.model_name}")

    image_key = f"image_path_{camera_name}" if camera_name != "agentview" else "image_path"
    # For agentview, the image_path key is just "image_path"
    # For secondary cameras, it's "image_path_frontview" etc.

    pairs: list[dict] = []
    for idx, item in enumerate(instructions):
        scene_info = item.get("scene_info", {})
        oracle_objects = scene_info.get("objects", [])

        # Determine correct image path key
        if camera_name == "agentview":
            img_path = item.get("image_path", "")
        else:
            img_path = item.get(f"image_path_{camera_name}", "")
        if not img_path or not Path(img_path).exists():
            continue

        # Run detector
        from PIL import Image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        scene = detector.perceive(rgb_image=img, width=512, height=512)

        # Build detection lookup by color
        det_by_color: dict[str, dict] = {}
        for dobj in scene.objects:
            color = (dobj.color or dobj.label or "").lower().replace(" ", "_")
            color = _norm_color(color)
            det_by_color[color] = {
                "label": dobj.label,
                "color": dobj.color,
                "confidence": dobj.confidence,
                "bbox": dobj.attributes.get("bbox"),
                "center_px": dobj.attributes.get("center_px"),
                "center_norm": dobj.attributes.get("center_norm"),
            }

        for oracle_obj in oracle_objects:
            oracle_color = oracle_obj.get("color", "")
            oracle_pos = oracle_obj.get("position")
            if oracle_pos is None or not oracle_color:
                continue

            det = det_by_color.get(oracle_color)
            if det is None:
                continue

            # Extract detected pixels
            from vlm_nsm.perception.rgb_table_projection import (
                get_projection_pixel_from_bbox,
            )
            bbox = det.get("bbox")
            center_px = det.get("center_px")
            center_u = center_v = bottom_u = bottom_v = None

            if bbox:
                center_u, center_v = get_projection_pixel_from_bbox(bbox, mode="center")
                bottom_u, bottom_v = get_projection_pixel_from_bbox(bbox, mode="bottom_center")
            elif center_px and "x" in center_px:
                center_u = float(center_px["x"])
                center_v = float(center_px["y"])

            pair = {
                "frame_idx": idx,
                "object_id": oracle_obj.get("object_id", ""),
                "oracle_color": oracle_color,
                "oracle_position": oracle_pos,
                "detected_center_u": center_u,
                "detected_center_v": center_v,
                "detected_bottom_u": bottom_u,
                "detected_bottom_v": bottom_v,
            }
            pairs.append(pair)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(instructions)}] {len(pairs)} matched pairs")

    return pairs


def _norm_color(color: str) -> str:
    """Normalize color string for matching."""
    c = color.lower().replace(" ", "_")
    mapping = {"red_cube": "red", "green_cube": "green", "blue_cube": "blue"}
    return mapping.get(c, c)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _extract_cam_raw(env, camera_name: str) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Extract raw cam_xpos and cam_xmat from a Robosuite environment.

    Returns (cam_xpos, cam_xmat_flat, fov_y, width, height).
    """
    sim = env.env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    xpos = sim.data.cam_xpos[cam_id].copy()
    xmat_flat = sim.data.cam_xmat[cam_id].copy()
    fov_y = float(sim.model.cam_fovy[cam_id])
    return xpos, xmat_flat, fov_y, 512, 512


def evaluate_candidates(
    pairs: list[dict],
    candidates: list[ConventionCandidate],
    cam_xpos: np.ndarray,
    cam_xmat_flat: np.ndarray,
    fov_y_deg: float,
    width: int = 512,
    height: int = 512,
) -> list[dict]:
    """Evaluate each candidate convention against the oracle→pixel pairs.

    Returns candidates sorted by median center pixel error (ascending).
    """
    K = intrinsic_from_fov(fov_y_deg, width, height)

    results: list[dict] = []
    for cand in candidates:
        T = cand.build_T(cam_xpos, cam_xmat_flat)

        center_errors: list[float] = []
        bottom_errors: list[float] = []
        valid_count = 0
        behind_count = 0

        for pair in pairs:
            oracle_pos = pair["oracle_position"]
            if isinstance(oracle_pos, dict):
                oracle_xyz = np.array(
                    [oracle_pos["x"], oracle_pos["y"], oracle_pos["z"]],
                    dtype=np.float64,
                )
            else:
                oracle_xyz = np.array(oracle_pos[:3], dtype=np.float64)

            proj = project_world_point_to_pixel(oracle_xyz, K, T)
            if proj is None:
                behind_count += 1
                continue
            valid_count += 1
            pu, pv = proj

            cu = pair.get("detected_center_u")
            cv = pair.get("detected_center_v")
            if cu is not None and cv is not None:
                center_errors.append(float(np.hypot(pu - cu, pv - cv)))

            bu = pair.get("detected_bottom_u")
            bv = pair.get("detected_bottom_v")
            if bu is not None and bv is not None:
                bottom_errors.append(float(np.hypot(pu - bu, pv - bv)))

        result = {
            "candidate": cand.name,
            "reshape_mode": cand.reshape_mode,
            "R_direction": cand.R_direction,
            "y_flip": cand.y_flip,
            "z_flip": cand.z_flip,
            "valid_projections": valid_count,
            "behind_camera": behind_count,
        }

        if center_errors:
            arr = np.array(center_errors)
            result["center_pixel_error_mean"] = float(np.mean(arr))
            result["center_pixel_error_median"] = float(np.median(arr))
            result["center_pixel_error_p90"] = float(np.percentile(arr, 90))
            result["center_pixel_error_max"] = float(np.max(arr))
            result["center_pixel_error_std"] = float(np.std(arr))

        if bottom_errors:
            arr = np.array(bottom_errors)
            result["bottom_pixel_error_mean"] = float(np.mean(arr))
            result["bottom_pixel_error_median"] = float(np.median(arr))
            result["bottom_pixel_error_p90"] = float(np.percentile(arr, 90))
            result["bottom_pixel_error_max"] = float(np.max(arr))
            result["bottom_pixel_error_std"] = float(np.std(arr))

        results.append(result)

    # Sort by median center pixel error
    results.sort(key=lambda r: r.get("center_pixel_error_median", float("inf")))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def search_convention(
    eval_input_path: str,
    detector_model_path: str,
    camera_name: str = "agentview",
    detector_conf_threshold: float = 0.15,
    max_images: int | None = None,
    output_dir: str = "generated/results/stage_18",
) -> dict:
    """Run the full convention search pipeline.

    Returns a dict with results for the specified camera.
    """
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    # Step 1: Extract oracle-detection pairs
    print(f"Extracting oracle-detection pairs for {camera_name}...")
    pairs = _get_oracle_pixel_pairs(
        eval_input_path=eval_input_path,
        camera_name=camera_name,
        detector_model_path=detector_model_path,
        detector_conf_threshold=detector_conf_threshold,
        max_images=max_images,
    )
    print(f"  {len(pairs)} matched pairs")

    if not pairs:
        print("ERROR: No matched pairs found. Check detector model and eval data.")
        return {"error": "no_pairs"}

    # Step 2: Extract raw camera parameters from sim
    print(f"Extracting raw camera parameters for {camera_name}...")
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        render_mode="offscreen",
    )
    try:
        env.reset()
        cam_xpos, cam_xmat_flat, fov_y, width, height = _extract_cam_raw(
            env, camera_name,
        )
        print(f"  cam_xpos: {cam_xpos}")
        print(f"  cam_xmat_flat: {cam_xmat_flat}")
        print(f"  fov_y: {fov_y}")
    finally:
        env.env.close()

    # Step 3: Generate and evaluate candidates
    candidates = generate_candidates()
    print(f"\nEvaluating {len(candidates)} convention candidates...")
    results = evaluate_candidates(
        pairs, candidates, cam_xpos, cam_xmat_flat, fov_y, width, height,
    )

    # Step 4: Output
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary_path = out / f"camera_convention_search_{camera_name}.json"
    summary = {
        "camera_name": camera_name,
        "num_pairs": len(pairs),
        "num_candidates": len(results),
        "top_candidates": results[:5],
        "all_candidates": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {summary_path}")

    # Debug JSONL (per-candidate + per-pair detail)
    debug_path = out / f"camera_convention_search_{camera_name}_debug.jsonl"
    with open(debug_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Debug records saved to {debug_path}")

    # Print top 5
    print(f"\n{'=' * 80}")
    print(f"Top 5 convention candidates for {camera_name}:")
    print(f"{'Rank':<5} {'Candidate':<45} {'Center Med':>10} {'Bottom Med':>10} {'Valid':>7}")
    print("-" * 80)
    for rank, r in enumerate(results[:5], 1):
        cmed = r.get("center_pixel_error_median", float("nan"))
        bmed = r.get("bottom_pixel_error_median", float("nan"))
        valid = r.get("valid_projections", 0)
        print(f"{rank:<5} {r['candidate']:<45} {cmed:>10.1f} {bmed:>10.1f} {valid:>7}")

    # Best candidate details
    best = results[0] if results else None
    if best:
        print(f"\nBest convention: {best['candidate']}")
        print(f"  Center pixel error: mean={best.get('center_pixel_error_mean', 'N/A'):.1f} px, "
              f"median={best.get('center_pixel_error_median', 'N/A'):.1f} px")
        print(f"  Valid/total: {best.get('valid_projections', 0)}/{len(pairs)}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Search for correct cam_xmat convention by enumerating"
                    " interpretations and comparing projection errors",
    )
    parser.add_argument(
        "--eval_input",
        default="generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json",
    )
    parser.add_argument(
        "--detector_model_path",
        default="generated/models/rgb_perception_yolov8n_medium/best.pt",
    )
    parser.add_argument("--detector_conf_threshold", type=float, default=0.15)
    parser.add_argument("--camera_name", default="agentview")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default="generated/results/stage_18",
    )
    args = parser.parse_args()

    search_convention(
        eval_input_path=args.eval_input,
        detector_model_path=args.detector_model_path,
        camera_name=args.camera_name,
        detector_conf_threshold=args.detector_conf_threshold,
        max_images=args.max_images,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
