#!/usr/bin/env python3
"""Evaluate dual-camera stereo triangulation pose error against oracle positions.

For each paired frame (agentview + frontview):
1. Run YOLO detector on both images
2. Match objects across views
3. Triangulate 3D position from stereo rays
4. Compare against oracle ground-truth positions

No DashScope/Qwen required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
from vlm_nsm.perception.camera_geometry import (
    pixel_to_camera_ray,
    transform_ray_to_world,
    triangulate_rays,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_calibration(path: str) -> CameraInfo:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {p}")
    data = json.loads(p.read_text())
    return CameraInfo(
        K=np.array(data["K"], dtype=np.float64),
        T_camera_world=np.array(data["T_camera_world"], dtype=np.float64),
        width=data.get("width", 512),
        height=data.get("height", 512),
        camera_name=data.get("camera_name", "unknown"),
    )


def _norm_color(color: str) -> str:
    c = color.lower().replace(" ", "_")
    mapping = {"red_cube": "red", "green_cube": "green", "blue_cube": "blue"}
    return mapping.get(c, c)


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------


def _extract_pixel_from_detection(det: dict, mode: str = "center") -> tuple[float, float] | None:
    """Extract a pixel coordinate from a detection dict.

    Tries bbox-derived modes first, then center_px fallback.
    """
    from vlm_nsm.perception.rgb_table_projection import get_projection_pixel_from_bbox

    bbox = det.get("bbox")
    if bbox:
        if mode == "center":
            return get_projection_pixel_from_bbox(bbox, mode="center")
        elif mode == "bottom_center":
            return get_projection_pixel_from_bbox(bbox, mode="bottom_center")

    center_px = det.get("center_px")
    if center_px and "x" in center_px:
        return (float(center_px["x"]), float(center_px["y"]))

    center_norm = det.get("center_norm")
    if center_norm and "x" in center_norm:
        return (float(center_norm["x"]) * 512, float(center_norm["y"]) * 512)

    return None


def _match_detections_by_color(dets_a: list[dict], dets_b: list[dict]) -> list[tuple[int, int]]:
    """Match detections across two views by normalized color."""
    matches = []
    used_b = set()
    for ia, da in enumerate(dets_a):
        color_a = _norm_color(da.get("color", "") or da.get("label", ""))
        for ib, db in enumerate(dets_b):
            if ib in used_b:
                continue
            color_b = _norm_color(db.get("color", "") or db.get("label", ""))
            if color_a == color_b:
                matches.append((ia, ib))
                used_b.add(ib)
                break
    return matches


def _process_pair(
    item: dict,
    detector,
    calib_a: CameraInfo,
    calib_b: CameraInfo,
    pixel_mode: str,
    width: int,
    height: int,
) -> list[dict]:
    """Process one paired frame and return per-object records."""
    from PIL import Image

    img_path_a = item.get("image_path", "")
    img_path_b = item.get("image_path_frontview", "")

    if not img_path_a or not img_path_b:
        return []

    if not Path(img_path_a).exists() or not Path(img_path_b).exists():
        return []

    # Load images
    try:
        img_a = Image.open(img_path_a).convert("RGB")
        img_b = Image.open(img_path_b).convert("RGB")
    except Exception:
        return []

    # Run detector on both
    scene_a = detector.perceive(rgb_image=img_a, width=width, height=height)
    scene_b = detector.perceive(rgb_image=img_b, width=width, height=height)

    dets_a = []
    for dobj in scene_a.objects:
        bbox = dobj.attributes.get("bbox")
        dets_a.append({
            "label": dobj.label,
            "color": dobj.color,
            "confidence": dobj.confidence,
            "bbox": bbox,
            "center_px": dobj.attributes.get("center_px"),
            "center_norm": dobj.attributes.get("center_norm"),
        })

    dets_b = []
    for dobj in scene_b.objects:
        bbox = dobj.attributes.get("bbox")
        dets_b.append({
            "label": dobj.label,
            "color": dobj.color,
            "confidence": dobj.confidence,
            "bbox": bbox,
            "center_px": dobj.attributes.get("center_px"),
            "center_norm": dobj.attributes.get("center_norm"),
        })

    # Match by color
    matches = _match_detections_by_color(dets_a, dets_b)

    # Get oracle objects
    oracle_objects = item.get("scene_info", {}).get("objects", [])
    oracle_by_color: dict[str, dict] = {}
    for obj in oracle_objects:
        c = _norm_color(obj.get("color", ""))
        pos = obj.get("position")
        if pos and c:
            oracle_by_color[c] = {
                "object_id": obj.get("object_id", ""),
                "oracle_position": _to_xyz(pos),
            }

    records = []
    for ia, ib in matches:
        da = dets_a[ia]
        db = dets_b[ib]
        color_a = _norm_color(da.get("color", "") or da.get("label", ""))

        pixel_a = _extract_pixel_from_detection(da, mode=pixel_mode)
        pixel_b = _extract_pixel_from_detection(db, mode=pixel_mode)

        record = {
            "frame_idx": item.get("_frame_idx", 0),
            "color": color_a,
            "label_a": da.get("label"),
            "label_b": db.get("label"),
            "confidence_a": da.get("confidence"),
            "confidence_b": db.get("confidence"),
            "pixel_a": list(pixel_a) if pixel_a else None,
            "pixel_b": list(pixel_b) if pixel_b else None,
            "pixel_mode": pixel_mode,
        }

        oracle = oracle_by_color.get(color_a, {})
        record["object_id"] = oracle.get("object_id", "")

        if pixel_a is None or pixel_b is None:
            record["failure"] = "no_pixel"
            records.append(record)
            continue

        # Build rays and triangulate
        ray_a_origin_cam, ray_a_dir_cam = pixel_to_camera_ray(
            pixel_a[0], pixel_a[1], calib_a.K,
        )
        ray_b_origin_cam, ray_b_dir_cam = pixel_to_camera_ray(
            pixel_b[0], pixel_b[1], calib_b.K,
        )

        ray_a_origin, ray_a_dir = transform_ray_to_world(
            ray_a_origin_cam, ray_a_dir_cam, calib_a.T_camera_world,
        )
        ray_b_origin, ray_b_dir = transform_ray_to_world(
            ray_b_origin_cam, ray_b_dir_cam, calib_b.T_camera_world,
        )

        tri_result = triangulate_rays(
            ray_a_origin, ray_a_dir, ray_b_origin, ray_b_dir,
        )

        if tri_result is None:
            record["failure"] = "triangulation_failed"
            records.append(record)
            continue

        record["triangulated_position"] = {
            "x": float(tri_result[0]),
            "y": float(tri_result[1]),
            "z": float(tri_result[2]),
        }

        # Compute residual (distance between closest-approach points)
        denom = 1.0 - np.dot(ray_a_dir, ray_b_dir) ** 2
        if abs(denom) > 1e-12:
            w = ray_a_origin - ray_b_origin
            b = np.dot(ray_a_dir, ray_b_dir)
            d = np.dot(ray_a_dir, w)
            e = np.dot(ray_b_dir, w)
            t1 = (b * e - d) / denom
            t2 = (e - b * d) / denom
            p1 = ray_a_origin + t1 * ray_a_dir
            p2 = ray_b_origin + t2 * ray_b_dir
            record["triangulation_residual"] = float(np.linalg.norm(p1 - p2))
            record["t1"] = float(t1)
            record["t2"] = float(t2)

        # Compare against oracle
        oracle_pos = oracle.get("oracle_position")
        if oracle_pos is not None:
            record["oracle_position"] = {
                "x": float(oracle_pos[0]),
                "y": float(oracle_pos[1]),
                "z": float(oracle_pos[2]),
            }
            dx = float(tri_result[0] - oracle_pos[0])
            dy = float(tri_result[1] - oracle_pos[1])
            dz = float(tri_result[2] - oracle_pos[2])
            record["x_error"] = dx
            record["y_error"] = dy
            record["z_error"] = dz
            record["xy_error"] = float(np.hypot(dx, dy))
            record["xyz_error"] = float(np.sqrt(dx*dx + dy*dy + dz*dz))

        records.append(record)

    return records


def _to_xyz(position) -> np.ndarray:
    if isinstance(position, dict):
        return np.array([position["x"], position["y"], position["z"]], dtype=np.float64)
    return np.array(position[:3], dtype=np.float64)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(records: list[dict]) -> dict:
    xy_errors = [r["xy_error"] for r in records if "xy_error" in r]
    xyz_errors = [r["xyz_error"] for r in records if "xyz_error" in r]
    residuals = [r["triangulation_residual"] for r in records
                 if "triangulation_residual" in r and r["triangulation_residual"] is not None]
    x_errs = [r["x_error"] for r in records if "x_error" in r]
    y_errs = [r["y_error"] for r in records if "y_error" in r]
    z_errs = [r["z_error"] for r in records if "z_error" in r]

    fail_count = sum(1 for r in records if "failure" in r)
    tri_fail_count = sum(1 for r in records if r.get("failure") == "triangulation_failed")
    no_pixel_count = sum(1 for r in records if r.get("failure") == "no_pixel")

    result: dict[str, Any] = {
        "num_objects_total": len(records),
        "num_valid_triangulations": len(xy_errors),
        "failure_count": fail_count,
        "triangulation_fail_count": tri_fail_count,
        "no_pixel_count": no_pixel_count,
    }

    if xy_errors:
        arr = np.array(xy_errors)
        result["mean_xy_error"] = float(np.mean(arr))
        result["median_xy_error"] = float(np.median(arr))
        result["p90_xy_error"] = float(np.percentile(arr, 90))
        result["max_xy_error"] = float(np.max(arr))
        result["min_xy_error"] = float(np.min(arr))

    if xyz_errors:
        arr = np.array(xyz_errors)
        result["mean_xyz_error"] = float(np.mean(arr))
        result["median_xyz_error"] = float(np.median(arr))

    if residuals:
        arr = np.array(residuals)
        result["triangulation_residual_mean"] = float(np.mean(arr))
        result["triangulation_residual_median"] = float(np.median(arr))
        result["triangulation_residual_p90"] = float(np.percentile(arr, 90))
        result["triangulation_residual_max"] = float(np.max(arr))

    if x_errs:
        result["per_axis_abs_error_mean"] = {
            "x": float(np.mean(np.abs(x_errs))),
            "y": float(np.mean(np.abs(y_errs))),
            "z": float(np.mean(np.abs(z_errs))),
        }
        result["per_axis_abs_error_median"] = {
            "x": float(np.median(np.abs(x_errs))),
            "y": float(np.median(np.abs(y_errs))),
            "z": float(np.median(np.abs(z_errs))),
        }

    # Per-color breakdown
    by_color: dict[str, list[float]] = {}
    for r in records:
        if "xy_error" in r:
            c = r.get("color", "unknown")
            by_color.setdefault(c, []).append(r["xy_error"])
    result["per_color"] = {}
    for c, errs in by_color.items():
        arr = np.array(errs)
        result["per_color"][c] = {
            "count": len(errs),
            "mean_xy_error": float(np.mean(arr)),
            "median_xy_error": float(np.median(arr)),
        }

    return result


# ---------------------------------------------------------------------------
# Validation data generation (dual-camera)
# ---------------------------------------------------------------------------


def _generate_dual_camera_val_data(
    num_frames: int,
    camera_a: str,
    camera_b: str,
    detector_model_path: str,
    detector_conf_threshold: float,
    width: int,
    height: int,
) -> list[dict]:
    """Generate paired validation frames from a live Robosuite environment."""
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.perception.detector_perception import RGBDetectorPerception
    from PIL import Image
    import tempfile
    import os

    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        render_mode="offscreen",
        camera_names=[camera_a, camera_b],
    )

    detector = RGBDetectorPerception(
        model_path=Path(detector_model_path),
        conf_threshold=detector_conf_threshold,
    )

    instructions = []
    try:
        env.reset()
        for i in range(num_frames):
            # Random action
            action = env.env.action_space.sample()
            env.env.step(action)

            # Render from both cameras
            img_a = env.render_rgb(camera=camera_a, width=width, height=height)
            img_b = env.render_rgb(camera=camera_b, width=width, height=height)

            # Save to temp files
            import uuid
            tmpdir = Path(tempfile.gettempdir()) / "vlm_nsm_dual_cam_val"
            tmpdir.mkdir(parents=True, exist_ok=True)
            path_a = tmpdir / f"frame_{i:04d}_{camera_a}.png"
            path_b = tmpdir / f"frame_{i:04d}_{camera_b}.png"
            Image.fromarray(img_a).save(path_a)
            Image.fromarray(img_b).save(path_b)

            # Get oracle positions
            objects = []
            for canonical in env.cube_names:
                try:
                    pose = env.get_object_pose(canonical)
                    color = canonical.replace("_main", "").lower()
                    # Map cube names to colors used in eval
                    color_map = {"cube1": "red", "cube2": "green", "cubea": "red", "cubeb": "green"}
                    c = "unknown"
                    for k, v in color_map.items():
                        if k in canonical.lower() or k in color.lower():
                            c = v
                            break
                    if c == "unknown":
                        c = color
                    objects.append({
                        "object_id": canonical,
                        "category": "block",
                        "color": c,
                        "position": [float(pose[0]), float(pose[1]), float(pose[2])],
                    })
                except Exception:
                    pass

            instructions.append({
                "instruction": f"Frame {i}",
                "image_path": str(path_a),
                "image_path_frontview": str(path_b),
                "scene_info": {"objects": objects},
                "_frame_idx": i,
            })

            if (i + 1) % 10 == 0:
                print(f"  Collected {i + 1}/{num_frames}")

    finally:
        env.env.close()

    return instructions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_dual_camera(
    eval_input: str | None = None,
    dataset_root: str | None = None,
    generate_val_data: int = 0,
    camera_a: str = "agentview",
    camera_b: str = "frontview",
    camera_calibration_a: str = "",
    camera_calibration_b: str = "",
    detector_model_path: str = "",
    detector_conf_threshold: float = 0.15,
    pixel_mode: str = "center",
    width: int = 512,
    height: int = 512,
    max_images: int | None = None,
    output: str | None = None,
    save_debug_jsonl: str | None = None,
) -> dict:
    """Run dual-camera triangulation pose error evaluation."""

    # Load calibration
    calib_a = load_calibration(camera_calibration_a)
    calib_b = load_calibration(camera_calibration_b)
    print(f"Camera A ({camera_a}): {calib_a.camera_name}")
    print(f"Camera B ({camera_b}): {calib_b.camera_name}")

    # Load instructions
    if generate_val_data > 0:
        print(f"\nGenerating {generate_val_data} paired validation frames...")
        instructions = _generate_dual_camera_val_data(
            num_frames=generate_val_data,
            camera_a=camera_a,
            camera_b=camera_b,
            detector_model_path=detector_model_path,
            detector_conf_threshold=detector_conf_threshold,
            width=width,
            height=height,
        )
        print(f"Done: {len(instructions)} frames collected.")
    elif eval_input:
        with open(eval_input) as f:
            instructions = json.load(f)
        # Add frame_idx if missing
        for i, item in enumerate(instructions):
            if "_frame_idx" not in item:
                item["_frame_idx"] = i
    elif dataset_root:
        print("ERROR: dataset_root mode not yet implemented. Use --eval_input or --generate_val_data.")
        sys.exit(1)
    else:
        print("ERROR: Must provide --eval_input, --dataset_root, or --generate_val_data.")
        sys.exit(1)

    if max_images:
        instructions = instructions[:max_images]

    print(f"Eval instructions: {len(instructions)}")

    # Load detector
    from vlm_nsm.perception.detector_perception import RGBDetectorPerception
    detector_path = Path(detector_model_path)
    if not detector_path.exists():
        print(f"ERROR: Detector not found at {detector_path}")
        sys.exit(1)
    detector = RGBDetectorPerception(
        model_path=detector_path, conf_threshold=detector_conf_threshold,
    )
    print(f"Detector loaded: {detector.model_name}")
    print(f"Pixel mode: {pixel_mode}")
    print()

    # Process frames
    all_records = []
    for idx, item in enumerate(instructions):
        records = _process_pair(
            item, detector, calib_a, calib_b, pixel_mode, width, height,
        )
        all_records.extend(records)
        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(instructions)}] {len(all_records)} records")

    # Aggregate
    metrics = _aggregate(all_records)
    metrics["camera_a"] = camera_a
    metrics["camera_b"] = camera_b
    metrics["calibration_a"] = camera_calibration_a
    metrics["calibration_b"] = camera_calibration_b
    metrics["num_frames"] = len(instructions)
    metrics["pixel_mode"] = pixel_mode

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Dual-Camera Triangulation ({camera_a} + {camera_b})")
    print(f"  Pixel mode: {pixel_mode}")
    print(f"  Frames: {len(instructions)}")
    print(f"  Valid triangulations: {metrics.get('num_valid_triangulations', 0)}/{metrics.get('num_objects_total', 0)}")
    if "median_xy_error" in metrics:
        print(f"  Median XY error: {metrics['median_xy_error']:.4f} m")
    if "mean_xy_error" in metrics:
        print(f"  Mean XY error:   {metrics['mean_xy_error']:.4f} m")
    if "p90_xy_error" in metrics:
        print(f"  P90 XY error:    {metrics['p90_xy_error']:.4f} m")
    if "triangulation_residual_median" in metrics:
        print(f"  Residual median: {metrics['triangulation_residual_median']:.6f} m")
    if "triangulation_residual_mean" in metrics:
        print(f"  Residual mean:   {metrics['triangulation_residual_mean']:.6f} m")

    # Save metrics
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nMetrics saved to {out}")

    # Save debug JSONL
    if save_debug_jsonl:
        out = Path(save_debug_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Debug records saved to {out} ({len(all_records)} records)")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-camera stereo triangulation pose error",
    )
    parser.add_argument("--eval_input", default=None,
                        help="Path to eval input JSON with paired image paths")
    parser.add_argument("--dataset_root", default=None,
                        help="Path to dataset directory")
    parser.add_argument("--generate_val_data", type=int, default=0,
                        help="Generate N paired frames from live Robosuite")
    parser.add_argument("--camera_a", default="agentview")
    parser.add_argument("--camera_b", default="frontview")
    parser.add_argument("--camera_calibration_a",
                        default="generated/configs/camera_calibration_agentview_512.json")
    parser.add_argument("--camera_calibration_b",
                        default="generated/configs/camera_calibration_frontview_512.json")
    parser.add_argument("--detector_model_path",
                        default="generated/models/rgb_perception_yolov8n_medium/best.pt")
    parser.add_argument("--detector_conf_threshold", type=float, default=0.15)
    parser.add_argument("--pixel_mode", default="center",
                        choices=["center", "bottom_center"])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_debug_jsonl", default=None)
    args = parser.parse_args()

    evaluate_dual_camera(
        eval_input=args.eval_input,
        dataset_root=args.dataset_root,
        generate_val_data=args.generate_val_data,
        camera_a=args.camera_a,
        camera_b=args.camera_b,
        camera_calibration_a=args.camera_calibration_a,
        camera_calibration_b=args.camera_calibration_b,
        detector_model_path=args.detector_model_path,
        detector_conf_threshold=args.detector_conf_threshold,
        pixel_mode=args.pixel_mode,
        width=args.width,
        height=args.height,
        max_images=args.max_images,
        output=args.output,
        save_debug_jsonl=args.save_debug_jsonl,
    )


if __name__ == "__main__":
    main()
