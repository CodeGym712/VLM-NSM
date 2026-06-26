#!/usr/bin/env python3
"""Debug camera calibration by comparing forward-projected oracle positions
against detected bbox pixels.

For each instruction in the scene grounding eval input:
1. Run YOLO detector on the image
2. Forward-project the oracle 3D position to a pixel using extracted calibration
3. Compare projected pixel vs detected bbox centre / bottom_centre
4. Output aggregate metrics and per-object debug JSONL

A large systematic pixel offset (>> 10 px) indicates calibration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from vlm_nsm.perception.camera_geometry import project_world_point_to_pixel
from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
from vlm_nsm.perception.rgb_table_projection import get_projection_pixel_from_bbox


def _to_xyz(position) -> np.ndarray:
    """Convert a position (list or dict) to a 3-vector."""
    if isinstance(position, dict):
        return np.array(
            [position["x"], position["y"], position["z"]],
            dtype=np.float64,
        )
    return np.array(position[:3], dtype=np.float64)


def load_calibration(path: str) -> CameraInfo:
    """Load a CameraInfo from a JSON calibration file."""
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


def load_detector(model_path: str, conf_threshold: float = 0.15):
    """Load YOLO detector. Returns None if model not found."""
    from vlm_nsm.perception.detector_perception import RGBDetectorPerception
    p = Path(model_path)
    if not p.exists():
        print(f"WARNING: Detector model not found at {p}, will skip detection.")
        return None
    return RGBDetectorPerception(model_path=p, conf_threshold=conf_threshold)


def color_map_for_matching(detected_label: str) -> str:
    """Normalize YOLO label/color to simple color name."""
    label = detected_label.lower().replace(" ", "_")
    mapping = {
        "red": "red", "red_cube": "red",
        "green": "green", "green_cube": "green",
        "blue": "blue", "blue_cube": "blue",
    }
    return mapping.get(label, label)


def check_consistency(
    eval_input_path: str,
    calibration_agentview: str,
    calibration_secondary: str | None,
    detector_model_path: str,
    detector_conf_threshold: float = 0.15,
    secondary_camera_name: str = "frontview",
    table_z: float = 0.8,
    output_metrics: str | None = None,
    output_debug_jsonl: str | None = None,
    max_images: int | None = None,
) -> dict:
    """Run the calibration consistency check.

    Returns aggregate metrics dict.
    """
    # Load inputs
    with open(eval_input_path) as f:
        instructions = json.load(f)
    if max_images:
        instructions = instructions[:max_images]

    calib_agent = load_calibration(calibration_agentview)
    calib_secondary = load_calibration(calibration_secondary) if calibration_secondary else None
    detector = load_detector(detector_model_path, detector_conf_threshold)

    if detector is not None:
        print(f"Detector loaded: {detector.model_name}")
    print(f"Camera (agentview): {calib_agent.camera_name}")
    if calib_secondary:
        print(f"Camera ({secondary_camera_name}): {calib_secondary.camera_name}")
    print(f"Eval instructions: {len(instructions)}")
    print()

    all_records: list[dict] = []

    for idx, item in enumerate(instructions):
        scene_info = item.get("scene_info", {})
        oracle_objects = scene_info.get("objects", [])
        image_path = item.get("image_path", "")
        image_path_secondary = None
        if calib_secondary:
            image_path_secondary = item.get(
                f"image_path_{secondary_camera_name}", ""
            )

        # Run detector on primary image
        detections: list[dict] = []
        if detector is not None and image_path:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            scene = detector.perceive(
                rgb_image=img,
                width=calib_agent.width,
                height=calib_agent.height,
            )
            for dobj in scene.objects:
                detections.append({
                    "label": dobj.label,
                    "color": dobj.color,
                    "confidence": dobj.confidence,
                    "bbox": dobj.attributes.get("bbox"),
                    "center_px": dobj.attributes.get("center_px"),
                    "center_norm": dobj.attributes.get("center_norm"),
                })

        # Run detector on secondary image
        detections_secondary: list[dict] = []
        if detector is not None and calib_secondary and image_path_secondary:
            from PIL import Image
            img2 = Image.open(image_path_secondary).convert("RGB")
            scene2 = detector.perceive(
                rgb_image=img2,
                width=calib_secondary.width,
                height=calib_secondary.height,
            )
            for dobj in scene2.objects:
                detections_secondary.append({
                    "label": dobj.label,
                    "color": dobj.color,
                    "confidence": dobj.confidence,
                    "bbox": dobj.attributes.get("bbox"),
                    "center_px": dobj.attributes.get("center_px"),
                    "center_norm": dobj.attributes.get("center_norm"),
                })

        # Match oracle objects to detections by color, check projection
        for oracle_obj in oracle_objects:
            oid = oracle_obj.get("object_id", "")
            oracle_color = oracle_obj.get("color", "")
            oracle_pos = oracle_obj.get("position")
            if oracle_pos is None or not oracle_color:
                continue

            oracle_xyz = _to_xyz(oracle_pos)

            # ---- Agentview ----
            match_agent = _find_match(oracle_color, detections)
            record = _make_record(
                idx, oid, oracle_color, oracle_xyz,
                match_agent, calib_agent, "agentview", table_z,
            )

            # ---- Secondary camera ----
            if calib_secondary:
                match_sec = _find_match(oracle_color, detections_secondary)
                record_sec = _make_record(
                    idx, oid, oracle_color, oracle_xyz,
                    match_sec, calib_secondary, secondary_camera_name, table_z,
                )
                # Merge into one record
                for k, v in record_sec.items():
                    if k not in record:
                        record[k] = v
                record[f"{secondary_camera_name}_detected"] = (
                    match_sec is not None
                )

            all_records.append(record)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(instructions)}] {len(all_records)} objects")

    # Aggregate
    metrics = _aggregate(all_records)
    metrics["num_instructions"] = len(instructions)
    metrics["calibration_agentview"] = calibration_agentview
    if calib_secondary:
        metrics[f"calibration_{secondary_camera_name}"] = calibration_secondary

    # Print summary
    print(f"\n{'=' * 60}")
    for cam_label in metrics.get("cameras", []):
        _print_camera_summary(metrics, cam_label)

    # Save metrics
    if output_metrics:
        out = Path(output_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nMetrics saved to {out}")

    # Save debug JSONL
    if output_debug_jsonl:
        out = Path(output_debug_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Debug records saved to {out}  ({len(all_records)} records)")

    return metrics


def _find_match(oracle_color: str, detections: list[dict]) -> dict | None:
    """Find the detection matching the oracle object by color."""
    for det in detections:
        det_color = color_map_for_matching(det.get("color", "") or det.get("label", ""))
        if det_color == oracle_color:
            return det
    return None


def _make_record(
    frame_idx: int,
    object_id: str,
    oracle_color: str,
    oracle_xyz: np.ndarray,
    detection: dict | None,
    camera_info: CameraInfo,
    camera_label: str,
    table_z: float,
) -> dict[str, Any]:
    """Create a per-object debug record for one camera."""
    cam_key = f"{camera_label}_"
    record: dict[str, Any] = {
        "frame_idx": frame_idx,
        "object_id": object_id,
        "oracle_color": oracle_color,
        "oracle_position": {
            "x": float(oracle_xyz[0]),
            "y": float(oracle_xyz[1]),
            "z": float(oracle_xyz[2]),
        },
    }

    # Forward-project oracle position to pixel
    proj_pixel = project_world_point_to_pixel(
        oracle_xyz, camera_info.K, camera_info.T_camera_world,
    )
    if proj_pixel is not None:
        record[f"{cam_key}projected_u"] = proj_pixel[0]
        record[f"{cam_key}projected_v"] = proj_pixel[1]
        record[f"{cam_key}projected_valid"] = True
    else:
        record[f"{cam_key}projected_u"] = None
        record[f"{cam_key}projected_v"] = None
        record[f"{cam_key}projected_valid"] = False
        record[f"{cam_key}projected_failure"] = "behind_camera"

    record[f"{cam_key}detected"] = detection is not None

    if detection is not None and proj_pixel is not None:
        bbox = detection.get("bbox")
        detected_center_u: float | None = None
        detected_center_v: float | None = None
        detected_bottom_u: float | None = None
        detected_bottom_v: float | None = None

        # Center pixel
        center_px = detection.get("center_px")
        if center_px and "x" in center_px and "y" in center_px:
            detected_center_u = float(center_px["x"])
            detected_center_v = float(center_px["y"])
        elif bbox:
            detected_center_u, detected_center_v = get_projection_pixel_from_bbox(
                bbox, mode="center"
            )

        # Bottom-center pixel
        if bbox:
            detected_bottom_u, detected_bottom_v = get_projection_pixel_from_bbox(
                bbox, mode="bottom_center"
            )

        record[f"{cam_key}detected_center_u"] = detected_center_u
        record[f"{cam_key}detected_center_v"] = detected_center_v
        record[f"{cam_key}detected_bottom_u"] = detected_bottom_u
        record[f"{cam_key}detected_bottom_v"] = detected_bottom_v
        record[f"{cam_key}bbox"] = bbox

        # Pixel errors
        if detected_center_u is not None:
            record[f"{cam_key}center_pixel_error"] = float(np.hypot(
                proj_pixel[0] - detected_center_u,
                proj_pixel[1] - detected_center_v,
            ))
        if detected_bottom_u is not None:
            record[f"{cam_key}bottom_pixel_error"] = float(np.hypot(
                proj_pixel[0] - detected_bottom_u,
                proj_pixel[1] - detected_bottom_v,
            ))

    elif detection is not None:
        # Detection exists but projection failed
        record[f"{cam_key}detected_center_u"] = None
        record[f"{cam_key}projected_failure"] = "behind_camera"

    return record


def _aggregate(records: list[dict]) -> dict:
    """Compute aggregate metrics from per-object records."""
    metrics: dict[str, Any] = {
        "num_objects_total": len(records),
        "cameras": [],
    }

    # Discover camera labels from record keys
    camera_labels = set()
    for r in records:
        for k in r:
            if k.endswith("_projected_valid"):
                camera_labels.add(k.replace("_projected_valid", ""))
    camera_labels = sorted(camera_labels)

    for cam_label in camera_labels:
        proj_key = f"{cam_label}_projected_valid"
        center_err_key = f"{cam_label}_center_pixel_error"
        bottom_err_key = f"{cam_label}_bottom_pixel_error"
        detected_key = f"{cam_label}_detected"

        # Count valid projections
        valid_count = sum(1 for r in records if r.get(proj_key))
        detected_count = sum(1 for r in records if r.get(detected_key))

        center_errors = [r[center_err_key] for r in records
                         if r.get(center_err_key) is not None]
        bottom_errors = [r[bottom_err_key] for r in records
                         if r.get(bottom_err_key) is not None]

        cam_metrics: dict[str, Any] = {
            "camera": cam_label,
            "num_valid_projections": valid_count,
            "num_detected": detected_count,
        }

        if center_errors:
            arr = np.array(center_errors)
            cam_metrics["center_pixel_error"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "std": float(np.std(arr)),
            }

        if bottom_errors:
            arr = np.array(bottom_errors)
            cam_metrics["bottom_pixel_error"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "std": float(np.std(arr)),
            }

        metrics["cameras"].append(cam_metrics)

    # Summary verdict
    verdicts = []
    for cm in metrics["cameras"]:
        cerr = cm.get("center_pixel_error", {})
        median_center = cerr.get("median", float("inf"))
        cam = cm["camera"]
        if median_center < 10:
            verdicts.append(f"{cam}: GOOD (median center error {median_center:.1f} px)")
        elif median_center < 50:
            verdicts.append(f"{cam}: SUSPICIOUS (median center error {median_center:.1f} px)")
        else:
            verdicts.append(f"{cam}: BAD (median center error {median_center:.1f} px)")

    metrics["verdicts"] = verdicts
    return metrics


def _print_camera_summary(metrics: dict, cam_label: str):
    """Print summary for one camera."""
    for cm in metrics.get("cameras", []):
        if cm.get("camera") != cam_label:
            continue
        print(f"\nCamera: {cam_label}")
        print(f"  Valid projections: {cm['num_valid_projections']}")
        print(f"  Detected: {cm['num_detected']}")
        cerr = cm.get("center_pixel_error")
        if cerr:
            print(f"  Center pixel error: "
                  f"mean={cerr['mean']:.1f} px, "
                  f"median={cerr['median']:.1f} px, "
                  f"p90={cerr['p90']:.1f} px, "
                  f"max={cerr['max']:.1f} px")
        berr = cm.get("bottom_pixel_error")
        if berr:
            print(f"  Bottom pixel error: "
                  f"mean={berr['mean']:.1f} px, "
                  f"median={berr['median']:.1f} px, "
                  f"p90={berr['p90']:.1f} px, "
                  f"max={berr['max']:.1f} px")


def main():
    parser = argparse.ArgumentParser(
        description="Debug camera calibration: compare projected oracle positions"
                    " against detected bbox pixels",
    )
    parser.add_argument(
        "--eval_input",
        default="generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json",
    )
    parser.add_argument(
        "--calibration_agentview",
        default="generated/configs/camera_calibration_agentview_512.json",
    )
    parser.add_argument(
        "--calibration_frontview",
        default=None,
        help="Secondary camera calibration JSON (e.g. frontview)",
    )
    parser.add_argument(
        "--secondary_camera_name",
        default="frontview",
    )
    parser.add_argument(
        "--detector_model_path",
        default="generated/models/rgb_perception_yolov8n_medium/best.pt",
    )
    parser.add_argument("--detector_conf_threshold", type=float, default=0.15)
    parser.add_argument("--table_z", type=float, default=0.8)
    parser.add_argument("--output_metrics", default=None)
    parser.add_argument("--output_debug_jsonl", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    # Default output paths
    if args.output_metrics is None:
        args.output_metrics = (
            "generated/results/stage_18/"
            "camera_projection_consistency_metrics.json"
        )
    if args.output_debug_jsonl is None:
        args.output_debug_jsonl = (
            "generated/results/stage_18/"
            "camera_projection_consistency_debug.jsonl"
        )

    check_consistency(
        eval_input_path=args.eval_input,
        calibration_agentview=args.calibration_agentview,
        calibration_secondary=args.calibration_frontview,
        detector_model_path=args.detector_model_path,
        detector_conf_threshold=args.detector_conf_threshold,
        secondary_camera_name=args.secondary_camera_name,
        table_z=args.table_z,
        output_metrics=args.output_metrics,
        output_debug_jsonl=args.output_debug_jsonl,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
