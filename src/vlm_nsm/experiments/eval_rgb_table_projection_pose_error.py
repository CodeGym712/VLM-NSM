"""Evaluate RGB bbox table-plane projection accuracy against oracle sim poses.

Compares estimated 3D positions (from camera calibration + bbox projection)
against ground-truth oracle positions from the Robosuite simulation state.

Two data sources:
1. **Pre-existing annotations** – when the dataset's ``annotations.jsonl``
   contains non-null ``position`` fields (oracle sim poses).
2. **On-the-fly generation** – when ``--generate_val_data N`` is passed,
   the script creates a Robosuite environment, captures N random frames, and
   uses the sim state as ground truth.

No DashScope/Qwen API required.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# ── Annotation loading ────────────────────────────────────────────────────


def _load_annotations(dataset_root: Path) -> list[dict]:
    """Load annotations from ``dataset_root / annotations.jsonl``."""
    path = dataset_root / "annotations.jsonl"
    if not path.exists():
        print(f"ERROR: annotations not found at {path}")
        sys.exit(1)
    annotations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations


def _check_positions(annotations: list[dict]) -> bool:
    """Return True if any annotation object has non-null position."""
    for ann in annotations[:50]:
        for obj in ann.get("objects", []):
            if obj.get("position") is not None:
                return True
    return False


def _describe_required_fields():
    print(
        "ERROR: No oracle positions found in annotations.\n"
        "\n"
        "The annotations must contain per-object sim positions, e.g.:\n"
        '  {"objects": [{"object_id": "cube1", "position": {"x": 0.1, "y": -0.05, "z": 0.8}, ...}]}\n'
        "\n"
        "Options to provide oracle positions:\n"
        "  1. Add --generate_val_data N to create N validation frames\n"
        "     from a live Robosuite environment (requires GPU + robosuite).\n"
        "  2. Re-generate the dataset with a modified collector that\n"
        "     saves sim positions in annotations.jsonl.\n"
    )


# ── Validation data generation ────────────────────────────────────────────


def get_sim_table_z(env, table_thickness: float = 0.05) -> float:
    """Detect the table-surface Z from a Robosuite simulation.

    Returns the body-centre Z plus half the (default) table thickness
    to approximate the surface height.
    """
    sim = env.env.sim
    for i in range(sim.model.nbody):
        name = str(sim.model.body_names[i])
        if "table" in name.lower():
            body_z = float(sim.data.body_xpos[i][2])
            return body_z + table_thickness / 2.0
    return 0.0


def generate_validation_data(
    num_frames: int,
    camera: str = "agentview",
    width: int = 512,
    height: int = 512,
    detector_model_path: str | None = None,
    detector_conf_threshold: float = 0.15,
) -> list[dict]:
    """Generate validation frames with oracle sim positions.

    Creates a Robosuite Stack environment, runs random actions, and at
    each step renders RGB + records sim poses.

    Returns a list of per-frame dicts consumable by ``evaluate_annotations``.
    """
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
        from vlm_nsm.perception.detector_perception import RGBDetectorPerception
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: {e} — cannot generate validation data without robosuite.")
        sys.exit(1)

    print(f"Generating {num_frames} validation frames via Robosuite...")
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        render_mode="offscreen",
        control_freq=20,
    )

    # Detect table surface Z
    detected_table_z = get_sim_table_z(env)
    print(f"  Detected table surface at z={detected_table_z:.3f}")

    # Detector (optional — if not provided, store only oracle info)
    detector = None
    if detector_model_path:
        detector_path = Path(detector_model_path)
        if detector_path.exists():
            detector = RGBDetectorPerception(
                model_path=detector_path,
                conf_threshold=detector_conf_threshold,
            )
            print(f"  Detector loaded: {detector.model_name}")

    frames: list[dict] = []
    np.random.seed(42)
    env.reset()
    cubes = env.cube_names

    for frame_idx in range(num_frames):
        # Step with random action
        action = np.random.uniform(-1, 1, size=7).tolist()
        env.step(action)

        # Render RGB
        rgb_array = env.render_rgb(camera=camera, height=height, width=width)
        if rgb_array is None:
            continue

        # Get oracle poses
        objects_gt: list[dict] = []
        for cn in cubes:
            try:
                pose = env.get_object_pose(cn)
            except (ValueError, KeyError):
                pose = None
            color = {"cube1": "red", "cube2": "green"}.get(cn, "unknown")
            objects_gt.append({
                "object_id": cn,
                "color": color,
                "position": pose,
            })

        # Run detector
        detections: list[dict] = []
        if detector is not None:
            from vlm_nsm.perception.data_types import SceneInfo
            scene = detector.perceive(rgb_image=rgb_array, width=width, height=height)
            for dobj in scene.objects:
                detections.append({
                    "label": dobj.label,
                    "color": dobj.color,
                    "confidence": dobj.confidence,
                    "bbox": dobj.attributes.get("bbox"),
                    "center_px": dobj.attributes.get("center_px"),
                    "center_norm": dobj.attributes.get("center_norm"),
                })

        frames.append({
            "frame_idx": frame_idx,
            "image_width": width,
            "image_height": height,
            "camera_name": camera,
            "objects_gt": objects_gt,
            "detections": detections,
            "table_z": detected_table_z,
        })

        if (frame_idx + 1) % 10 == 0:
            print(f"  Collected {frame_idx + 1}/{num_frames}")

    env.env.close()
    print(f"Done: {len(frames)} frames collected.")
    return frames


def save_validation_frames(frames: list[dict], output_dir: Path) -> None:
    """Save generated validation frames as JSONL (one frame per line)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_data.jsonl"
    with open(path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame, ensure_ascii=False) + "\n")
    print(f"Validation data saved to {path}  ({len(frames)} frames)")


def load_validation_frames(path: Path) -> list[dict]:
    """Load validation frames from JSONL."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


# ── Evaluation ────────────────────────────────────────────────────────────


def _match_by_color(
    detected_color: str,
    gt_color: str,
) -> bool:
    """Match detected object to ground-truth by colour.

    Handles common colour-name variations.
    """
    color_map = {"red": "red", "green": "green", "blue": "blue",
                 "red_cube": "red", "green_cube": "green", "blue_cube": "blue"}
    return color_map.get(detected_color, detected_color) == color_map.get(gt_color, gt_color)


def evaluate_frame(
    frame: dict,
    camera_info,
    detector: Any = None,
    projection_point: str = "center",
    table_z: float = 0.0,
    object_height: float = 0.05,
) -> list[dict]:
    """Evaluate projection error for one frame.

    Args:
        frame: Dict with ``objects_gt`` (oracle positions), and either
            ``detections`` or stored image/detector info.
        camera_info: CameraInfo object (K, T_camera_world).
        detector: Optional RGBDetectorPerception (if detections not
            pre-computed).
        projection_point: Bbox projection mode.
        table_z: Table surface Z.
        object_height: Object height for pose Z offset.

    Returns:
        List of per-object result dicts.
    """
    from vlm_nsm.perception.camera_geometry import pixel_to_table_point
    from vlm_nsm.perception.rgb_table_projection import get_projection_pixel_from_bbox

    results: list[dict] = []

    objects_gt = frame.get("objects_gt", [])
    detections = frame.get("detections", [])

    # Build GT lookup by colour
    gt_by_color: dict[str, dict] = {}
    for obj in objects_gt:
        color = obj.get("color", "")
        if obj.get("position") is not None and color:
            gt_by_color[color] = obj

    dets_to_process = detections

    # If no detections but detector is provided, run inference
    if not dets_to_process and detector is not None:
        image_path = frame.get("image_path")
        if image_path:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            width = frame.get("image_width", 512)
            height = frame.get("image_height", 512)
            from vlm_nsm.perception.detector_perception import RGBDetectorPerception  # noqa
            scene = detector.perceive(rgb_image=img, width=width, height=height)
            dets_to_process = [
                {
                    "label": dobj.label,
                    "color": dobj.color,
                    "confidence": dobj.confidence,
                    "bbox": dobj.attributes.get("bbox"),
                    "center_px": dobj.attributes.get("center_px"),
                    "center_norm": dobj.attributes.get("center_norm"),
                }
                for dobj in scene.objects
            ]
        else:
            # No detections and no detector: skip
            return results

    for det in dets_to_process:
        bbox = det.get("bbox")
        color = det.get("color", "")
        label = det.get("label", "")
        confidence = det.get("confidence", 1.0)

        # Determine projection pixel
        if bbox is not None:
            try:
                u, v = get_projection_pixel_from_bbox(bbox, mode=projection_point)
            except ValueError:
                continue
        else:
            cp = det.get("center_px")
            if cp is not None and "x" in cp and "y" in cp:
                u = float(cp["x"])
                v = float(cp["y"])
            else:
                cn = det.get("center_norm")
                if cn is not None and "x" in cn and "y" in cn:
                    u = float(cn["x"] * camera_info.width)
                    v = float(cn["y"] * camera_info.height)
                else:
                    continue

        # Project to table plane
        point = pixel_to_table_point(u, v, camera_info.K, camera_info.T_camera_world, table_z)

        record: dict[str, Any] = {
            "color": color,
            "label": label,
            "bbox": bbox,
            "projection_pixel": [float(u), float(v)],
            "projection_point": projection_point,
            "detector_confidence": confidence,
        }

        if point is not None:
            est_pose = {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(table_z + object_height / 2.0),
            }
            record["estimated_position"] = est_pose
        else:
            record["estimated_position"] = None
            record["projection_failed"] = True

        # Match with GT
        gt_match = gt_by_color.get(color)
        if gt_match is not None:
            gt_pos = gt_match["position"]
            record["oracle_position"] = gt_pos

            if point is not None and gt_pos is not None:
                dx = float(point[0]) - float(gt_pos["x"])
                dy = float(point[1]) - float(gt_pos["y"])
                dz = (table_z + object_height / 2.0) - float(gt_pos["z"])
                record["x_error"] = dx
                record["y_error"] = dy
                record["z_error"] = dz
                record["xy_error"] = np.hypot(dx, dy)
                record["xyz_error"] = np.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            record["match_failed"] = True

        results.append(record)

    return results


def aggregate_results(per_object_results: list[dict]) -> dict:
    """Compute aggregate metrics from per-object results."""
    xy_errors = []
    x_errors = []
    y_errors = []
    per_color: dict[str, dict] = defaultdict(lambda: {"count": 0, "errors": []})
    projection_fail = 0
    match_fail = 0

    for r in per_object_results:
        if r.get("projection_failed"):
            projection_fail += 1
            continue
        if r.get("match_failed"):
            match_fail += 1
            continue

        xy_error = r.get("xy_error")
        if xy_error is not None:
            xy_errors.append(xy_error)
            x_errors.append(r.get("x_error", 0))
            y_errors.append(r.get("y_error", 0))
            color = r.get("color", "unknown")
            per_color[color]["count"] += 1
            per_color[color]["errors"].append(xy_error)

    if not xy_errors:
        return {
            "num_images": 0,
            "num_objects": 0,
            "mean_xy_error": None,
            "median_xy_error": None,
            "p90_xy_error": None,
            "max_xy_error": None,
            "failure_count": projection_fail + match_fail,
            "projection_fail_count": projection_fail,
            "detection_miss_count": 0,
            "match_fail_count": match_fail,
            "per_color": {},
        }

    xy_arr = np.array(xy_errors)
    per_color_agg = {}
    for color, data in per_color.items():
        per_color_agg[color] = {
            "count": data["count"],
            "mean_xy_error": float(np.mean(data["errors"])),
            "median_xy_error": float(np.median(data["errors"])),
            "p90_xy_error": float(np.percentile(data["errors"], 90)),
            "max_xy_error": float(np.max(data["errors"])),
        }

    return {
        "num_objects": len(xy_errors),
        "mean_xy_error": float(np.mean(xy_arr)),
        "median_xy_error": float(np.median(xy_arr)),
        "p90_xy_error": float(np.percentile(xy_arr, 90)),
        "max_xy_error": float(np.max(xy_arr)),
        "mean_x_error": float(np.mean(x_errors)),
        "mean_y_error": float(np.mean(y_errors)),
        "failure_count": projection_fail + match_fail,
        "projection_fail_count": projection_fail,
        "detection_miss_count": 0,
        "match_fail_count": match_fail,
        "per_color": dict(per_color_agg),
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB bbox table-plane projection pose error"
    )
    parser.add_argument("--dataset_root", default=None,
                        help="Path to RGB perception dataset (with annotations.jsonl)")
    parser.add_argument("--camera_calibration", required=True)
    parser.add_argument("--detector_model_path", default=None)
    parser.add_argument("--detector_conf_threshold", type=float, default=0.15)
    parser.add_argument("--camera_name", default="agentview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--table_z", type=float, default=0.0)
    parser.add_argument("--object_height", type=float, default=0.05)
    parser.add_argument("--projection_point", default="center",
                        choices=["center", "bottom_center", "weighted_bottom"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_debug_jsonl", default=None)
    parser.add_argument("--generate_val_data", type=int, default=0,
                        help="Generate N validation frames from Robosuite (no GPU needed for small N)")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    # Load camera calibration
    calib_path = Path(args.camera_calibration)
    if not calib_path.exists():
        print(f"ERROR: Camera calibration not found: {calib_path}")
        sys.exit(1)
    calib = json.loads(calib_path.read_text())
    from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
    K = np.array(calib["K"], dtype=np.float64)
    T = np.array(calib["T_camera_world"], dtype=np.float64)
    camera_info = CameraInfo(
        K=K, T_camera_world=T,
        width=calib.get("width", 512),
        height=calib.get("height", 512),
        camera_name=calib.get("camera_name", "agentview"),
    )

    # Load detector
    detector = None
    if args.detector_model_path:
        detector_path = Path(args.detector_model_path)
        if detector_path.exists():
            from vlm_nsm.perception.detector_perception import RGBDetectorPerception
            detector = RGBDetectorPerception(
                model_path=detector_path,
                conf_threshold=args.detector_conf_threshold,
            )
            print(f"Detector loaded: {detector.model_name}")

    # Data source: generate or load
    frames: list[dict] = []

    if args.generate_val_data > 0:
        print(f"Generating {args.generate_val_data} validation frames...")
        frames = generate_validation_data(
            num_frames=args.generate_val_data,
            camera=args.camera_name,
            width=args.width,
            height=args.height,
            detector_model_path=args.detector_model_path,
            detector_conf_threshold=args.detector_conf_threshold,
        )
    elif args.dataset_root:
        dataset_root = Path(args.dataset_root)
        annotations = _load_annotations(dataset_root)
        if not _check_positions(annotations):
            _describe_required_fields()
            print("\nTip: use --generate_val_data N to create validation data on the fly.")
            sys.exit(1)

        # Validate image paths
        image_dir = dataset_root / "images"
        for ann in annotations:
            image_path = ann.get("image_path", "")
            if image_path:
                ann["image_fullpath"] = str(image_dir / image_path)

        if args.max_images:
            annotations = annotations[:args.max_images]

        # Convert annotations to frame format
        for ann in annotations:
            objects_gt = []
            for obj in ann.get("objects", []):
                objects_gt.append({
                    "object_id": obj["object_id"],
                    "color": obj.get("color"),
                    "position": obj.get("position"),
                })
            frames.append({
                "image_path": ann.get("image_fullpath", ""),
                "image_width": ann.get("image_width", 512),
                "image_height": ann.get("image_height", 512),
                "camera_name": ann.get("camera_name", args.camera_name),
                "objects_gt": objects_gt,
                "detections": [],
            })
        print(f"Loaded {len(frames)} frames from dataset")
    else:
        print("ERROR: Provide --dataset_root or --generate_val_data N.")
        sys.exit(1)

    # Evaluate
    all_results: list[dict] = []
    for i, frame in enumerate(frames):
        # Use per-frame table_z when available (generated data), else CLI arg
        frame_table_z = frame.get("table_z", args.table_z)
        frame_results = evaluate_frame(
            frame, camera_info, detector=detector,
            projection_point=args.projection_point,
            table_z=frame_table_z,
            object_height=args.object_height,
        )
        for r in frame_results:
            r["frame_idx"] = i
        all_results.extend(frame_results)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames, "
                  f"{len(all_results)} objects so far")

    metrics = aggregate_results(all_results)
    metrics["dataset_root"] = str(args.dataset_root) if args.dataset_root else "generated"
    metrics["detector_model_path"] = args.detector_model_path
    metrics["camera_calibration"] = args.camera_calibration
    metrics["projection_point"] = args.projection_point
    metrics["num_images"] = len(frames)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Projection mode: {args.projection_point}")
    print(f"Images: {metrics['num_images']}, Objects: {metrics['num_objects']}")
    print(f"Mean XY error:   {metrics['mean_xy_error']:.4f} m" if metrics['mean_xy_error'] is not None else "Mean XY error:   N/A")
    print(f"Median XY error: {metrics['median_xy_error']:.4f} m" if metrics['median_xy_error'] is not None else "Median XY error: N/A")
    print(f"P90 XY error:    {metrics['p90_xy_error']:.4f} m" if metrics['p90_xy_error'] is not None else "P90 XY error:    N/A")
    print(f"Max XY error:    {metrics['max_xy_error']:.4f} m" if metrics['max_xy_error'] is not None else "Max XY error:    N/A")
    if metrics["per_color"]:
        for color, cdata in metrics["per_color"].items():
            print(f"  {color}: {cdata['count']} objects, mean error={cdata['mean_xy_error']:.4f} m")
    if metrics["failure_count"] > 0:
        print(f"  Failures: {metrics['failure_count']} "
              f"(proj={metrics['projection_fail_count']}, "
              f"match={metrics['match_fail_count']})")

    # Save metrics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics saved to {output_path}")

    # Save debug JSONL
    if args.save_debug_jsonl:
        debug_path = Path(args.save_debug_jsonl)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Debug records saved to {debug_path} ({len(all_results)} records)")

    return metrics


if __name__ == "__main__":
    main()
