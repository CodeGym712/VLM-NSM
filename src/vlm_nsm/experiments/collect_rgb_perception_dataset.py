"""Collect RGB perception dataset with oracle labels from Robosuite.

Renders RGB images from the simulation camera and annotates them with
ground-truth object state using the oracle perception module.

Supports YOLO label generation (approx bbox from 3D pose projection)
and split-aware output organisation.

No model training — creates the labelled dataset only.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from vlm_nsm.perception.oracle_perception import perceive_scene
from vlm_nsm.perception.bbox_utils import BBox, bbox_to_yolo, approximate_bbox_from_center
from vlm_nsm.perception.yolo_labels import write_yolo_label, write_yolo_yaml
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper


# Default box size (pixels) for approx bbox generation at 512x512
DEFAULT_BOX_SIZE = (80, 80)

# Simple linear mapping from sim coords to image coords (agentview camera)
# These are rough approximations for the agentview camera in Robosuite Stack env.
# sim x: [-0.3, 0.3] → image x: [50, width-50]
# sim y: [-0.3, 0.3] → image y: [height-50, 50] (flipped)
# sim z is ignored for 2D projection but used to adjust box size
_SIM_X_RANGE = (-0.35, 0.35)
_SIM_Y_RANGE = (-0.30, 0.30)


def _estimate_2d_center(
    sim_pose: dict[str, float] | None,
    width: int,
    height: int,
    default_center: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Estimate 2D image center from 3D sim pose using linear projection.

    Args:
        sim_pose: Dict with 'x', 'y', 'z' keys in sim coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
        default_center: Fallback center if pose is None.

    Returns:
        (cx, cy) in pixel coordinates.
    """
    if default_center:
        return default_center
    if sim_pose is None:
        return (width / 2.0, height / 2.0)

    sx = sim_pose.get("x", 0.0)
    sy = sim_pose.get("y", 0.0)

    # Linear mapping: sim → pixel
    x_range = _SIM_X_RANGE[1] - _SIM_X_RANGE[0]
    y_range = _SIM_Y_RANGE[1] - _SIM_Y_RANGE[0]

    # Clamp to range
    sx = max(_SIM_X_RANGE[0], min(_SIM_X_RANGE[1], sx))
    sy = max(_SIM_Y_RANGE[0], min(_SIM_Y_RANGE[1], sy))

    # Normalize to [0, 1]
    nx = (sx - _SIM_X_RANGE[0]) / x_range if x_range > 0 else 0.5
    ny = (sy - _SIM_Y_RANGE[0]) / y_range if y_range > 0 else 0.5

    # Map to pixel coords (y is flipped: sim y+ = away from robot = up in image)
    px = 50 + nx * (width - 100)
    py = (height - 50) - ny * (height - 100)

    return (px, py)


def _random_action() -> np.ndarray:
    """Generate a random 7-DoF action for scene variation."""
    action = np.random.uniform(low=-1.0, high=1.0, size=7).astype(np.float32)
    action[6] = -1.0  # keep gripper open by default
    return action


def _discover_cube_geom_ids(env) -> dict[str, int]:
    """Discover visual geom IDs for each cube in the simulation.

    Maps canonical cube name (e.g. 'cube1') to its visual geom ID
    for segmentation-based bbox extraction.

    Returns:
        dict: {canonical_name: visual_geom_id}
    """
    sim = env.env.sim
    # Build reverse mapping: body_name -> canonical_name
    body_to_canonical: dict[str, str] = {}
    for cn in env.cube_names:
        body = env._cube_to_body.get(cn)
        if body:
            body_to_canonical[body] = cn

    cube_geom_ids: dict[str, int] = {}
    for i in range(sim.model.ngeom):
        name = str(sim.model.geom_names[i])
        if name.startswith("cube") and (name.endswith("_vis") or "_g" in name):
            body_id = sim.model.geom_bodyid[i]
            body_name = str(sim.model.body_names[body_id])
            canonical = body_to_canonical.get(body_name)
            if canonical:
                cube_geom_ids[canonical] = i
                continue
            # Fallback: match by prefix
            for cn in env.cube_names:
                body = env._cube_to_body.get(cn, "")
                if body and body in body_name:
                    cube_geom_ids[cn] = i

    return cube_geom_ids


def _segmentation_bboxes(
    env,
    camera: str,
    height: int,
    width: int,
    cube_geom_ids: dict[str, int],
) -> dict[str, BBox]:
    """Extract pixel bboxes from segmentation rendering.

    Renders segmentation mask and computes axis-aligned bbox for each
    cube by finding min/max pixel coordinates where the cube's geom mask
    is non-zero.

    Returns:
        dict: {canonical_name: BBox} for visible cubes only.
    """
    seg = env.env.sim.render(
        camera_name=camera, height=height, width=width, segmentation=True,
    )
    geom_ids_img = seg[:, :, 1]  # channel 1 = visual geom ID
    bboxes: dict[str, BBox] = {}
    for cn, geom_id in cube_geom_ids.items():
        mask = (geom_ids_img == geom_id)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        bboxes[cn] = BBox(
            x1=int(xs.min()), y1=int(ys.min()),
            x2=int(xs.max()), y2=int(ys.max()),
        )
    return bboxes


def _save_image(rgb_array: np.ndarray, path: Path) -> None:
    """Save an RGB numpy array as PNG."""
    from PIL import Image
    img = Image.fromarray(rgb_array.astype("uint8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format="PNG")


def _write_annotations(annotations: list[dict], path: Path) -> None:
    """Write annotations as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")


def _write_split(
    annotations: list[dict],
    train_ratio: float,
    output_dir: Path,
) -> tuple[int, int]:
    """Write train.txt and val.txt with image filenames."""
    image_paths = list({ann["image_path"] for ann in annotations})
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]

    (output_dir / "train.txt").write_text("\n".join(sorted(train_paths)) + "\n")
    (output_dir / "val.txt").write_text("\n".join(sorted(val_paths)) + "\n")
    return len(train_paths), len(val_paths)


def _draw_bbox_on_image(
    image: np.ndarray,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a bounding box and label on an image (in-place)."""
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle(
        [int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)],
        outline=color,
        width=2,
    )
    draw.text((int(bbox.x1) + 2, int(bbox.y1) + 2), label, fill=color)
    return np.array(pil_img)


def _collect_episode(
    env,
    num_steps: int,
    camera: str,
    height: int,
    width: int,
    episode_id: int,
    output_dir: Path,
    seed: int,
    split: str,
    bbox_source: str,
    save_yolo_labels: bool,
    yolo_class_mode: str,
    visualize: bool,
    vis_dir: Path | None,
) -> list[dict]:
    """Run one episode collecting RGB frames and annotations."""
    np.random.seed(seed + episode_id)
    env.reset()
    annotations: list[dict] = []

    # Discover cube geom IDs for segmentation bbox
    cube_geom_ids: dict[str, int] | None = None
    if bbox_source == "segmentation":
        cube_geom_ids = _discover_cube_geom_ids(env)
        print(f"  Discovered cube geom IDs: {cube_geom_ids}")

    for step_id in range(num_steps):
        rgb = env.render_rgb(camera=camera, height=height, width=width)
        if rgb is None:
            env.step(_random_action())
            continue

        # Oracle perception
        try:
            scene = perceive_scene(env)
        except Exception:
            scene = None

        # Save image (split-aware path)
        image_filename = f"step_{episode_id:06d}_{step_id:06d}.png"
        split_image_dir = output_dir / "images" / split
        image_path = split_image_dir / image_filename
        _save_image(rgb, image_path)

        # Get segmentation bboxes if enabled
        seg_bboxes: dict[str, BBox] | None = None
        if bbox_source == "segmentation" and cube_geom_ids:
            seg_bboxes = _segmentation_bboxes(env, camera, height, width, cube_geom_ids)

        # Build annotation with bbox/center
        obj_list = []
        yolo_objects: list[tuple] = []  # (DetectedObject, BBox)

        if scene:
            for i, obj in enumerate(scene.objects):
                # Non-block objects (table) don't get bboxes
                if obj.category != "block":
                    obj_entry = {
                        "object_id": obj.label,
                        "category": obj.category,
                        "color": obj.color,
                        "position": obj.pose if obj.pose else None,
                        "bbox": None,
                        "center": None,
                        "visibility": "visible",
                    }
                    obj_list.append(obj_entry)
                    continue

                # Bbox generation method
                bbox = None
                bbox_source_val = "none"
                center_px_out = None

                if bbox_source == "segmentation" and seg_bboxes:
                    bbox = seg_bboxes.get(obj.label)
                    if bbox:
                        bbox_source_val = "segmentation"
                        cx = (bbox.x1 + bbox.x2) / 2.0
                        cy = (bbox.y1 + bbox.y2) / 2.0
                        center_px_out = (float(cx), float(cy))

                elif bbox_source == "approx":
                    # Use sim pose → 2D projection
                    center_px = _estimate_2d_center(
                        obj.pose, width, height,
                        default_center=(width * (0.3 + i * 0.4), height / 2),
                    )
                    # Scale box size by z-height (higher = closer = larger)
                    z = obj.pose.get("z", 0.8) if obj.pose else 0.8
                    z_scale = max(0.5, 0.5 + (z - 0.7) * 2.0)
                    box_size = (int(DEFAULT_BOX_SIZE[0] * z_scale),
                                int(DEFAULT_BOX_SIZE[1] * z_scale))
                    bbox = approximate_bbox_from_center(center_px, box_size, width, height)
                    bbox_source_val = "approx"
                    center_px_out = (float(center_px[0]), float(center_px[1]))

                obj_entry = {
                    "object_id": obj.label,
                    "category": obj.category,
                    "color": obj.color,
                    "position": obj.pose if obj.pose else None,
                    "bbox": {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}
                    if bbox else None,
                    "center": list(center_px_out) if center_px_out else None,
                    "visibility": "visible",
                }
                obj_list.append(obj_entry)

                if bbox is not None and obj.category == "block":
                    yolo_objects.append((obj, bbox))

        annotation = {
            "image_path": f"images/{split}/{image_filename}",
            "split": split,
            "episode_id": episode_id,
            "step_id": step_id,
            "camera_name": camera,
            "image_width": width,
            "image_height": height,
            "objects": obj_list,
            "scene_source": "oracle",
            "bbox_source": bbox_source,
            "on_relations": list(scene.on_relations) if scene and scene.on_relations else [],
            "free_gripper": scene.free_gripper if scene else True,
        }
        annotations.append(annotation)

        # Write YOLO label
        if save_yolo_labels and yolo_objects:
            label_dir = output_dir / "labels" / split
            label_path = label_dir / f"step_{episode_id:06d}_{step_id:06d}.txt"
            warnings = write_yolo_label(
                label_path, yolo_objects, width, height,
                class_mode=yolo_class_mode,
            )
            annotation["yolo_label_warnings"] = warnings if warnings else None
            annotation["yolo_label_path"] = str(label_path)

        # Visualization
        has_any_bbox = any(obj_entry.get("bbox") for obj_entry in obj_list)
        if visualize and vis_dir and has_any_bbox:
            vis_path = vis_dir / f"step_{episode_id:06d}_{step_id:06d}.png"
            vis_img = rgb.copy()
            for obj_entry in obj_list:
                b = obj_entry.get("bbox")
                if b:
                    color_map = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
                    col = color_map.get(obj_entry.get("color", ""), (0, 255, 0))
                    vis_img = _draw_bbox_on_image(
                        vis_img,
                        BBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]),
                        label=f"{obj_entry['object_id']}",
                        color=col,
                    )
            _save_image(vis_img, vis_path)

        # Step env
        env.step(_random_action())

    return annotations


def collect_rgb_perception_dataset(
    output_dir: str,
    num_images: int = 100,
    camera: str = "agentview",
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    steps_per_episode: int = 20,
    split: str = "auto",
    save_yolo_labels: bool = False,
    bbox_source: str = "none",
    yolo_class_mode: str = "color_cube",
    visualize: bool = False,
    num_visualizations: int = 20,
) -> dict:
    """Collect an RGB perception dataset from Robosuite.

    Args:
        output_dir: Output directory for dataset.
        num_images: Total number of images to collect.
        camera: Camera name in the sim.
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed for reproducibility.
        steps_per_episode: Number of steps per episode before reset.
        split: 'auto' for train/val split, or 'train'/'val'/'test'.
        save_yolo_labels: Whether to write YOLO-format labels.
        bbox_source: Bbox generation method ('none', 'approx').
        yolo_class_mode: YOLO class mode ('color_cube', 'cube').
        visualize: Whether to generate visualisation images.
        num_visualizations: Max vis images to generate.

    Returns:
        Metadata dict with collection statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve split
    if split == "auto":
        effective_split = "train"  # will be split later; save all under "train"
        train_ratio = 0.8
    else:
        effective_split = split
        train_ratio = 1.0

    # Create dirs
    (output_path / "images" / effective_split).mkdir(parents=True, exist_ok=True)
    if save_yolo_labels:
        (output_path / "labels" / effective_split).mkdir(parents=True, exist_ok=True)

    vis_dir: Path | None = None
    if visualize:
        vis_dir = output_path / "vis"
        vis_dir.mkdir(exist_ok=True)

    random.seed(seed)

    # Initialise env
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        cube_names=["cube1", "cube2"],
        render_mode="offscreen",
    )

    all_annotations: list[dict] = []
    episode_id = 0
    total_frames = 0
    skipped_labels = 0
    start_time = time.time()

    try:
        while total_frames < num_images:
            frames_needed = num_images - total_frames
            ep_steps = min(steps_per_episode, frames_needed)
            annotations = _collect_episode(
                env=env,
                num_steps=ep_steps,
                camera=camera,
                height=height,
                width=width,
                episode_id=episode_id,
                output_dir=output_path,
                seed=seed,
                split=effective_split,
                bbox_source=bbox_source,
                save_yolo_labels=save_yolo_labels,
                yolo_class_mode=yolo_class_mode,
                visualize=visualize and total_frames < num_visualizations,
                vis_dir=vis_dir,
            )
            all_annotations.extend(annotations)
            total_frames += len(annotations)
            episode_id += 1
            print(f"  Episode {episode_id}: collected {len(annotations)} frames "
                  f"({total_frames}/{num_images})")

    except KeyboardInterrupt:
        print("  Collection interrupted by user.")
    finally:
        env.env.close()

    elapsed = time.time() - start_time

    # Count skipped labels
    for ann in all_annotations:
        for obj_entry in ann.get("objects", []):
            if bbox_source != "none" and obj_entry.get("bbox") is None:
                skipped_labels += 1

    # Write annotations
    anno_path = output_path / "annotations.jsonl"
    _write_annotations(all_annotations, anno_path)

    # Train/val split (only in auto mode)
    if split == "auto" and total_frames > 1:
        n_train, n_val = _write_split(all_annotations, train_ratio=0.8, output_dir=output_path)
    else:
        n_train = total_frames if split != "val" else 0
        n_val = total_frames if split == "val" else 0

    # Write yolo.yaml if labels saved
    yaml_path = None
    if save_yolo_labels:
        yaml_path = write_yolo_yaml(output_path, class_mode=yolo_class_mode)

    # Metadata
    meta = {
        "dataset_name": "rgb_perception_v0_1",
        "dataset_version": "0.1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "Robosuite Stack env, oracle labels",
        "total_images": total_frames,
        "image_size": {"height": height, "width": width},
        "camera": camera,
        "num_episodes": episode_id,
        "objects": ["cube1", "cube2", "table"],
        "categories": ["block", "table"],
        "colors": {"cube1": "red", "cube2": "green"},
        "split": {"train": n_train, "val": n_val},
        "has_bbox": bbox_source != "none",
        "has_center": bbox_source != "none",
        "bbox_source": bbox_source,
        "yolo_labels": save_yolo_labels,
        "yolo_class_mode": yolo_class_mode if save_yolo_labels else None,
        "yolo_yaml": yaml_path if yaml_path else None,
        "skipped_labels": skipped_labels,
        "elapsed_seconds": round(elapsed, 1),
        "purpose": "RGB perception training",
    }
    (output_path / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"\nDataset collection complete:")
    print(f"  Images: {total_frames}")
    print(f"  Episodes: {episode_id}")
    print(f"  Train/val: {n_train}/{n_val}")
    print(f"  Bbox source: {bbox_source}")
    print(f"  YOLO labels: {save_yolo_labels}")
    if skipped_labels:
        print(f"  Skipped labels: {skipped_labels}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {output_dir}/")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Collect RGB perception dataset with oracle labels"
    )
    parser.add_argument("--output_dir",
                        default="generated/datasets/rgb_perception_v0_1",
                        help="Output directory for the dataset")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Total number of images to collect (default: 100)")
    parser.add_argument("--camera", default="agentview",
                        help="Camera name (default: agentview)")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width (default: 512)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--steps_per_episode", type=int, default=20,
                        help="Steps per episode before env reset (default: 20)")
    parser.add_argument("--split", default="auto",
                        choices=["auto", "train", "val", "test"],
                        help="Dataset split mode (default: auto)")
    parser.add_argument("--save_yolo_labels", action="store_true", default=False,
                        help="Write YOLO-format label files")
    parser.add_argument("--bbox_source", default="none",
                        choices=["segmentation", "projection", "approx", "none"],
                        help="Bounding box generation method: segmentation (sim mask), approx (projected center), none (default: none)")
    parser.add_argument("--yolo_class_mode", default="color_cube",
                        choices=["color_cube", "cube"],
                        help="YOLO class mapping mode (default: color_cube)")
    parser.add_argument("--visualize_labels", action="store_true", default=False,
                        help="Generate visualisation images with bboxes drawn")
    parser.add_argument("--num_visualizations", type=int, default=20,
                        help="Number of visualisation images (default: 20)")
    args = parser.parse_args()

    collect_rgb_perception_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        camera=args.camera,
        height=args.height,
        width=args.width,
        seed=args.seed,
        steps_per_episode=args.steps_per_episode,
        split=args.split,
        save_yolo_labels=args.save_yolo_labels,
        bbox_source=args.bbox_source,
        yolo_class_mode=args.yolo_class_mode,
        visualize=args.visualize_labels,
        num_visualizations=args.num_visualizations,
    )


if __name__ == "__main__":
    main()
