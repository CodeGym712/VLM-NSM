"""Analyze RGB dataset labels and compute statistics.

Scans YOLO-format labels and optional annotations.jsonl to produce
per-class and per-image distribution metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_labels(
    dataset_root: str | Path,
    output: str | None = None,
) -> dict:
    """Analyze YOLO labels in a dataset directory.

    Args:
        dataset_root: Root directory with labels/ subdirs.
        output: Optional JSON path for output.

    Returns:
        Dict of label statistics.
    """
    root = Path(dataset_root)

    # Gather all label files
    label_dirs = sorted(root.glob("labels/*/"))
    all_label_paths: list[Path] = []
    for ld in label_dirs:
        all_label_paths.extend(sorted(ld.glob("*.txt")))

    num_images = len(all_label_paths)
    if num_images == 0:
        stats = {
            "dataset_root": str(root),
            "num_images": 0,
            "num_label_files": 0,
            "num_objects": 0,
            "error": "no label files found",
        }
        return _write_output(stats, output)

    # Parse labels
    class_counts: dict[int, int] = {}
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    objects_per_image: list[int] = []
    yolo_coord_out_of_range: int = 0

    for label_path in all_label_paths:
        lines = label_path.read_text().strip().splitlines()
        objs_in_image = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            objs_in_image += 1

            w_px = w  # normalized
            h_px = h
            bbox_widths.append(w_px)
            bbox_heights.append(h_px)
            bbox_areas.append(w_px * h_px)

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                yolo_coord_out_of_range += 1

        objects_per_image.append(objs_in_image)

    total_objects = sum(class_counts.values())

    # Compute distribution
    class_distribution = {
        str(k): {
            "class_id": k,
            "count": v,
            "percentage": round(100.0 * v / total_objects, 1) if total_objects else 0,
        }
        for k, v in sorted(class_counts.items())
    }

    # Bbox statistics (normalized coords)
    bbox_stats = {}
    if bbox_widths:
        bbox_stats = {
            "width_mean": round(float(np.mean(bbox_widths)), 4),
            "width_std": round(float(np.std(bbox_widths)), 4),
            "height_mean": round(float(np.mean(bbox_heights)), 4),
            "height_std": round(float(np.std(bbox_heights)), 4),
            "area_mean": round(float(np.mean(bbox_areas)), 4),
            "area_std": round(float(np.std(bbox_areas)), 4),
        }

    # Objects per image
    obj_per_img_stats = {}
    if objects_per_image:
        obj_per_img_stats = {
            "mean": round(float(np.mean(objects_per_image)), 2),
            "std": round(float(np.std(objects_per_image)), 2),
            "min": int(min(objects_per_image)),
            "max": int(max(objects_per_image)),
        }

    # Count missing / invalid bboxes via annotations if available
    anno_path = root / "annotations.jsonl"
    missing_bbox = 0
    invalid_bbox = 0
    if anno_path.exists():
        for line in anno_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            try:
                ann = json.loads(line)
            except json.JSONDecodeError:
                continue
            for obj_entry in ann.get("objects", []):
                b = obj_entry.get("bbox")
                if b is None:
                    missing_bbox += 1
                elif (b.get("x2", 0) - b.get("x1", 0)) <= 0 or (b.get("y2", 0) - b.get("y1", 0)) <= 0:
                    invalid_bbox += 1

    stats: dict = {
        "dataset_root": str(root),
        "num_images": num_images,
        "num_label_files": len(all_label_paths),
        "num_objects": total_objects,
        "class_distribution": class_distribution,
        "bbox_stats_normalized": bbox_stats,
        "objects_per_image": obj_per_img_stats,
        "missing_bbox_count": missing_bbox,
        "invalid_bbox_count": invalid_bbox,
        "yolo_coord_out_of_range_count": yolo_coord_out_of_range,
    }

    return _write_output(stats, output)


def _write_output(stats: dict, output: str | None) -> dict:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze RGB dataset labels")
    parser.add_argument("--dataset_root", required=True,
                        help="Root directory of the RGB dataset")
    parser.add_argument("--output", default=None,
                        help="Output JSON path for label statistics")
    args = parser.parse_args()

    stats = analyze_labels(args.dataset_root, args.output)

    print(f"Label analysis for {args.dataset_root}:")
    print(f"  Images: {stats['num_images']}")
    print(f"  Objects: {stats['num_objects']}")
    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    print(f"  Class distribution:")
    for cid_str, cdata in stats.get("class_distribution", {}).items():
        print(f"    Class {cid_str}: {cdata['count']} ({cdata['percentage']}%)")

    bs = stats.get("bbox_stats_normalized", {})
    if bs:
        print(f"  Bbox (normalized):")
        print(f"    width:  mean={bs['width_mean']:.4f}  std={bs['width_std']:.4f}")
        print(f"    height: mean={bs['height_mean']:.4f}  std={bs['height_std']:.4f}")

    opi = stats.get("objects_per_image", {})
    if opi:
        print(f"  Objects/image: mean={opi['mean']:.1f}  range=[{opi['min']}, {opi['max']}]")

    if stats.get("yolo_coord_out_of_range_count", 0) > 0:
        print(f"  WARNING: {stats['yolo_coord_out_of_range_count']} coords out of [0,1]")
    if stats.get("missing_bbox_count", 0) > 0:
        print(f"  Missing bboxes: {stats['missing_bbox_count']}")
    if stats.get("invalid_bbox_count", 0) > 0:
        print(f"  Invalid bboxes: {stats['invalid_bbox_count']}")

    if stats.get("num_images", 0) > 0:
        ratio = stats["num_objects"] / stats["num_images"]
        print(f"  Objects/image ratio: {ratio:.2f}")


if __name__ == "__main__":
    main()
