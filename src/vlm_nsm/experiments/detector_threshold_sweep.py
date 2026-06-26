"""Confidence threshold sweep for YOLO detector on the val set.

Loads a trained YOLO model, runs inference on the validation split at
multiple confidence thresholds, and finds the optimal threshold for
maximizing F1 score.

Usage:
    python3 -m vlm_nsm.experiments.detector_threshold_sweep \
        --model generated/detection/yolov8n_medium/train/weights/best.pt \
        --data generated/datasets/rgb_perception_v0_1_medium/yolo.yaml \
        --output generated/detection/yolov8n_medium/threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def threshold_sweep(
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path | None = None,
    min_conf: float = 0.05,
    max_conf: float = 0.95,
    step: float = 0.05,
    iou_threshold: float = 0.5,
    device: int = 0,
) -> dict:
    """Run confidence threshold sweep on val set.

    Args:
        model_path: Path to trained YOLO weights.
        data_yaml: Path to dataset YAML (for val images and class names).
        output_path: Optional JSON output path.
        min_conf: Minimum confidence threshold.
        max_conf: Maximum confidence threshold.
        step: Step size for threshold sweep.
        iou_threshold: IoU threshold for matching detections to ground truth.
        device: CUDA device ID.

    Returns:
        Dict with sweep results and optimal threshold.
    """
    model = YOLO(str(model_path))
    print(f"Model loaded: {model_path}")
    print(f"Image size: {model.args.imgsz if hasattr(model.args, 'imgsz') else 'N/A'}")

    # Run val once with min_conf to get all detections
    print(f"\nRunning validation with conf={min_conf}...")
    val_results = model.val(
        data=str(data_yaml),
        conf=min_conf,
        iou=iou_threshold,
        imgsz=model.args.imgsz if hasattr(model.args, 'imgsz') else 640,
        plots=False,
        save_json=False,
        device=device,
    )

    # Get per-class AP values from val results
    ap50 = val_results.box.ap50  # (nc_detected,) array
    ap50_95 = val_results.box.ap  # (nc_detected,) array
    class_names = val_results.names
    ap_class = val_results.box.ap_class_index
    if callable(ap_class):
        ap_class = ap_class()
    nc_total = len(class_names)

    print(f"\nClasses in dataset: {nc_total}")
    for idx, cls_id in enumerate(ap_class):
        name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"  Class {cls_id} ({name}): AP50={ap50[idx]:.4f}  AP50-95={ap50_95[idx]:.4f}")

    # Run val at each threshold
    thresholds = np.arange(min_conf, max_conf + 1e-9, step)
    sweep_results: list[dict] = []

    print(f"\nThreshold sweep ({len(thresholds)} thresholds):")
    start = time.time()

    for conf_thresh in thresholds:
        r = model.val(
            data=str(data_yaml),
            conf=float(conf_thresh),
            iou=iou_threshold,
            imgsz=model.args.imgsz if hasattr(model.args, 'imgsz') else 640,
            plots=False,
            save_json=False,
            verbose=False,
            device=device,
        )
        entry = {
            "confidence_threshold": round(float(conf_thresh), 3),
            "precision": round(float(r.box.mp if hasattr(r.box, 'mp') else r.prec[-1]), 4),
            "recall": round(float(r.box.mr if hasattr(r.box, 'mr') else r.rec[-1]), 4),
            "mAP50": round(float(r.box.map50 if hasattr(r.box, 'map50') else r.map50), 4),
            "mAP50_95": round(float(r.box.map if hasattr(r.box, 'map') else r.map), 4),
            "f1": 0.0,
        }
        p = entry["precision"]
        rc = entry["recall"]
        entry["f1"] = round(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0, 4)

        sweep_results.append(entry)
        print(f"  conf={conf_thresh:.2f}  precision={p:.4f}  recall={rc:.4f}  "
              f"mAP50={entry['mAP50']:.4f}  F1={entry['f1']:.4f}")

    elapsed = time.time() - start

    # Find best threshold by F1
    f1_scores = [r["f1"] for r in sweep_results]
    best_idx = int(np.argmax(f1_scores))
    best = sweep_results[best_idx]

    per_class_ap = {}
    for idx, cls_id in enumerate(ap_class):
        name = class_names.get(cls_id, f"class_{cls_id}")
        per_class_ap[name] = {
            "AP50": round(float(ap50[idx]), 4),
            "AP50_95": round(float(ap50_95[idx]), 4),
        }

    output = {
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "num_classes": nc_total,
        "class_names": [class_names.get(i, f"class_{i}") for i in range(nc_total)],
        "per_class_AP": per_class_ap,
        "best_threshold": {
            "confidence": best["confidence_threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "mAP50": best["mAP50"],
            "mAP50_95": best["mAP50_95"],
            "f1": best["f1"],
            "criterion": "max_f1",
        },
        "sweep": sweep_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {output_path}")

    print(f"\n{'=' * 60}")
    print(f"Optimal threshold by F1: conf={best['confidence_threshold']}")
    print(f"  Precision: {best['precision']}")
    print(f"  Recall: {best['recall']}")
    print(f"  F1: {best['f1']}")
    print(f"  mAP50: {best['mAP50']}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Confidence threshold sweep for YOLO detector",
    )
    parser.add_argument("--model", default="generated/detection/yolov8n_medium/train/weights/best.pt",
                        help="Trained YOLO model path")
    parser.add_argument("--data", default="generated/datasets/rgb_perception_v0_1_medium/yolo.yaml",
                        help="Dataset YAML path")
    parser.add_argument("--output", default="generated/detection/yolov8n_medium/threshold_sweep.json",
                        help="Output JSON path")
    parser.add_argument("--min_conf", type=float, default=0.05,
                        help="Minimum confidence threshold (default: 0.05)")
    parser.add_argument("--max_conf", type=float, default=0.95,
                        help="Maximum confidence threshold (default: 0.95)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Step size (default: 0.05)")
    parser.add_argument("--device", default=0,
                        help="CUDA device (default: 0)")
    args = parser.parse_args()

    threshold_sweep(
        model_path=args.model,
        data_yaml=args.data,
        output_path=args.output,
        min_conf=args.min_conf,
        max_conf=args.max_conf,
        step=args.step,
        device=args.device,
    )


if __name__ == "__main__":
    main()
