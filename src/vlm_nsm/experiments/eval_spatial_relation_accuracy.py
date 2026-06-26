#!/usr/bin/env python3
"""Evaluate spatial relation accuracy of estimated positions vs oracle positions.

Compares left/right, front/behind, and near/far relations derived from estimated
positions against those derived from oracle positions.

Input: one or more pose-error debug JSONL files containing per-object records
with estimated_position (or triangulated_position) and oracle_position.

No DashScope/Qwen required.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Relation computation
# ---------------------------------------------------------------------------


def _compute_pairwise_relations(
    positions: dict[str, dict[str, float]],
    xy_threshold: float = 0.02,
    near_threshold: float = 0.05,
    far_threshold: float = 0.15,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute pairwise spatial relations for a set of objects.

    Args:
        positions: dict mapping object_id -> {"x": ..., "y": ...}
        xy_threshold: minimum abs(dx) or abs(dy) for directional relations (m)
        near_threshold: distance below which objects are "near"
        far_threshold: distance above which objects are "far"

    Returns:
        dict mapping (obj_a, obj_b) -> {"left/right": ..., "front/behind": ...,
        "near/far": ..., "ambiguous_x": bool, "ambiguous_y": bool}
    """
    obj_ids = sorted(positions.keys())
    relations: dict[tuple[str, str], dict[str, Any]] = {}

    for i, a in enumerate(obj_ids):
        for b in obj_ids[i + 1:]:
            pa = positions[a]
            pb = positions[b]
            dx = pb["x"] - pa["x"]
            dy = pb["y"] - pa["y"]
            dist = float(np.hypot(dx, dy))

            rel: dict[str, Any] = {"distance": dist}

            # Directional X
            if abs(dx) >= xy_threshold:
                rel["b_wrt_a_x"] = "right" if dx > 0 else "left"
            else:
                rel["b_wrt_a_x"] = "ambiguous"

            # Directional Y
            if abs(dy) >= xy_threshold:
                rel["b_wrt_a_y"] = "behind" if dy > 0 else "front"
            else:
                rel["b_wrt_a_y"] = "ambiguous"

            # Distance
            if dist <= near_threshold:
                rel["distance_label"] = "near"
            elif dist >= far_threshold:
                rel["distance_label"] = "far"
            else:
                rel["distance_label"] = "mid"

            relations[(a, b)] = rel

    return relations


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------


def _evaluate_relation_accuracy(
    oracle_relations: dict,
    estimated_relations: dict,
) -> dict[str, Any]:
    """Compare estimated relations against oracle relations.

    Returns per-category accuracy metrics.
    """
    left_right_total = 0
    left_right_correct = 0
    front_back_total = 0
    front_back_correct = 0
    near_far_total = 0
    near_far_correct = 0

    ambiguous_oracle_x = 0
    ambiguous_oracle_y = 0
    ambiguous_estimated_x = 0
    ambiguous_estimated_y = 0

    wrong_left_right = 0
    wrong_front_back = 0
    wrong_near_far = 0

    for pair_key, oracle_rel in oracle_relations.items():
        est_rel = estimated_relations.get(pair_key)
        if est_rel is None:
            continue

        # X direction
        oracle_x = oracle_rel.get("b_wrt_a_x")
        est_x = est_rel.get("b_wrt_a_x")
        if oracle_x != "ambiguous" and est_x != "ambiguous":
            left_right_total += 1
            if oracle_x == est_x:
                left_right_correct += 1
            else:
                wrong_left_right += 1
        elif oracle_x == "ambiguous":
            ambiguous_oracle_x += 1
            if est_x != "ambiguous":
                wrong_left_right += 1
        elif est_x == "ambiguous":
            ambiguous_estimated_x += 1

        # Y direction
        oracle_y = oracle_rel.get("b_wrt_a_y")
        est_y = est_rel.get("b_wrt_a_y")
        if oracle_y != "ambiguous" and est_y != "ambiguous":
            front_back_total += 1
            if oracle_y == est_y:
                front_back_correct += 1
            else:
                wrong_front_back += 1
        elif oracle_y == "ambiguous":
            ambiguous_oracle_y += 1
            if est_y != "ambiguous":
                wrong_front_back += 1
        elif est_y == "ambiguous":
            ambiguous_estimated_y += 1

        # Distance
        oracle_d = oracle_rel.get("distance_label")
        est_d = est_rel.get("distance_label")
        if oracle_d is not None and est_d is not None:
            near_far_total += 1
            if oracle_d == est_d:
                near_far_correct += 1
            else:
                wrong_near_far += 1

    return {
        "left_right_accuracy": left_right_correct / left_right_total
        if left_right_total > 0 else None,
        "left_right_total": left_right_total,
        "left_right_correct": left_right_correct,
        "front_back_accuracy": front_back_correct / front_back_total
        if front_back_total > 0 else None,
        "front_back_total": front_back_total,
        "front_back_correct": front_back_correct,
        "near_far_accuracy": near_far_correct / near_far_total
        if near_far_total > 0 else None,
        "near_far_total": near_far_total,
        "near_far_correct": near_far_correct,
        "ambiguous_oracle_x_count": ambiguous_oracle_x,
        "ambiguous_oracle_y_count": ambiguous_oracle_y,
        "ambiguous_estimated_x_count": ambiguous_estimated_x,
        "ambiguous_estimated_y_count": ambiguous_estimated_y,
        "wrong_left_right_count": wrong_left_right,
        "wrong_front_back_count": wrong_front_back,
        "wrong_near_far_count": wrong_near_far,
        "overall_accuracy": (
            (left_right_correct + front_back_correct + near_far_correct)
            / (left_right_total + front_back_total + near_far_total)
        ) if (left_right_total + front_back_total + near_far_total) > 0 else None,
        "total_comparable_pairs": (
            left_right_total + front_back_total + near_far_total
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_spatial_relation_accuracy(
    estimated_pose_jsonl: list[str],
    output: str | None = None,
    xy_threshold: float = 0.02,
    near_threshold: float = 0.05,
    far_threshold: float = 0.15,
    position_key: str = "triangulated_position",
) -> dict:
    """Evaluate spatial relation accuracy from pose error debug JSONL files.

    Args:
        estimated_pose_jsonl: list of paths to debug JSONL files
        output: output JSON path
        xy_threshold: min abs(dx/dy) for directional relations (m)
        near_threshold: distance below which objects are "near" (m)
        far_threshold: distance above which objects are "far" (m)
        position_key: key in JSONL records for estimated position

    Returns:
        dict with accuracy metrics.
    """
    # Load all records and group by frame
    frames: dict[int, dict[str, dict]] = defaultdict(dict)  # frame_idx -> {obj_id: {oracle, estimated}}

    for jsonl_path in estimated_pose_jsonl:
        p = Path(jsonl_path)
        if not p.exists():
            print(f"WARNING: {jsonl_path} not found, skipping")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)

                frame_idx = r.get("frame_idx", 0)
                oracle_pos = r.get("oracle_position")
                est_pos = r.get(position_key)

                if oracle_pos is None or est_pos is None:
                    continue

                obj_id = r.get("object_id") or r.get("color") or str(hash(str(oracle_pos)))
                frames[frame_idx][obj_id] = {
                    "oracle": oracle_pos,
                    "estimated": est_pos,
                }

    if not frames:
        print("ERROR: No valid frame data found in input files")
        return {"error": "no_data"}

    print(f"Loaded {sum(len(objs) for objs in frames.values())} objects across {len(frames)} frames")

    # Evaluate per-frame and aggregate
    all_lr_acc = []
    all_fb_acc = []
    all_nf_acc = []
    total_pairs = 0
    total_correct = 0
    total_ambiguous_oracle_x = 0
    total_ambiguous_oracle_y = 0
    total_wrong_lr = 0
    total_wrong_fb = 0
    total_wrong_nf = 0

    for frame_idx in sorted(frames.keys()):
        objs = frames[frame_idx]

        oracle_positions = {oid: data["oracle"] for oid, data in objs.items()}
        estimated_positions = {oid: data["estimated"] for oid, data in objs.items()}

        oracle_rels = _compute_pairwise_relations(
            oracle_positions, xy_threshold, near_threshold, far_threshold,
        )
        estimated_rels = _compute_pairwise_relations(
            estimated_positions, xy_threshold, near_threshold, far_threshold,
        )

        acc = _evaluate_relation_accuracy(oracle_rels, estimated_rels)

        if acc["left_right_accuracy"] is not None:
            all_lr_acc.append(acc["left_right_accuracy"])
        if acc["front_back_accuracy"] is not None:
            all_fb_acc.append(acc["front_back_accuracy"])
        if acc["near_far_accuracy"] is not None:
            all_nf_acc.append(acc["near_far_accuracy"])

        total_pairs += acc["total_comparable_pairs"]
        total_correct += (
            acc["left_right_correct"] + acc["front_back_correct"] + acc["near_far_correct"]
        )
        total_ambiguous_oracle_x += acc["ambiguous_oracle_x_count"]
        total_ambiguous_oracle_y += acc["ambiguous_oracle_y_count"]
        total_wrong_lr += acc["wrong_left_right_count"]
        total_wrong_fb += acc["wrong_front_back_count"]
        total_wrong_nf += acc["wrong_near_far_count"]

    result = {
        "num_frames_with_pairs": len([f for f in frames.values() if len(f) >= 2]),
        "num_frames_total": len(frames),
        "mean_left_right_accuracy": float(np.mean(all_lr_acc)) if all_lr_acc else None,
        "mean_front_back_accuracy": float(np.mean(all_fb_acc)) if all_fb_acc else None,
        "mean_near_far_accuracy": float(np.mean(all_nf_acc)) if all_nf_acc else None,
        "overall_relation_accuracy": total_correct / total_pairs if total_pairs > 0 else None,
        "total_comparable_pairs": total_pairs,
        "total_correct_pairs": total_correct,
        "total_wrong_left_right": total_wrong_lr,
        "total_wrong_front_back": total_wrong_fb,
        "total_wrong_near_far": total_wrong_nf,
        "total_ambiguous_oracle_x": total_ambiguous_oracle_x,
        "total_ambiguous_oracle_y": total_ambiguous_oracle_y,
        "xy_threshold": xy_threshold,
        "near_threshold": near_threshold,
        "far_threshold": far_threshold,
        "near_far_note": "near/far reference is relative pairwise distance between objects, "
                         "not absolute reference to robot/camera",
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("Spatial Relation Accuracy")
    print(f"  Frames with >=2 objects: {result['num_frames_with_pairs']}/{result['num_frames_total']}")
    print(f"  Total comparable pairs: {total_pairs}")
    if result["mean_left_right_accuracy"] is not None:
        print(f"  Left/Right accuracy:   {result['mean_left_right_accuracy']:.3f}")
    if result["mean_front_back_accuracy"] is not None:
        print(f"  Front/Behind accuracy: {result['mean_front_back_accuracy']:.3f}")
    if result["mean_near_far_accuracy"] is not None:
        print(f"  Near/Far accuracy:     {result['mean_near_far_accuracy']:.3f}")
    if result["overall_relation_accuracy"] is not None:
        print(f"  Overall accuracy:      {result['overall_relation_accuracy']:.3f}")
    print(f"  Wrong left/right: {total_wrong_lr}")
    print(f"  Wrong front/back: {total_wrong_fb}")
    print(f"  Wrong near/far:   {total_wrong_nf}")

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nMetrics saved to {out}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial relation accuracy of estimated vs oracle positions",
    )
    parser.add_argument("--estimated_pose_jsonl", nargs="+", default=[],
                        help="One or more pose error debug JSONL files")
    parser.add_argument("--output", default=None,
                        help="Output metrics JSON path")
    parser.add_argument("--xy_threshold", type=float, default=0.02,
                        help="Min abs(dx/dy) for directional relations (m)")
    parser.add_argument("--near_threshold", type=float, default=0.05,
                        help="Distance below which objects are 'near' (m)")
    parser.add_argument("--far_threshold", type=float, default=0.15,
                        help="Distance above which objects are 'far' (m)")
    parser.add_argument("--position_key", default="triangulated_position",
                        help="Key for estimated position in JSONL records")
    args = parser.parse_args()

    if not args.estimated_pose_jsonl:
        print("ERROR: --estimated_pose_jsonl is required")
        sys.exit(1)

    evaluate_spatial_relation_accuracy(
        estimated_pose_jsonl=args.estimated_pose_jsonl,
        output=args.output,
        xy_threshold=args.xy_threshold,
        near_threshold=args.near_threshold,
        far_threshold=args.far_threshold,
        position_key=args.position_key,
    )


if __name__ == "__main__":
    main()
