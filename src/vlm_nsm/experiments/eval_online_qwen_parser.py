"""Parse-only evaluation script for OnlineQwenGoalParser.

Loads instructions from a JSON or text file, runs them through
OnlineQwenGoalParser, optionally runs alignment and planner validation,
and saves results. Does NOT execute in Robosuite.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from vlm_nsm.core.goal_parser import MockGoalParser, extract_json_object
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description
from vlm_nsm.evaluation.object_grounding import evaluate_object_grounding

VALID_PREDICATES = ["on", "clear", "free-gripper", "grasped", "smaller"]
TYPE_TABLE = {
    "cube1": "block", "cube2": "block", "table": "table",
}
PREDICATE_SIGNATURE = {
    "on": [["block", "block"], ["block", "table"]],
    "clear": [["block"]],
    "free-gripper": [[]],
    "grasped": [["block"]],
    "smaller": [["block", "block"], ["block", "table"]],
}
OBJECTS = ["cube1", "cube2", "table"]


def parser_target_stack_from_alignment(align_result, available_objects: list[str]) -> list[str]:
    """Convert aligned on predicates into the planner's bottom-to-top stack."""
    on_predicates = [
        (pred.name, pred.args[0], pred.args[1])
        for pred in align_result.goal_predicates
        if pred.name == "on" and len(pred.args) == 2
    ]
    return target_stack_from_predicates(on_predicates, available_objects)


def load_instructions(path: str) -> list[dict]:
    """Load instructions from a JSON or text file.

    JSON format: list of {"instruction": "...", "task_type": "stacking", ...}
    Text format: one instruction per line.
    """
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "instructions" in data:
                return data["instructions"]
            return [{"instruction": k, **v} for k, v in data.items() if isinstance(v, dict)]
    lines = [line.strip() for line in p.read_text().strip().splitlines() if line.strip()]
    return [{"instruction": line, "task_type": "stacking"} for line in lines]


def main():
    parser = argparse.ArgumentParser(description="Evaluate OnlineQwenGoalParser")
    parser.add_argument("--instructions", required=True,
                        help="Path to instructions JSON or text file")
    parser.add_argument("--output", default="generated/results/online_qwen_eval_results.json",
                        help="Output JSON path")
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen-plus"))
    parser.add_argument("--api_base", default=os.environ.get("QWEN_API_BASE"))
    parser.add_argument("--qwen_api_key_env", default="QWEN_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--available_objects", nargs="+", default=["cube1", "cube2"],
                        help="Available object IDs for the parser")
    parser.add_argument("--save_raw_qwen_outputs", action="store_true", default=False)
    parser.add_argument("--no_alignment", action="store_true", default=False,
                        help="Skip alignment/planner validation")
    parser.add_argument("--scene_inventory", default="none", choices=["none", "oracle", "rgb_detector"],
                        help="Scene inventory mode for object grounding (default: none)")
    parser.add_argument("--detector_model_path",
                        default="generated/models/rgb_perception_yolov8n_smoke/weights/best.pt",
                        help="Path to YOLO weights for rgb_detector mode")
    parser.add_argument("--detector_conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for detector (default: 0.25)")
    parser.add_argument("--dataset_root",
                        default="generated/datasets/rgb_perception_v0_1_smoke",
                        help="Root directory for resolving relative image paths (rgb_detector mode)")
    parser.add_argument("--rgb_spatial_mode", default="image2d",
                        choices=["image2d", "table_projection", "stereo"],
                        help="Spatial relation mode for rgb_detector (default: image2d)")
    parser.add_argument("--camera_calibration",
                        help="Path to agentview camera calibration JSON")
    parser.add_argument("--camera_calibration_frontview",
                        help="Path to frontview camera calibration JSON (required for stereo mode)")
    parser.add_argument("--table_z", type=float, default=0.0,
                        help="Table surface Z in world frame (default: 0.0)")
    parser.add_argument("--object_height", type=float, default=0.05,
                        help="Approximate object height in metres (default: 0.05)")
    parser.add_argument("--projection_point", default="center",
                        choices=["center", "bottom_center", "weighted_bottom"],
                        help="Bbox projection point for table_projection mode (default: center)")
    parser.add_argument("--enable_predicate_inventory", action="store_true",
                        help="Enable predicate-rich scene inventory "
                             "(on_table, robot-relative, edge proximity, pose sanity)")
    parser.add_argument("--scene_reference_frame", default=None,
                        help="Path to scene reference frame JSON "
                             "(optional; uses robosuite Stack defaults if not provided)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    # Load scene reference frame for predicate inventory
    ref_frame = None
    if args.enable_predicate_inventory:
        if args.scene_reference_frame and Path(args.scene_reference_frame).exists():
            from vlm_nsm.perception.scene_context import load_reference_frame
            ref_frame = load_reference_frame(args.scene_reference_frame)
            print(f"Loaded scene reference frame from {args.scene_reference_frame}")
        else:
            from vlm_nsm.perception.scene_context import SceneReferenceFrame
            ref_frame = SceneReferenceFrame()
            print("Using default scene reference frame (robosuite Stack)")

    # Load instructions
    instructions = load_instructions(str(args.instructions))
    print(f"Loaded {len(instructions)} instructions from {args.instructions}")

    # Create parser
    qwen_parser = OnlineQwenGoalParser(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.qwen_api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        save_raw_outputs=args.save_raw_qwen_outputs,
        raw_output_dir=str(output_path.parent / "raw_qwen_outputs") if args.save_raw_qwen_outputs else None,
    )

    # Optional: RGB detector for scene inventory
    rgb_detector = None
    if args.scene_inventory == "rgb_detector":
        from vlm_nsm.perception.detector_perception import RGBDetectorPerception
        from PIL import Image
        detector_path = Path(args.detector_model_path)
        if not detector_path.exists():
            print(f"WARNING: Detector model not found at {detector_path}")
        else:
            rgb_detector = RGBDetectorPerception(
                model_path=detector_path,
                conf_threshold=args.detector_conf_threshold,
            )
            print(f"RGB detector loaded: {rgb_detector.model_name}")

    # Optional: camera calibration for table projection
    camera_info = None
    if args.camera_calibration:
        calib_path = Path(args.camera_calibration)
        if calib_path.exists():
            calib_data = json.loads(calib_path.read_text())
            from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
            K = np.array(calib_data["K"], dtype=np.float64)
            T = np.array(calib_data["T_camera_world"], dtype=np.float64)
            camera_info = CameraInfo(
                K=K,
                T_camera_world=T,
                width=calib_data.get("width", 512),
                height=calib_data.get("height", 512),
                camera_name=calib_data.get("camera_name", "agentview"),
            )
            print(f"Camera calibration loaded from {args.camera_calibration}")
        else:
            print(f"WARNING: Camera calibration not found at {args.camera_calibration}")

    # Optional: frontview calibration for stereo mode
    camera_info_frontview = None
    if args.rgb_spatial_mode == "stereo" and args.camera_calibration_frontview:
        calib_fv_path = Path(args.camera_calibration_frontview)
        if calib_fv_path.exists():
            calib_fv_data = json.loads(calib_fv_path.read_text())
            from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
            K_fv = np.array(calib_fv_data["K"], dtype=np.float64)
            T_fv = np.array(calib_fv_data["T_camera_world"], dtype=np.float64)
            camera_info_frontview = CameraInfo(
                K=K_fv,
                T_camera_world=T_fv,
                width=calib_fv_data.get("width", 512),
                height=calib_fv_data.get("height", 512),
                camera_name=calib_fv_data.get("camera_name", "frontview"),
            )
            print(f"Frontview calibration loaded from {args.camera_calibration_frontview}")
        else:
            print(f"WARNING: Frontview calibration not found at {args.camera_calibration_frontview}")

    # Optional validation components
    aligner = AlignmentLayer() if not args.no_alignment else None
    planner = PlannerAdapter("bfs") if not args.no_alignment else None

    results = []
    n_success = 0
    for i, item in enumerate(instructions):
        instruction = item.get("instruction", "")
        if not instruction:
            continue

        print(f"\n[{i + 1}/{len(instructions)}] {instruction[:60]}")

        # Scene inventory
        scene_info_raw = item.get("scene_info")
        scene_info = None
        scene_description = None
        if args.scene_inventory == "oracle" and scene_info_raw:
            objs = []
            for o in scene_info_raw.get("objects", []):
                pos_raw = o.get("position", [])
                if isinstance(pos_raw, list) and len(pos_raw) >= 3:
                    pose_dict = {"x": pos_raw[0], "y": pos_raw[1], "z": pos_raw[2]}
                else:
                    pose_dict = None
                objs.append(DetectedObject(
                    label=o["object_id"],
                    category=o.get("category", "block"),
                    color=o.get("color"),
                    pose=pose_dict,
                ))
            scene_info = SceneInfo(
                objects=objs,
                source="oracle",
            )
            scene_description = build_scene_description(scene_info)
        elif args.scene_inventory == "rgb_detector" and rgb_detector:
            image_path = item.get("image_path")
            if image_path:
                full_path = dataset_root / image_path if not Path(image_path).is_absolute() else Path(image_path)
                if full_path.exists():
                    img = Image.open(str(full_path)).convert("RGB")
                    width, height = img.size
                    scene_info = rgb_detector.perceive(
                        rgb_image=img, width=width, height=height,
                    )
                    scene_info.source = "rgb_detector"
                    # Enrich with image-space spatial relations from bbox centers
                    from vlm_nsm.perception.spatial_relations import compute_image_spatial_relations
                    scene_info = compute_image_spatial_relations(scene_info, width, height)

                    # Optional: spatial mode for 3D estimates
                    if args.rgb_spatial_mode == "stereo" and camera_info is not None and camera_info_frontview is not None:
                        # --- Dual-camera geometric triangulation ---
                        image_path_fv = item.get("image_path_frontview")
                        if image_path_fv:
                            full_path_fv = dataset_root / image_path_fv if not Path(image_path_fv).is_absolute() else Path(image_path_fv)
                            if full_path_fv.exists():
                                img_fv = Image.open(str(full_path_fv)).convert("RGB")
                                width_fv, height_fv = img_fv.size
                                scene_info_fv = rgb_detector.perceive(
                                    rgb_image=img_fv, width=width_fv, height=height_fv,
                                )
                                from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene
                                from vlm_nsm.perception.spatial_relations import compute_world_spatial_relations
                                from vlm_nsm.perception.rgb_table_projection import add_table_plane_pose_estimates

                                # Baseline: table projection for all objects
                                scene_info = add_table_plane_pose_estimates(
                                    scene_info, camera_info,
                                    table_z=args.table_z,
                                    object_height=args.object_height,
                                    projection_point=args.projection_point,
                                )
                                tp_poses = {
                                    i: dict(obj.pose) if obj.pose else None
                                    for i, obj in enumerate(scene_info.objects)
                                }

                                # Stereo triangulation (overwrites poses for matched objects)
                                scene_info = triangulate_scene(
                                    scene_info,
                                    scene_info_fv,
                                    camera_info,
                                    camera_info_frontview,
                                    table_z=args.table_z,
                                    object_height=args.object_height,
                                    pixel_mode=args.projection_point,
                                )

                                # Fallback: restore table-projection pose for failed stereo
                                for i, obj in enumerate(scene_info.objects):
                                    if obj.pose is None and obj.attributes.get("position_source") == "triangulation_failed":
                                        fb = tp_poses.get(i)
                                        if fb is not None:
                                            obj.pose = fb
                                            obj.attributes["position_source"] = "rgb_bbox_table_plane_projection"

                                scene_info = compute_world_spatial_relations(scene_info)
                                mode_tag = "stereo"
                            else:
                                print(f"  WARNING: Frontview image not found: {full_path_fv}, falling back to table_projection")
                                from vlm_nsm.perception.rgb_table_projection import add_table_plane_pose_estimates
                                from vlm_nsm.perception.spatial_relations import compute_world_spatial_relations
                                scene_info = add_table_plane_pose_estimates(
                                    scene_info, camera_info,
                                    table_z=args.table_z,
                                    object_height=args.object_height,
                                    projection_point=args.projection_point,
                                )
                                scene_info = compute_world_spatial_relations(scene_info)
                                mode_tag = "stereo_fallback_table_projection"
                        else:
                            print(f"  WARNING: No frontview image path in instruction, falling back to table_projection")
                            from vlm_nsm.perception.rgb_table_projection import add_table_plane_pose_estimates
                            from vlm_nsm.perception.spatial_relations import compute_world_spatial_relations
                            scene_info = add_table_plane_pose_estimates(
                                scene_info, camera_info,
                                table_z=args.table_z,
                                object_height=args.object_height,
                                projection_point=args.projection_point,
                            )
                            scene_info = compute_world_spatial_relations(scene_info)
                            mode_tag = "stereo_fallback_table_projection"
                    elif args.rgb_spatial_mode == "table_projection" and camera_info is not None:
                        from vlm_nsm.perception.rgb_table_projection import add_table_plane_pose_estimates
                        from vlm_nsm.perception.spatial_relations import compute_world_spatial_relations
                        scene_info = add_table_plane_pose_estimates(
                            scene_info,
                            camera_info,
                            table_z=args.table_z,
                            object_height=args.object_height,
                            projection_point=args.projection_point,
                        )
                        scene_info = compute_world_spatial_relations(scene_info)
                        mode_tag = "table_projection"
                    else:
                        mode_tag = "image2d"

                    if args.enable_predicate_inventory and ref_frame is not None and mode_tag != "image2d":
                        from vlm_nsm.perception.scene_predicates import compute_scene_predicates
                        scene_info = compute_scene_predicates(
                            scene_info, ref_frame,
                            object_height=args.object_height,
                        )
                    scene_description = build_scene_description(scene_info)
                    print(f"  Detector: {len(scene_info.objects)} objects detected ({mode_tag})")
                else:
                    print(f"  WARNING: Image not found: {full_path}")

        # Parse
        parse_result = qwen_parser.parse(
            instruction=instruction,
            available_objects=args.available_objects,
            scene_description=scene_description,
        )

        entry = {
            "instruction": instruction,
            "parse_success": parse_result.success,
            "error_type": parse_result.error,
            "error_message": None,
            "raw_output_path": parse_result.raw_output_path,
            "goal_json": parse_result.goal_json,
            "schema_valid": False,
            "alignment_success": None,
            "target_stack_success": None,
            "parser_target_stack": None,
            "target_stack_error": None,
            "planner_success": None,
            "planner_error": None,
            "scene_inventory_source": args.scene_inventory,
            "scene_description": scene_description,
            "scene_objects": (
                [{"object_id": o.label, "color": o.color} for o in scene_info.objects]
                if scene_info else None
            ),
            "object_grounding_valid": None,
            "grounding_errors": None,
            "expected_bindings": None,
            "predicted_bindings": None,
        }

        if not parse_result.success:
            entry["error_message"] = parse_result.error
            print(f"  Parse: ❌ ({parse_result.error})")
            results.append(entry)
            continue

        print(f"  Parse: ✅")

        # Extract goal_json for schema check
        goal_json = parse_result.goal_json
        has_schema = (isinstance(goal_json, dict)
                      and "goal_predicates" in goal_json
                      and isinstance(goal_json["goal_predicates"], list))
        entry["schema_valid"] = has_schema

        if not has_schema:
            entry["error_type"] = "schema_invalid"
            entry["error_message"] = "Missing goal_predicates or invalid type"
            print(f"  Schema: ❌")
            results.append(entry)
            continue

        print(f"  Schema: ✅")

        # Ungroundable check — parser may reject references not in scene
        if isinstance(goal_json, dict) and goal_json.get("status") == "ungroundable":
            entry["status"] = "ungroundable"
            entry["failure_type"] = "ungroundable_object"
            entry["ungroundable_detected"] = True
            entry["ungroundable_reason"] = goal_json.get("reason", "")
            entry["alignment_success"] = False
            entry["target_stack_success"] = False
            entry["planner_success"] = False
            entry["error_type"] = "ungroundable_object"
            entry["error_message"] = f"ungroundable: {goal_json.get('reason', '')}"
            print(f"  Ungroundable: ✅ ({goal_json.get('reason', '')})")

            # Determine if this is a correct rejection or false rejection
            category = item.get("category", "")
            entry["category"] = category
            if category == "adversarial_invalid_object":
                entry["adversarial_rejection_success"] = True
                entry["false_rejection"] = False
                print(f"  Adversarial rejection: ✅")
            else:
                entry["adversarial_rejection_success"] = False
                entry["false_rejection"] = True
                print(f"  False rejection: ❌ (non-adversarial instruction rejected)")

            results.append(entry)
            continue

        # Object grounding evaluation (scene inventory)
        if args.scene_inventory in ("oracle", "rgb_detector") and scene_info is not None:
            grounding_result = evaluate_object_grounding(
                instruction=instruction,
                goal_json=goal_json,
                scene_info=scene_info,
                expected_bindings=item.get("expected_bindings"),
            )
            entry["object_grounding_valid"] = grounding_result["object_grounding_valid"]
            entry["grounding_errors"] = grounding_result["grounding_errors"]
            entry["expected_bindings"] = grounding_result["expected_bindings"]
            entry["predicted_bindings"] = grounding_result["predicted_bindings"]
            if not grounding_result["object_grounding_valid"]:
                print(f"  Grounding: ❌ ({len(grounding_result['grounding_errors'])} errors)")

        # Alignment validation
        if aligner and planner:
            align_result = aligner.align_and_validate(
                goal_json=goal_json,
                valid_objects=args.available_objects + ["table"],
                valid_predicates=VALID_PREDICATES,
                type_table={k: v for k, v in TYPE_TABLE.items() if k in args.available_objects or k == "table"},
                predicate_signature=PREDICATE_SIGNATURE,
                task_type="stacking",
                n_blocks=len(args.available_objects),
            )
            entry["alignment_success"] = align_result.success

            if align_result.success:
                entry["aligned_goal_predicates"] = [
                    [pred.name, *pred.args] for pred in align_result.goal_predicates
                ]

                try:
                    parser_target_stack = parser_target_stack_from_alignment(
                        align_result,
                        args.available_objects,
                    )
                    entry["target_stack_success"] = True
                    entry["parser_target_stack"] = parser_target_stack
                except Exception as e:
                    entry["target_stack_success"] = False
                    entry["target_stack_error"] = str(e)
                    entry["planner_success"] = False
                    entry["error_message"] = f"Target stack error: {e}"
                    print(f"  Align: ✅  TargetStack: ❌")
                    results.append(entry)
                    continue

                # Planner validation (lightweight, parser-driven)
                try:
                    plan_result = planner.solve(
                        cubes=args.available_objects,
                        target_stack=parser_target_stack,
                    )
                    entry["planner_success"] = plan_result.success
                    entry["symbolic_plan"] = [
                        {"name": action.name, "args": action.args}
                        for action in plan_result.actions
                    ]
                    if plan_result.error:
                        entry["planner_error"] = plan_result.error
                except Exception as e:
                    entry["planner_success"] = False
                    entry["planner_error"] = str(e)
                    entry["error_message"] = f"Planner error: {e}"

                if entry["planner_success"]:
                    n_success += 1
                    print(f"  Align: ✅  Plan: ✅")
                else:
                    print(f"  Align: ✅  Plan: ❌")
            else:
                entry["error_type"] = align_result.error or "alignment_failed"
                entry["error_message"] = str(align_result.details) if align_result.details else None
                print(f"  Align: ❌ ({align_result.error})")
        else:
            n_success += 1

        results.append(entry)

    # Summary — aggregate metrics
    total = len(results)
    parse_ok = sum(1 for r in results if r.get("parse_success"))
    schema_valid = sum(1 for r in results if r.get("schema_valid"))
    alignment_success = sum(1 for r in results if r.get("alignment_success"))
    target_stack_success = sum(1 for r in results if r.get("target_stack_success"))
    planner_success = sum(1 for r in results if r.get("planner_success"))
    fully_successful = n_success

    # Failure taxonomy
    failure_taxonomy = {}
    for r in results:
        error_type = r.get("error_type") or (None if r.get("planner_success") else "planner_error")
        if error_type:
            failure_taxonomy[error_type] = failure_taxonomy.get(error_type, 0) + 1

    # Identify per-sample error types for failure_cases
    for r in results:
        if r.get("status") == "ungroundable":
            r["error_type"] = "ungroundable_object"
        elif not r.get("fully_successful", r.get("planner_success", False)):
            if not r.get("parse_success"):
                r["error_type"] = r.get("error_type", "parse_error")
            elif not r.get("schema_valid"):
                r["error_type"] = "schema_invalid"
            elif r.get("alignment_success") is False:
                r["error_type"] = r.get("error_type", "alignment_failed")
            elif r.get("target_stack_success") is False:
                r["error_type"] = "target_stack_error"
            elif not r.get("planner_success"):
                r["error_type"] = "planner_error"

    # Ungroundable / adversarial rejection metrics
    ungroundable_detected = sum(1 for r in results if r.get("ungroundable_detected"))
    adversarial_rejection_success = sum(1 for r in results if r.get("adversarial_rejection_success"))
    adversarial_total = sum(1 for r in results if r.get("category") == "adversarial_invalid_object")
    false_rejection = sum(1 for r in results if r.get("false_rejection"))

    metrics = {
        "total": total,
        "parse_ok": parse_ok,
        "schema_valid": schema_valid,
        "alignment_success": alignment_success,
        "target_stack_success": target_stack_success,
        "planner_success": planner_success,
        "fully_successful": fully_successful,
        "ungroundable_detected": ungroundable_detected,
        "adversarial_rejection_success": adversarial_rejection_success,
        "adversarial_rejection_rate": round(adversarial_rejection_success / adversarial_total, 3)
        if adversarial_total > 0 else 0,
        "false_rejection": false_rejection,
        "false_rejection_rate": round(false_rejection / total, 3) if total > 0 else 0,
        "failure_taxonomy": failure_taxonomy,
    }

    failure_cases = [
        {k: r[k] for k in r if k != "results"}
        for r in results
        if not r.get("fully_successful", r.get("planner_success", False))
        or r.get("status") == "ungroundable"
    ]

    print(f"\n{'=' * 50}")
    print(f"Summary: {fully_successful}/{total} fully successful")
    print(f"  Parse OK:           {parse_ok}/{total} ({100 * parse_ok / total:.1f}%)")
    print(f"  Schema Valid:       {schema_valid}/{total} ({100 * schema_valid / total:.1f}%)")
    print(f"  Alignment OK:       {alignment_success}/{total} ({100 * alignment_success / total:.1f}%)")
    print(f"  Target Stack OK:    {target_stack_success}/{total} ({100 * target_stack_success / total:.1f}%)")
    print(f"  Planner OK:         {planner_success}/{total} ({100 * planner_success / total:.1f}%)")
    if ungroundable_detected:
        print(f"  Ungroundable:       {ungroundable_detected}/{total}")
        if adversarial_total > 0:
            print(f"  Adversarial reject: {adversarial_rejection_success}/{adversarial_total} ({100 * adversarial_rejection_success / adversarial_total:.1f}% rate)")
        else:
            print(f"  Adversarial reject: N/A (no adversarial cases in this set)")
    if false_rejection:
        print(f"  False rejection:    {false_rejection} (⚠️ )")
    if failure_taxonomy:
        print(f"  Failure taxonomy:   {failure_taxonomy}")

    # Save summary
    output_data = {
        **metrics,
        "results": results,
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"Results saved to {output_path}")

    # Write metrics.json and failure_cases.json alongside the output
    output_dir = output_path.parent
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    failure_path = output_dir / "failure_cases.json"
    failure_path.write_text(json.dumps(failure_cases, indent=2, ensure_ascii=False))
    print(f"Failure cases saved to {failure_path} ({len(failure_cases)} cases)")


if __name__ == "__main__":
    main()
