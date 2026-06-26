#!/usr/bin/env bash
# Camera Calibration Audit — Stage 18-FixCam
#
# Runs the full calibration inspection, consistency check, and convention
# search pipeline for agentview, frontview, and birdview cameras.
#
# Prerequisites:
#   - Robosuite Stack environment available
#   - YOLO detector model at generated/models/rgb_perception_yolov8n_medium/best.pt
#   - Eval input at generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json
#
# Usage:
#   bash scripts/run_camera_calibration_audit.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESULT_DIR="generated/results/stage_18"
EVAL_INPUT="generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json"
DETECTOR_MODEL="generated/models/rgb_perception_yolov8n_medium/best.pt"

mkdir -p "$RESULT_DIR"

echo "============================================"
echo "Stage 18-FixCam: Camera Calibration Audit"
echo "============================================"
echo "Python: $PYTHON_BIN"
echo "Results: $RESULT_DIR"
echo ""

# ── Step 1: World Geometry Inspection ────────────────────────────────────────

echo "── Step 1/4: World Geometry Inspection ──"
"$PYTHON_BIN" src/vlm_nsm/experiments/inspect_robosuite_world_geometry.py \
    --output "$RESULT_DIR/robosuite_world_geometry_inspection.json"
echo ""

# ── Step 2: Projection Consistency Check (Agentview) ─────────────────────────

echo "── Step 2/4: Projection Consistency (agentview) ──"
"$PYTHON_BIN" src/vlm_nsm/experiments/debug_camera_projection_consistency.py \
    --eval_input "$EVAL_INPUT" \
    --calibration_agentview generated/configs/camera_calibration_agentview_512.json \
    --detector_model_path "$DETECTOR_MODEL" \
    --table_z 0.8 \
    --output_metrics "$RESULT_DIR/projection_consistency_agentview.json" \
    --output_debug_jsonl "$RESULT_DIR/projection_consistency_agentview.jsonl"
echo ""

# ── Step 3: Convention Search ────────────────────────────────────────────────

for CAMERA in agentview frontview birdview; do
    echo "── Step 3/4: Convention Search ($CAMERA) ──"
    "$PYTHON_BIN" src/vlm_nsm/experiments/search_camera_convention.py \
        --eval_input "$EVAL_INPUT" \
        --detector_model_path "$DETECTOR_MODEL" \
        --camera_name "$CAMERA" \
        --output_dir "$RESULT_DIR"
    echo ""
done

# ── Step 4: Frontview & Birdview Projection Consistency ──────────────────────

for CAMERA in frontview birdview; do
    CALIB_FILE="generated/configs/camera_calibration_${CAMERA}_512.json"
    if [ -f "$CALIB_FILE" ]; then
        echo "── Step 4/4: Projection Consistency ($CAMERA) ──"
        "$PYTHON_BIN" src/vlm_nsm/experiments/debug_camera_projection_consistency.py \
            --eval_input "$EVAL_INPUT" \
            --calibration_agentview generated/configs/camera_calibration_agentview_512.json \
            --calibration_frontview "$CALIB_FILE" \
            --secondary_camera_name "$CAMERA" \
            --detector_model_path "$DETECTOR_MODEL" \
            --table_z 0.8 \
            --output_metrics "$RESULT_DIR/projection_consistency_${CAMERA}.json" \
            --output_debug_jsonl "$RESULT_DIR/projection_consistency_${CAMERA}.jsonl"
        echo ""
    else
        echo "── Step 4/4: Skipping $CAMERA (no calibration file) ──"
    fi
done

echo "============================================"
echo "Audit complete. Results in $RESULT_DIR/"
echo "============================================"
ls -la "$RESULT_DIR"/
