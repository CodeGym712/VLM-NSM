#!/usr/bin/env bash
# Stage 18-F3: qwen-plus four-way comparison with fixed stereo triangulation.
#
# Runs all four scene conditions with the SAME LLM (qwen-plus).
# The rgb_detector_stereo condition uses dual-camera geometric ray
# triangulation for calibrated 3D positions and world spatial relations.
#
# Conditions:
#   1. without_scene          — no scene inventory (baseline)
#   2. oracle                 — sim-state oracle
#   3. rgb_detector_image2d   — YOLOv8n + image-space spatial relations
#   4. rgb_detector_stereo    — YOLOv8n + dual-camera stereo triangulation
#
# Outputs:
#   generated/results/stage_18/qwen_plus_without_scene/summary.json
#   generated/results/stage_18/qwen_plus_oracle/summary.json
#   generated/results/stage_18/qwen_plus_rgb_detector_image2d/summary.json
#   generated/results/stage_18/qwen_plus_rgb_detector_stereo_fixed/summary.json
#   generated/results/stage_18/scene_source_comparison_qwen_plus_stereo_fixed.json
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f configs/dashscope_qwen.env ]]; then
  # shellcheck disable=SC1091
  set +x
  source configs/dashscope_qwen.env
fi

: "${QWEN_API_KEY:?Set QWEN_API_KEY to your DashScope API key first.}"
QWEN_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen-plus}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT="generated/results/stage_18"
CALIB_AGENT="generated/configs/camera_calibration_agentview_512.json"
CALIB_FRONT="generated/configs/camera_calibration_frontview_512.json"
MEDIUM_MODEL="generated/models/rgb_perception_yolov8n_medium/best.pt"
EVAL_INPUT="generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json"

mkdir -p "$OUTPUT"

# ------------------------------------------------------------------
# 1. WITHOUT_SCENE (qwen-plus)
# ------------------------------------------------------------------
echo "=== [1/4] without_scene (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions "$EVAL_INPUT" \
  --output "${OUTPUT}/qwen_plus_without_scene/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory none

# ------------------------------------------------------------------
# 2. ORACLE (qwen-plus)
# ------------------------------------------------------------------
echo "=== [2/4] oracle (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions "$EVAL_INPUT" \
  --output "${OUTPUT}/qwen_plus_oracle/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory oracle

# ------------------------------------------------------------------
# 3. RGB DETECTOR IMAGE2D (qwen-plus)
# ------------------------------------------------------------------
echo "=== [3/4] rgb_detector_image2d (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions "$EVAL_INPUT" \
  --output "${OUTPUT}/qwen_plus_rgb_detector_image2d/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory rgb_detector \
  --rgb_spatial_mode image2d \
  --detector_model_path "$MEDIUM_MODEL" \
  --detector_conf_threshold 0.15

# ------------------------------------------------------------------
# 4. RGB DETECTOR STEREO FIXED (qwen-plus)
# ------------------------------------------------------------------
echo "=== [4/4] rgb_detector_stereo_fixed (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions "$EVAL_INPUT" \
  --output "${OUTPUT}/qwen_plus_rgb_detector_stereo_fixed/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory rgb_detector \
  --rgb_spatial_mode stereo \
  --camera_calibration "$CALIB_AGENT" \
  --camera_calibration_frontview "$CALIB_FRONT" \
  --table_z 0.0 \
  --object_height 0.05 \
  --detector_model_path "$MEDIUM_MODEL" \
  --detector_conf_threshold 0.15 \
  --projection_point center

# ------------------------------------------------------------------
# 5. Comparisons (image2d and stereo_fixed)
# ------------------------------------------------------------------
echo "=== Comparison (image2d) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.compare_scene_grounding_results \
  --without_scene "${OUTPUT}/qwen_plus_without_scene/summary.json" \
  --with_oracle "${OUTPUT}/qwen_plus_oracle/summary.json" \
  --with_detector "${OUTPUT}/qwen_plus_rgb_detector_image2d/summary.json" \
  --input "$EVAL_INPUT" \
  --output "${OUTPUT}/scene_source_comparison_qwen_plus_image2d.json"

echo "=== Comparison (stereo_fixed) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.compare_scene_grounding_results \
  --without_scene "${OUTPUT}/qwen_plus_without_scene/summary.json" \
  --with_oracle "${OUTPUT}/qwen_plus_oracle/summary.json" \
  --with_detector "${OUTPUT}/qwen_plus_rgb_detector_stereo_fixed/summary.json" \
  --input "$EVAL_INPUT" \
  --output "${OUTPUT}/scene_source_comparison_qwen_plus_stereo_fixed.json"

echo "=== Done ==="
echo "Results:"
echo "  without_scene:                 ${OUTPUT}/qwen_plus_without_scene/summary.json"
echo "  oracle:                        ${OUTPUT}/qwen_plus_oracle/summary.json"
echo "  rgb_detector_image2d:          ${OUTPUT}/qwen_plus_rgb_detector_image2d/summary.json"
echo "  rgb_detector_stereo_fixed:     ${OUTPUT}/qwen_plus_rgb_detector_stereo_fixed/summary.json"
echo "  comparison (image2d):          ${OUTPUT}/scene_source_comparison_qwen_plus_image2d.json"
echo "  comparison (stereo):           ${OUTPUT}/scene_source_comparison_qwen_plus_stereo_fixed.json"
