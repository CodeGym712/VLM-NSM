#!/usr/bin/env bash
# Stage 18-E: qwen-plus three-way comparison with table-plane projection.
#
# Runs all three scene conditions with the SAME LLM (qwen-plus).
# The rgb_detector condition now uses camera-calibrated table-plane
# projection for estimated 3D positions and world spatial relations.
#
# Conditions:
#   1. without_scene          — no scene inventory (baseline)
#   2. oracle                 — sim-state oracle
#   3. rgb_detector_table_projection — YOLOv8n + table-plane 3D projection
#
# Outputs:
#   generated/results/stage_18/qwen_plus_without_scene/summary.json
#   generated/results/stage_18/qwen_plus_oracle/summary.json
#   generated/results/stage_18/qwen_plus_rgb_detector_table_projection/summary.json
#   generated/results/stage_18/scene_source_comparison_qwen_plus_table_projection.json
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
CALIB="generated/configs/camera_calibration_agentview_512.json"

mkdir -p "$OUTPUT"

# ------------------------------------------------------------------
# 1. WITHOUT_SCENE (qwen-plus)
# ------------------------------------------------------------------
echo "=== [1/3] without_scene (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json \
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
echo "=== [2/3] oracle (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json \
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
# 3. RGB DETECTOR TABLE PROJECTION (qwen-plus)
# ------------------------------------------------------------------
MEDIUM_MODEL="generated/models/rgb_perception_yolov8n_medium/best.pt"
echo "=== [3/3] rgb_detector_table_projection (qwen-plus) ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json \
  --output "${OUTPUT}/qwen_plus_rgb_detector_table_projection/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory rgb_detector \
  --rgb_spatial_mode table_projection \
  --camera_calibration "$CALIB" \
  --table_z 0.0 \
  --object_height 0.05 \
  --detector_model_path "$MEDIUM_MODEL" \
  --detector_conf_threshold 0.15 \
  --projection_point bottom_center

# ------------------------------------------------------------------
# 4. Three-way comparison
# ------------------------------------------------------------------
echo "=== Comparison ==="
"$PYTHON_BIN" -m vlm_nsm.experiments.compare_scene_grounding_results \
  --without_scene "${OUTPUT}/qwen_plus_without_scene/summary.json" \
  --with_oracle "${OUTPUT}/qwen_plus_oracle/summary.json" \
  --with_detector "${OUTPUT}/qwen_plus_rgb_detector_table_projection/summary.json" \
  --input generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json \
  --output "${OUTPUT}/scene_source_comparison_qwen_plus_table_projection.json"

echo "=== Done ==="
echo "Results:"
echo "  without_scene:             ${OUTPUT}/qwen_plus_without_scene/summary.json"
echo "  oracle:                    ${OUTPUT}/qwen_plus_oracle/summary.json"
echo "  rgb_detector_table_projection: ${OUTPUT}/qwen_plus_rgb_detector_table_projection/summary.json"
echo "  comparison:                ${OUTPUT}/scene_source_comparison_qwen_plus_table_projection.json"
