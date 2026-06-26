#!/usr/bin/env bash
# Stage 18-A: Scene grounding parse eval WITH RGB detector scene inventory.
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

# Model path: use medium model if available, fall back to smoke model
MEDIUM_MODEL="generated/models/rgb_perception_yolov8n_medium/best.pt"
SMOKE_MODEL="generated/models/rgb_perception_yolov8n_smoke/weights/best.pt"

if [[ -f "$MEDIUM_MODEL" ]]; then
  DETECTOR_MODEL="$MEDIUM_MODEL"
  echo "Using medium detector model: $MEDIUM_MODEL"
else
  DETECTOR_MODEL="$SMOKE_MODEL"
  echo "Medium model not found, using smoke model: $SMOKE_MODEL"
fi

OUTPUT_DIR="generated/results/stage_18/scene_grounding_rgb_detector"

mkdir -p generated/results/stage_18

"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json \
  --output "${OUTPUT_DIR}/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory rgb_detector \
  --detector_model_path "$DETECTOR_MODEL" \
  --detector_conf_threshold 0.15
