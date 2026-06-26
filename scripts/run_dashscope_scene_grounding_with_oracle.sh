#!/usr/bin/env bash
# Stage 17-F: Scene grounding parse eval WITH oracle scene inventory.
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

OUTPUT_DIR="generated/results/stage_17/scene_grounding_with_oracle"

"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions generated/eval_inputs/stage_17/scene_grounding_parse50.json \
  --output "${OUTPUT_DIR}/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory oracle
