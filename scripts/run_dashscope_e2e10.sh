#!/usr/bin/env bash
# Stage 15.3-A: Run DashScope E2E 10-episode demo with standardized summary.
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

# Override with nsm venv if available
if command -v nsm/bin/python &> /dev/null; then
  PYTHON_BIN="nsm/bin/python"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON_BIN" -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --parser online_qwen \
  --qwen_model "$QWEN_MODEL" \
  --qwen_api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --qwen_temperature 0.0 \
  --qwen_max_tokens 512 \
  --qwen_retries 1 \
  --save_raw_qwen_outputs \
  --bc_phases reach_pick \
  --instructions generated/eval_inputs/stage_15/dashscope_e2e10_instructions.json \
  --output_dir generated/results/stage_15/e2e_dashscope_qwen_parser_driven_10
