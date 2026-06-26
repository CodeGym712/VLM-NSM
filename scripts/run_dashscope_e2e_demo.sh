#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

if [[ -f configs/dashscope_qwen.env ]]; then
  # Private, untracked credentials file. See configs/dashscope_qwen.env.example.
  # shellcheck disable=SC1091
  set +x
  source configs/dashscope_qwen.env
fi

: "${QWEN_API_KEY:?Set QWEN_API_KEY to your DashScope API key first.}"
QWEN_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen-plus}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
  --output_dir generated/results/stage_15/e2e_dashscope_qwen_parser_driven
