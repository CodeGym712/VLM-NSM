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

"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions generated/results/stage_12/cached_qwen_e2e_inputs.json \
  --output generated/results/stage_15/dashscope_qwen_parse_5/summary.json \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --save_raw_qwen_outputs
