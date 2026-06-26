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

"$PYTHON_BIN" -m vlm_nsm.experiments.smoke_dashscope_qwen \
  --api_base "$QWEN_API_BASE" \
  --model "$QWEN_MODEL" \
  --api_key_env QWEN_API_KEY
