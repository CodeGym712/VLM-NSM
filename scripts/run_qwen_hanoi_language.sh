#!/usr/bin/env bash
# Run language variation evaluation with Qwen model on Hanoi
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen model directory}"

echo "===== Hanoi Language Variation (Qwen) ====="
python3 -m vlm_nsm.experiments.run_language_variation \
  --task hanoi \
  --n_blocks 3 \
  --parser qwen_text \
  --model_path "$MODEL_PATH" \
  --instruction_file data/language_variations/hanoi_zh_en.jsonl

echo ""
echo "===== Done ====="
