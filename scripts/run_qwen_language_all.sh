#!/usr/bin/env bash
# Run all language variation evaluations with Qwen model
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen model directory}"

echo "===== Hanoi Language Variation (Qwen) ====="
python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi \
  --n_blocks 3 \
  --parser qwen_text \
  --model_path "$MODEL_PATH" \
  --instruction_file data/language_variations/hanoi_zh_en.jsonl

echo ""
echo "===== Stacking Language Variation (Qwen) ====="
python -m vlm_nsm.experiments.run_language_variation \
  --task stacking \
  --n_blocks 3 \
  --parser qwen_text \
  --model_path "$MODEL_PATH" \
  --instruction_file data/language_variations/stacking_zh_en.jsonl

echo ""
echo "===== Summarize Results ====="
python -m vlm_nsm.experiments.summarize_language_variation

echo ""
echo "===== All Qwen language evaluations completed ====="
