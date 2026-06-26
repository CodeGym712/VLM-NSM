#!/usr/bin/env bash
# Run Qwen2.5-32B hard language variation evaluations
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

: "${QWEN32_PATH:?Set QWEN32_PATH to the Qwen2.5-32B-Instruct directory}"

echo "===== Hanoi Hard (Qwen2.5-32B, full prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi \
  --n_blocks 3 \
  --parser qwen_text \
  --model_path "$QWEN32_PATH" \
  --model_name qwen25_32b \
  --difficulty hard \
  --prompt_variant full \
  --alignment_mode full

echo ""
echo "===== Stacking Hard (Qwen2.5-32B, full prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task stacking \
  --n_blocks 3 \
  --parser qwen_text \
  --model_path "$QWEN32_PATH" \
  --model_name qwen25_32b \
  --difficulty hard \
  --prompt_variant full \
  --alignment_mode full

echo ""
echo "===== All Qwen2.5-32B hard evaluations completed ====="
