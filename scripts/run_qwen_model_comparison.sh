#!/usr/bin/env bash
# Run full model comparison: Qwen2.5-14B vs Qwen2.5-32B on hard datasets
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

: "${QWEN32_PATH:?Set QWEN32_PATH to the Qwen2.5-32B-Instruct directory}"
: "${QWEN14_PATH:?Set QWEN14_PATH to the Qwen2.5-14B-Instruct directory}"

echo "========================================"
echo " Model Comparison: Qwen2.5-14B vs 32B"
echo "========================================"

# ---- 32B Full ----
echo "===== Hanoi Hard 32B (full prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN32_PATH" --model_name qwen25_32b \
  --difficulty hard --prompt_variant full --alignment_mode full

echo "===== Stacking Hard 32B (full prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task stacking --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN32_PATH" --model_name qwen25_32b \
  --difficulty hard --prompt_variant full --alignment_mode full

# ---- 14B Full ----
echo "===== Hanoi Hard 14B (full prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN14_PATH" --model_name qwen25_14b \
  --difficulty hard --prompt_variant full --alignment_mode full

echo "===== Stacking Hard 14B (full prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task stacking --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN14_PATH" --model_name qwen25_14b \
  --difficulty hard --prompt_variant full --alignment_mode full

# ---- 32B Weak Prompt ----
echo "===== Hanoi Hard 32B (weak prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN32_PATH" --model_name qwen25_32b \
  --difficulty hard --prompt_variant weak --alignment_mode full

echo "===== Stacking Hard 32B (weak prompt, full alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task stacking --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN32_PATH" --model_name qwen25_32b \
  --difficulty hard --prompt_variant weak --alignment_mode full

# ---- 32B Weak Prompt + No Alignment ----
echo "===== Hanoi Hard 32B (weak prompt, no alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN32_PATH" --model_name qwen25_32b \
  --difficulty hard --prompt_variant weak --alignment_mode no_alignment

echo "===== Stacking Hard 32B (weak prompt, no alignment) ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task stacking --n_blocks 3 --parser qwen_text \
  --model_path "$QWEN32_PATH" --model_name qwen25_32b \
  --difficulty hard --prompt_variant weak --alignment_mode no_alignment

# ---- Summary & Analysis ----
echo ""
echo "===== Summarize Results ====="
python -m vlm_nsm.experiments.summarize_language_variation

echo ""
echo "===== Analyze Failures ====="
python -m vlm_nsm.experiments.analyze_qwen_failures

echo ""
echo "===== Alignment Repair Stats ====="
python -m vlm_nsm.experiments.analyze_alignment_repairs

echo ""
echo "===== Export Report Tables ====="
python -m vlm_nsm.experiments.export_report_tables

echo ""
echo "===== Full model comparison completed ====="
