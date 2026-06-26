#!/bin/bash
# Run all remaining Qwen evaluations sequentially
# Each invocation loads the model once, processes, then exits.
# CUDA memory is freed between invocations.

set -euo pipefail

PROJECT_ROOT="${VLM_NSM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_CPP_MIN_LOG_LEVEL=3
PYTHON="${PYTHON:-python3}"
: "${QWEN32_PATH:?Set QWEN32_PATH to the Qwen2.5-32B-Instruct directory}"
: "${QWEN14_PATH:?Set QWEN14_PATH to the Qwen2.5-14B-Instruct directory}"
QWEN32="$QWEN32_PATH"
QWEN14="$QWEN14_PATH"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/generated/logs}"
mkdir -p "$LOG_DIR"

run() {
    local tag=$1
    shift
    echo ""
    echo "======================================================"
    echo "[$(date)] Starting: $tag"
    echo "======================================================"
    $PYTHON -m vlm_nsm.experiments.run_language_variation "$@" 2>&1 | tee -a "$LOG_DIR/${tag}.log"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "[$(date)] FAILED ($rc): $tag" >&2
    else
        echo "[$(date)] OK: $tag"
    fi
    return $rc
}

# ===== Qwen2.5-32B =====

# Easy sets (new format)
run "hanoi_easy_32b_full" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty easy --prompt_variant full --alignment_mode full

run "stacking_easy_32b_full" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty easy --prompt_variant full --alignment_mode full

# Weak prompt (hard set)
run "hanoi_hard_32b_weak" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant weak --alignment_mode full

run "stacking_hard_32b_weak" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant weak --alignment_mode full

# Alignment: no_alignment (hard set)
run "hanoi_hard_32b_no_alignment" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_alignment

run "stacking_hard_32b_no_alignment" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_alignment

# Alignment: no_repair (hard set)
run "hanoi_hard_32b_no_repair" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_repair

run "stacking_hard_32b_no_repair" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_repair

# Alignment: no_alias_repair (hard set)
run "hanoi_hard_32b_no_alias_repair" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_alias_repair

run "stacking_hard_32b_no_alias_repair" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_alias_repair

# Alignment: no_completeness_repair (hard set)
run "hanoi_hard_32b_no_completeness" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_completeness_repair

run "stacking_hard_32b_no_completeness" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN32" --model_name qwen25_32b \
    --difficulty hard --prompt_variant full --alignment_mode no_completeness_repair

# ===== Qwen2.5-14B =====
run "hanoi_hard_14b_full" \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN14" --model_name qwen25_14b \
    --difficulty hard --prompt_variant full --alignment_mode full

run "stacking_hard_14b_full" \
    --task stacking --n_blocks 3 --parser qwen_text \
    --model_path "$QWEN14" --model_name qwen25_14b \
    --difficulty hard --prompt_variant full --alignment_mode full

echo ""
echo "======================================================"
echo "[$(date)] All evaluations complete!"
echo "======================================================"
