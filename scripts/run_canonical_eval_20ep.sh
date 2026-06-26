#!/usr/bin/env bash
# Stage 11 Release Candidate: Canonical BC Evaluation (20ep)
#
# Evaluates the canonical reach_pick-only deployment policy on 20 Robosuite
# stacking episodes using the shared episode runner (eval_bc_hybrid).
#
# Usage:
#   bash scripts/run_canonical_eval_20ep.sh [--seed N]
#
# Prerequisites:
#   - conda environment with robosuite 1.5.2, torch, etc.
#   - CUDA-capable GPU
#   - MUJOCO_GL=egl (headless) or osmesa
#
# Environment:
#   CUDA_VISIBLE_DEVICES (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

# Defaults
CHECKPOINT="generated/results/stage_7B-4/dagger_reach_bc/best.pt"
CONFIG="generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json"
NUM_EPISODES=20
BC_PHASES="reach_pick"
FIXED_ZERO_DIMS="3 4 5"
OUTPUT="generated/results/release_candidate/reach_pick_20ep.json"
SEED=0
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --num_episodes) NUM_EPISODES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --bc_phases) BC_PHASES="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo "Canonical BC Evaluation (${NUM_EPISODES}ep)"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Config:     ${CONFIG}"
echo "BC phases:  ${BC_PHASES}"
echo "Episodes:   ${NUM_EPISODES}"
echo "Seed:       ${SEED}"
echo "Output:     ${OUTPUT}"
echo "============================================"

python3 -m vlm_nsm.learning.eval_bc_hybrid \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --num_episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --condition_on_phase \
    --bc_phases ${BC_PHASES} \
    --fixed_zero_action_dims ${FIXED_ZERO_DIMS} \
    --device "${DEVICE}" \
    --output "${OUTPUT}"

echo ""
echo "Done. Results saved to ${OUTPUT}"
