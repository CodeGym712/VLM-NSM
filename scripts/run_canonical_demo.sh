#!/usr/bin/env bash
# Stage 11 Release Candidate: Canonical E2E Language → Simulation Demo
#
# Runs the end-to-end NL → goal → plan → Robosuite stacking pipeline
# using the canonical reach_pick-only deployment policy.
#
# Usage:
#   bash scripts/run_canonical_demo.sh [--no_bc] [--bc_phases reach_pick]
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
BC_PHASES="reach_pick"
OUTPUT_DIR="generated/results/release_candidate/e2e_demo"
USE_BC="--use_bc"
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_bc) USE_BC="--no_bc"; shift ;;
        --bc_phases) BC_PHASES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo "Canonical E2E Language → Simulation Demo"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Config:     ${CONFIG}"
echo "BC phases:  ${BC_PHASES}"
echo "Output:     ${OUTPUT_DIR}"
echo "Device:     ${DEVICE}"
echo "============================================"

python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    ${USE_BC} \
    --bc_phases ${BC_PHASES} \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Done. Results saved to ${OUTPUT_DIR}/"
