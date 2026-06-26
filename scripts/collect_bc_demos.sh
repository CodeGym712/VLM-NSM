#!/usr/bin/env bash
# Stage 7B-1: Collect full-trajectory demos for BC training
# Run on a GPU machine with MuJoCo/EGL support.
#
# Usage:
#   bash scripts/collect_bc_demos.sh
#
# Requires: CUDA_VISIBLE_DEVICES=0-3, MUJOCO_GL=egl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUTPUT_DIR="generated/demos/stage_7B-1"
CONFIG_PATH="generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11.json"
NUM_EPISODES=100

echo "=== Stage 7B-1: BC Demo Collection ==="
echo "Config: $CONFIG_PATH"
echo "Episodes: $NUM_EPISODES"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Collect demos with full trajectory recording
echo "--- Step 1: Collecting demos with trajectory recording ---"
CUDA_VISIBLE_DEVICES=0,1,2,3 MUJOCO_GL=egl python3 \
    -m vlm_nsm.experiments.collect_robosuite_demos \
    --config "$CONFIG_PATH" \
    --num_episodes "$NUM_EPISODES" \
    --output_dir "$OUTPUT_DIR/robosuite_stage7b1_raw" \
    --record

echo ""
echo "=== Demo collection complete ==="
echo "Raw demos (w/ init/final frames): $OUTPUT_DIR/robosuite_stage7b1_raw/"
echo "BC training data: generated/demos/stage_7B-1/robosuite_stage7b1_trajectories.npz"
echo ""

# Step 2: Train BC model
echo "--- Step 2: Training BC policy ---"
mkdir -p generated/models/stage_7B-1

CUDA_VISIBLE_DEVICES=0 python3 \
    -m vlm_nsm.learning.train_bc \
    --npz generated/demos/stage_7B-1/robosuite_stage7b1_trajectories.npz \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda \
    --save generated/models/stage_7B-1/bc_mlp.pt

echo ""
echo "=== BC training complete ==="
echo "Model saved to: generated/models/stage_7B-1/bc_mlp.pt"
