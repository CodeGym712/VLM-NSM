#!/usr/bin/env bash
# Stage 20-E offline MLP training scaffold.
# This script is not intended to be run automatically by tests.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-nsm/bin/python}"
DATASET_DIR="${DATASET_DIR:-generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_1}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON_BIN" -m vlm_nsm.training.train_stage20_goal_conditioned_mlp \
  --dataset_dir "$DATASET_DIR" \
  --output_dir generated/results/stage_20/obs_only_mlp_v0_1 \
  --feature_mode obs_only \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --num_layers 2 \
  --seed 0 \
  --device "$DEVICE"

"$PYTHON_BIN" -m vlm_nsm.training.train_stage20_goal_conditioned_mlp \
  --dataset_dir "$DATASET_DIR" \
  --output_dir generated/results/stage_20/goal_conditioned_mlp_v0_1 \
  --feature_mode goal_conditioned \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --num_layers 2 \
  --seed 0 \
  --device "$DEVICE"

"$PYTHON_BIN" -m vlm_nsm.experiments.stage20_e_report \
  --obs_only_metrics generated/results/stage_20/obs_only_mlp_v0_1/metrics.json \
  --goal_metrics generated/results/stage_20/goal_conditioned_mlp_v0_1/metrics.json \
  --dataset_summary "$DATASET_DIR/summary.json" \
  --output generated/results/stage_20/stage20_e_goal_mlp_report.md
