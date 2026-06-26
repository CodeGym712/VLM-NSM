#!/usr/bin/env bash
set -euo pipefail

# Stage 20-I recurrent BC training entrypoint.
# Not intended to be run automatically by tests.

nsm/bin/python -m vlm_nsm.training.train_stage20_goal_conditioned_rnn \
  --dataset_dir generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_1_plus_v0_2 \
  --output_dir generated/results/stage_20/goal_conditioned_gru_combined_v0_1 \
  --model_type gru \
  --feature_mode goal_conditioned \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --num_layers 1 \
  --dropout 0.0 \
  --seed 0 \
  --device cuda
