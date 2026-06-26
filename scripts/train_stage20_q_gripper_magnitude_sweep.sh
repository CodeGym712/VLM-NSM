#!/usr/bin/env bash
set -euo pipefail

# Stage 20-Q gripper magnitude auxiliary weight sweep.
# Not intended to be run automatically by tests.

nsm/bin/python -c "import argparse, json; from vlm_nsm.training.train_stage20_goal_conditioned_rnn import train; args=argparse.Namespace(dataset_dir='generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled', output_dir='generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_gripper_magnitude_weight025_stage20_q', model_type='lstm', feature_mode='goal_conditioned', epochs=50, batch_size=256, lr=1e-3, hidden_dim=256, num_layers=1, dropout=0.0, seed=0, device='cuda', sampler='none', loss_weighting='none', manual_phase_weights=None, weighted_metric_report=False, gripper_aux_weight=0.0, gripper_aux_phase_ids='1', gripper_magnitude_aux_weight=0.25, gripper_magnitude_aux_phase_ids='1', overwrite=False); print(json.dumps(train(args), indent=2, ensure_ascii=False))"

nsm/bin/python -c "import argparse, json; from vlm_nsm.training.train_stage20_goal_conditioned_rnn import train; args=argparse.Namespace(dataset_dir='generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled', output_dir='generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_gripper_magnitude_weight075_stage20_q', model_type='lstm', feature_mode='goal_conditioned', epochs=50, batch_size=256, lr=1e-3, hidden_dim=256, num_layers=1, dropout=0.0, seed=0, device='cuda', sampler='none', loss_weighting='none', manual_phase_weights=None, weighted_metric_report=False, gripper_aux_weight=0.0, gripper_aux_phase_ids='1', gripper_magnitude_aux_weight=0.75, gripper_magnitude_aux_phase_ids='1', overwrite=False); print(json.dumps(train(args), indent=2, ensure_ascii=False))"
