#!/usr/bin/env bash
# Stage 20 teacher demo collection scaffold.
# This script is not intended to be run automatically by tests.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-nsm/bin/python}"
SUITE_PATH="${SUITE_PATH:-generated/eval_inputs/stage_20/stage20_teacher_demo_collection_200.json}"

"$PYTHON_BIN" -m vlm_nsm.experiments.stage20_generate_teacher_demo_suite \
  --output "$SUITE_PATH" \
  --seeds_per_bin 20 \
  --placement_bins center,front,back,left,right \
  --goals red_on_green,green_on_red \
  --scripted_profile cube2_robust

"$PYTHON_BIN" -m vlm_nsm.experiments.stage19_run_symbolic_rollouts \
  --rollout_suite "$SUITE_PATH" \
  --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --scripted_only \
  --scripted_profile cube2_robust \
  --save_diagnostics \
  --render none \
  --output_summary generated/results/stage_20/stage20_teacher_demo_collection_summary.json \
  --output_episodes generated/results/stage_20/stage20_teacher_demo_collection_episodes.jsonl \
  --diagnostics_output generated/results/stage_20/stage20_teacher_demo_collection_diagnostics.jsonl
