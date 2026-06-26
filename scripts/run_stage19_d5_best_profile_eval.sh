#!/usr/bin/env bash
# Stage 19-D5: 40-episode scripted-only eval for selected best profile.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-nsm/bin/python}"
SCRIPTED_PROFILE="${SCRIPTED_PROFILE:-cube2_robust}"

"$PYTHON_BIN" -m vlm_nsm.experiments.stage19_run_symbolic_rollouts \
  --rollout_suite generated/eval_inputs/stage_19/stage19_d5_scripted_best_profile_eval_40.json \
  --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --scripted_only \
  --scripted_profile "$SCRIPTED_PROFILE" \
  --save_diagnostics \
  --render none \
  --output_summary generated/results/stage_19/stage19_d5_best_profile_eval_summary.json \
  --output_episodes generated/results/stage_19/stage19_d5_best_profile_eval_episodes.jsonl \
  --diagnostics_output generated/results/stage_19/stage19_d5_best_profile_eval_diagnostics.jsonl

"$PYTHON_BIN" -m vlm_nsm.experiments.stage19_d5_profile_report \
  --best_profile "$SCRIPTED_PROFILE"
