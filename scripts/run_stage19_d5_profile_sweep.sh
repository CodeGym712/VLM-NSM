#!/usr/bin/env bash
# Stage 19-D5: green_on_red profile sweep, scripted-only.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-nsm/bin/python}"
COMMON_ARGS=(
  -m vlm_nsm.experiments.stage19_run_symbolic_rollouts
  --rollout_suite generated/eval_inputs/stage_19/stage19_d5_green_on_red_20.json
  --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json
  --scripted_only
  --save_diagnostics
  --render none
)

run_profile() {
  local profile="$1"
  local tag="$2"
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --scripted_profile "$profile" \
    --output_summary "generated/results/stage_19/stage19_d5_profile_${tag}_summary.json" \
    --output_episodes "generated/results/stage_19/stage19_d5_profile_${tag}_episodes.jsonl" \
    --diagnostics_output "generated/results/stage_19/stage19_d5_profile_${tag}_diagnostics.jsonl"
}

run_profile default default
run_profile cube2_grasp_lower grasp_lower
run_profile cube2_place_higher place_higher
run_profile cube2_robust robust

"$PYTHON_BIN" -m vlm_nsm.experiments.stage19_d5_profile_report
