#!/usr/bin/env bash
# Stage 19 smoke: parser/planner -> SkillPlan -> ScriptedSkillExecutor.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --parser mock \
  --bc_phases reach_pick \
  --control_interface skill_executor \
  --output_dir generated/results/stage_19/e2e_skill_executor5
