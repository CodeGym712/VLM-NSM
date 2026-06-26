#!/bin/bash
# Stage 19-D1: Horizon ablation
# Tests whether timeout failures become successes with longer horizon.
# Only runs episodes that failed with timeout in R1.
#
# Usage:
#   bash scripts/run_stage19_d1_horizon_ablation.sh

set -euo pipefail

EXEC_CONFIG="generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json"
CHECKPOINT="generated/results/stage_7B-4/dagger_reach_bc/best.pt"
ROOT="generated/results/stage_19"
OUTDIR="${ROOT}/horizon_ablation"
mkdir -p "${OUTDIR}"

HORIZONS=(1000 1500 2000)

echo "=== Stage 19-D1 Horizon Ablation ==="
echo "Original horizon: 1000 (baseline, already measured in R1)"
echo "Test horizons: ${HORIZONS[*]}"
echo ""

for H in "${HORIZONS[@]}"; do
    echo "--- Horizon = ${H} ---"
    MUJOCO_GL=egl python3 -m vlm_nsm.experiments.stage19_run_symbolic_rollouts \
        --rollout_suite generated/eval_inputs/stage_19/symbolic_goal_rollout_smoke_12.json \
        --execution_config "${EXEC_CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output_summary "${OUTDIR}/horizon_${H}_summary.json" \
        --output_episodes "${OUTDIR}/horizon_${H}_episodes.jsonl" \
        --max_steps "${H}" \
        --render none \
        --abort_on_unsafe_goal \
        2>&1 | tail -20
    echo ""
done

echo "=== Ablation complete ==="
echo "Results in: ${OUTDIR}/"

# Quick comparison
echo ""
echo "Success rate comparison:"
for H in "${HORIZONS[@]}"; do
    if [ -f "${OUTDIR}/horizon_${H}_summary.json" ]; then
        RATE=$(python3 -c "import json; d=json.load(open('${OUTDIR}/horizon_${H}_summary.json')); print(f\"{d['task_success_rate']:.1%}\")")
        echo "  horizon=${H}: ${RATE}"
    fi
done
