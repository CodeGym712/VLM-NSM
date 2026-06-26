#!/bin/bash
# Stage 19-D3-4: Train balanced BC reach-pick policy on bidirectional skill dataset.
#
# Usage:
#   bash scripts/train_stage19_d3_balanced_bc.sh
#
# Prerequisites:
#   1. D3-2: Reverse trajectories collected
#   2. D3-3: Balanced dataset built
#
# Output:
#   generated/results/stage_19/balanced_bc_reach_pick_v0_1/

set -euo pipefail

BALANCED_DIR="generated/datasets/stage_19/balanced_stack_skill_dataset_v0_1"
OUTPUT_DIR="generated/results/stage_19/balanced_bc_reach_pick_v0_1"
EPOCHS=200
BATCH_SIZE=64
LR=0.001
DEVICE="${DEVICE:-cuda}"

echo "=== Stage 19-D3-4: Train Balanced BC Reach-Pick ==="
echo "Dataset: ${BALANCED_DIR}"
echo "Output:  ${OUTPUT_DIR}"

if [ ! -d "${BALANCED_DIR}/trajectories" ]; then
    echo "ERROR: Balanced dataset not found at ${BALANCED_DIR}"
    echo "Run D3-2 first to collect reverse trajectories, then D3-3 to build the dataset."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Save training config
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "training_stage": "stage19_d3_balanced_bc",
    "dataset": "${BALANCED_DIR}",
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "lr": ${LR},
    "device": "${DEVICE}",
    "condition_on_phase": true,
    "obs_key": "filtered_obs",
    "notes": "Balanced BC trained on 44 forward + 50 reverse stacking demos."
}
EOF

echo "Training config saved to ${OUTPUT_DIR}/config.json"

# Train with existing train_bc.py
nsm/bin/python -m vlm_nsm.learning.train_bc \
    --demo_dir "${BALANCED_DIR}/trajectories" \
    --obs_key filtered_obs \
    --condition_on_phase \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --device "${DEVICE}" \
    --save "${OUTPUT_DIR}/best.pt" \
    --save_last "${OUTPUT_DIR}/last.pt" \
    --train_log "${OUTPUT_DIR}/train_log.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/train_stdout.log"

echo ""
echo "=== Training complete ==="
echo "Checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Log: ${OUTPUT_DIR}/train_log.jsonl"
