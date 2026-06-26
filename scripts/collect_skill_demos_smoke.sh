#!/usr/bin/env bash
# Stage 19 P0: collect skill-labeled demonstrations with the current teacher.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN="${PYTHON_BIN:-python}"

MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" -m vlm_nsm.experiments.collect_skill_demonstrations \
  --output_dir generated/datasets/skill_demos_v0_1_smoke \
  --num_episodes "${NUM_EPISODES:-50}" \
  --task two_block_stack \
  --executor scripted \
  --seed "${SEED:-0}" \
  --save_trajectories \
  --teacher_mode hybrid_reach_pick \
  --device "${DEVICE:-cuda}"
