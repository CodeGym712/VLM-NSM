#!/usr/bin/env bash
# Stage 18-B: Compare scene sources — none vs oracle vs rgb_detector.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTPUT_DIR="generated/results/stage_18"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -m vlm_nsm.experiments.compare_scene_grounding_results \
  --without_scene generated/results/stage_17/scene_grounding_without_scene/summary.json \
  --with_oracle generated/results/stage_17/scene_grounding_with_oracle/summary.json \
  --with_detector generated/results/stage_18/scene_grounding_rgb_detector/summary.json \
  --input generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json \
  --output "${OUTPUT_DIR}/scene_grounding_comparison.json"
