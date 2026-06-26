#!/usr/bin/env bash
# Stage 18-H2: Targeted Qwen eval for 8 referential/color cases.
#
# Runs only the instructions affected by H2-1 (shared predicate ambiguity)
# and H2-2 (color synonym rules) prompt changes.
#
# Success criteria:
#   - cube on table -> ungroundable
#   - block on table -> ungroundable
#   - closer to edge -> ungroundable
#   - first cube you see -> ungroundable
#   - reddish/greenish -> grounded correctly
#   - lighter/darker -> ungroundable
#   - bright/dull -> ungroundable
#   - nearer/farther -> grounded correctly
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f configs/dashscope_qwen.env ]]; then
  set +x
  # shellcheck disable=SC1091
  source configs/dashscope_qwen.env
fi

: "${QWEN_API_KEY:?Set QWEN_API_KEY to your DashScope API key first.}"
QWEN_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen-plus}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CALIB_AGENT="generated/configs/camera_calibration_agentview_512.json"
CALIB_FRONT="generated/configs/camera_calibration_frontview_512.json"
MEDIUM_MODEL="generated/models/rgb_perception_yolov8n_medium/best.pt"
TARGETED_INPUT="generated/eval_inputs/stage_18/scene_grounding_h2_targeted.json"
OUTPUT_DIR="generated/results/stage_18/qwen_plus_stage18_h2_targeted"

mkdir -p "$OUTPUT_DIR"

echo "=== Stage 18-H2 Targeted Eval (8 cases) ==="
echo "Model: $QWEN_MODEL"
echo "Output: $OUTPUT_DIR/summary.json"
echo ""

"$PYTHON_BIN" -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions "$TARGETED_INPUT" \
  --output "${OUTPUT_DIR}/summary.json" \
  --model "$QWEN_MODEL" \
  --api_base "$QWEN_API_BASE" \
  --qwen_api_key_env QWEN_API_KEY \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 1 \
  --available_objects cube1 cube2 \
  --save_raw_qwen_outputs \
  --scene_inventory rgb_detector \
  --rgb_spatial_mode stereo \
  --camera_calibration "$CALIB_AGENT" \
  --camera_calibration_frontview "$CALIB_FRONT" \
  --table_z 0.0 \
  --object_height 0.05 \
  --detector_model_path "$MEDIUM_MODEL" \
  --detector_conf_threshold 0.15 \
  --projection_point center \
  --enable_predicate_inventory

echo ""
echo "=== Done ==="
echo "Results: ${OUTPUT_DIR}/summary.json"
echo ""
echo "Quick check:"
"$PYTHON_BIN" -c "
import json
with open('${OUTPUT_DIR}/summary.json') as f:
    data = json.load(f)
for r in data['results']:
    inst = r['instruction']
    binding = r.get('predicted_bindings')
    status = r.get('status', '')
    ungroundable = r.get('ungroundable_detected', False)
    if ungroundable:
        print(f'  [UNGROUNDABLE] {inst}')
        print(f'    Reason: {r.get(\"ungroundable_reason\", \"?\")}')
    elif binding:
        print(f'  [GROUNDED] {inst}')
        print(f'    Binding: {binding}')
    else:
        print(f'  [OTHER] {inst}')
        print(f'    Status: {status}')
"
