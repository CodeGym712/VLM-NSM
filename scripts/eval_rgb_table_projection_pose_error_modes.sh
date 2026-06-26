#!/usr/bin/env bash
# Compare RGB table-projection pose error across three projection-point modes.
#
# Runs the evaluator with center, bottom_center, and weighted_bottom,
# then prints a summary comparison table.
#
# This script generates its own validation data via Robosuite (--generate_val_data).
# No DashScope/Qwen or pre-existing dataset required.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
CALIB="generated/configs/camera_calibration_agentview_512.json"
DETECTOR="generated/models/rgb_perception_yolov8n_medium/best.pt"
OUTDIR="generated/results/stage_18"
N_FRAMES="${N_FRAMES:-100}"  # 100 frames per mode

mkdir -p "$OUTDIR"

MODES=("center" "bottom_center" "weighted_bottom")
declare -A RESULTS

for MODE in "${MODES[@]}"; do
  echo ""
  echo "============================================"
  echo "  Mode: ${MODE}"
  echo "============================================"

  OUTPUT="${OUTDIR}/table_projection_pose_error_${MODE}.json"
  DEBUG="${OUTDIR}/table_projection_pose_error_${MODE}.jsonl"

  $PYTHON_BIN -m vlm_nsm.experiments.eval_rgb_table_projection_pose_error \
    --camera_calibration "$CALIB" \
    --detector_model_path "$DETECTOR" \
    --detector_conf_threshold 0.15 \
    --projection_point "$MODE" \
    --generate_val_data "$N_FRAMES" \
    --output "$OUTPUT" \
    --save_debug_jsonl "$DEBUG"

  RESULTS["$MODE"]="$OUTPUT"
done

# Print comparison table
echo ""
echo "============================================"
echo "  Comparison Table"
echo "============================================"
printf "%-20s %-12s %-12s %-12s %-12s\n" "projection_point" "mean_xy_err" "median_xy_err" "p90_xy_err" "max_xy_err"
echo "--------------------------------------------------------------------------------"

for MODE in "${MODES[@]}"; do
  OUTFILE="${RESULTS[$MODE]}"
  if [[ -f "$OUTFILE" ]]; then
    MEAN=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('mean_xy_error', 'N/A'):.4f}\" if d.get('mean_xy_error') is not None else 'N/A')")
    MEDIAN=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('median_xy_error', 'N/A'):.4f}\" if d.get('median_xy_error') is not None else 'N/A')")
    P90=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('p90_xy_error', 'N/A'):.4f}\" if d.get('p90_xy_error') is not None else 'N/A')")
    MAX=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('max_xy_error', 'N/A'):.4f}\" if d.get('max_xy_error') is not None else 'N/A')")
    printf "%-20s %-12s %-12s %-12s %-12s\n" "$MODE" "$MEAN" "$MEDIAN" "$P90" "$MAX"
  else
    printf "%-20s %-12s\n" "$MODE" "NO RESULT"
  fi
done
echo ""

echo "=== Done ==="
echo "Results saved to ${OUTDIR}/table_projection_pose_error_{center,bottom_center,weighted_bottom}.json"
