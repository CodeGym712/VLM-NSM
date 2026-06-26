#!/usr/bin/env bash
# Compare RGB table-projection pose error across cameras and projection modes
# using FIXED (y_flip) calibrations.
#
# Runs for agentview, frontview, and birdview.
# Three projection modes per camera: center, bottom_center, weighted_bottom.
#
# No DashScope/Qwen required.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DETECTOR="generated/models/rgb_perception_yolov8n_medium/best.pt"
OUTDIR="generated/results/stage_18"
N_FRAMES="${N_FRAMES:-45}"

mkdir -p "$OUTDIR"

declare -A CALIB_FILES
CALIB_FILES["agentview"]="generated/configs/camera_calibration_agentview_512.json"
CALIB_FILES["frontview"]="generated/configs/camera_calibration_frontview_512.json"
CALIB_FILES["birdview"]="generated/configs/camera_calibration_birdview_512.json"

CAMERAS=("agentview" "frontview" "birdview")
MODES=("center" "bottom_center" "weighted_bottom")

for CAMERA in "${CAMERAS[@]}"; do
  CALIB="${CALIB_FILES[$CAMERA]}"

  for MODE in "${MODES[@]}"; do
    echo ""
    echo "============================================"
    echo "  Camera: ${CAMERA}  Mode: ${MODE}"
    echo "============================================"

    OUTPUT="${OUTDIR}/table_projection_pose_error_${CAMERA}_${MODE}_fixed_calib.json"
    DEBUG="${OUTDIR}/table_projection_pose_error_${CAMERA}_${MODE}_fixed_calib.jsonl"

    $PYTHON_BIN -m vlm_nsm.experiments.eval_rgb_table_projection_pose_error \
      --camera_calibration "$CALIB" \
      --camera_name "$CAMERA" \
      --detector_model_path "$DETECTOR" \
      --detector_conf_threshold 0.15 \
      --projection_point "$MODE" \
      --generate_val_data "$N_FRAMES" \
      --output "$OUTPUT" \
      --save_debug_jsonl "$DEBUG"
  done
done

# Print comparison table
echo ""
echo "============================================"
echo "  Comparison Table (Fixed Calibration)"
echo "============================================"
printf "%-15s %-20s %-12s %-12s %-12s %-12s\n" "camera" "projection_point" "mean_xy" "median_xy" "p90_xy" "max_xy"
echo "--------------------------------------------------------------------------------------------"

for CAMERA in "${CAMERAS[@]}"; do
  for MODE in "${MODES[@]}"; do
    OUTFILE="${OUTDIR}/table_projection_pose_error_${CAMERA}_${MODE}_fixed_calib.json"
    if [[ -f "$OUTFILE" ]]; then
      MEAN=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('mean_xy_error', 'N/A'):.4f}\" if d.get('mean_xy_error') is not None else 'N/A')")
      MEDIAN=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('median_xy_error', 'N/A'):.4f}\" if d.get('median_xy_error') is not None else 'N/A')")
      P90=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('p90_xy_error', 'N/A'):.4f}\" if d.get('p90_xy_error') is not None else 'N/A')")
      MAX=$(python3 -c "import json; d=json.load(open('${OUTFILE}')); print(f\"{d.get('max_xy_error', 'N/A'):.4f}\" if d.get('max_xy_error') is not None else 'N/A')")
      printf "%-15s %-20s %-12s %-12s %-12s %-12s\n" "$CAMERA" "$MODE" "$MEAN" "$MEDIAN" "$P90" "$MAX"
    else
      printf "%-15s %-20s %-12s\n" "$CAMERA" "$MODE" "NO RESULT"
    fi
  done
done

echo ""
echo "=== Done ==="
echo "Results saved to ${OUTDIR}/table_projection_pose_error_{agentview,frontview,birdview}_{center,bottom_center,weighted_bottom}_fixed_calib.json"
