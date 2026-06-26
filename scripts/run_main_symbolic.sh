#!/usr/bin/env bash
# Run all main symbolic experiments (Hanoi + Stacking)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "===== Hanoi 3 ====="
python3 -m vlm_nsm.experiments.run_hanoi_symbolic --n_blocks 3 --parser mock --planner internal_hanoi --executor symbolic
echo ""

echo "===== Hanoi 4 ====="
python3 -m vlm_nsm.experiments.run_hanoi_symbolic --n_blocks 4 --parser mock --planner internal_hanoi --executor symbolic
echo ""

echo "===== Hanoi 5 ====="
python3 -m vlm_nsm.experiments.run_hanoi_symbolic --n_blocks 5 --parser mock --planner internal_hanoi --executor symbolic
echo ""

echo "===== Stacking 3 ====="
python3 -m vlm_nsm.experiments.run_stacking_symbolic --n_blocks 3 --parser mock --planner internal_stacking --executor symbolic
echo ""

echo "===== Stacking 4 ====="
python3 -m vlm_nsm.experiments.run_stacking_symbolic --n_blocks 4 --parser mock --planner internal_stacking --executor symbolic
echo ""

echo "===== Stacking 5 ====="
python3 -m vlm_nsm.experiments.run_stacking_symbolic --n_blocks 5 --parser mock --planner internal_stacking --executor symbolic
echo ""

echo "===== All main symbolic experiments completed ====="
