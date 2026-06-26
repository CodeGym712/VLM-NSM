#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python3 -m vlm_nsm.experiments.run_hanoi_symbolic --n_blocks 5 --parser mock --planner internal_hanoi --executor symbolic
