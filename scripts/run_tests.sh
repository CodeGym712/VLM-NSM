#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
if "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pytest -q
else
  "$PYTHON_BIN" tests/run_all.py
fi
