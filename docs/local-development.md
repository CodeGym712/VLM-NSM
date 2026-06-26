# Windows Local Development

This guide sets up the lightweight development environment for source changes, pure-Python unit tests, and symbolic reasoning tests. Simulation, vision, GPU, model, and generated-experiment validation remains a server workflow.

## Prerequisites

- Windows 10 or 11
- Python 3.10
- Git
- PowerShell

## Create the environment

From the repository root:

```powershell
py -3.10 -m venv .venv-minimal
Set-ExecutionPolicy -Scope Process Bypass
.\.venv-minimal\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm the interpreter:

```powershell
python --version
```

The supported baseline is Python 3.10.

## Run local tests

```powershell
python scripts\run_local_tests.py
```

The runner reads `tests/local_test_manifest.yaml`, enables Python UTF-8 mode, collects only the listed modules, and selects tests marked `unit` or `symbolic`. Extra pytest arguments can be appended:

```powershell
python scripts\run_local_tests.py -q
python scripts\run_local_tests.py -x
```

Do not use the old repository-wide marker command as the local entry point. Pytest imports test modules before normal marker filtering, so modules that import optional server packages can fail during collection.

## What the local profile excludes

- PyTorch and HDF5 dataset/model tests
- MuJoCo and robosuite simulation
- OpenCV, YOLO, and other vision stacks
- CUDA and local Qwen model execution
- online DashScope calls
- assertions over generated experiment results that are intentionally absent from Git
- Linux executable-bit checks

Run those checks on the configured GPU server with the full project environment and required artifacts.

## Optional runtime configuration

Online DashScope calls:

```powershell
$env:QWEN_API_KEY = "<read from your secret manager>"
```

Metric-FF outside the repository:

```powershell
$env:METRIC_FF_BIN = "D:\Tools\Metric-FF\ff.exe"
```

Server-side Qwen scripts require `MODEL_PATH`, `QWEN14_PATH`, or `QWEN32_PATH`, depending on the script. No private machine path is used as a default.

Never save credential values in tracked files.

## Development loop

1. Create or switch to a feature branch.
2. Make the source change.
3. Run `python scripts\run_local_tests.py`.
4. Commit and push the feature branch.
5. Pull that branch on the GPU server and run the relevant full simulation/model checks.
6. Merge only after both local and server validation pass.
