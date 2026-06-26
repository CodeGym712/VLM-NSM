# Windows Local Development Design

## Goal

Make the repository safe and predictable to develop from Windows with Python 3.10 and the minimal dependency set, while keeping simulation, vision, GPU, VLM, and generated-artifact validation on the server.

## Boundaries

- Local Windows: edit code, run pure-Python unit and symbolic tests, validate repository hygiene.
- GitHub: review and synchronize source changes; never store API keys or machine-specific paths.
- Server: run PyTorch, HDF5, MuJoCo/robosuite, vision, model, and generated-result tests.

## Design

The local test entry point uses an explicit YAML manifest of known lightweight test modules. A pytest hook applies `unit` or `symbolic` markers from that manifest, and a small Python runner sets UTF-8 mode before collecting only those modules. This avoids importing optional server dependencies and avoids treating absent generated artifacts as code failures.

Repository hygiene tests prevent reintroduction of API-key-shaped values and `/rhome` paths in active source, tests, scripts, and the README. Runtime paths are resolved from `PROJECT_ROOT`, script location, command-line arguments, or environment variables. Server model paths remain configurable but have no repository-specific defaults.

## Error handling

- The local runner fails clearly if the manifest is empty, duplicated, malformed, or references missing files.
- Qwen scripts fail immediately with an actionable message when their model-path environment variable is absent.
- Metric-FF reports its existing `METRIC_FF_BIN` guidance when no executable is found.

## Verification

1. Reproduce the old collection failure in the minimal environment.
2. Run regression tests for profile parsing, marker application, path portability, and secret scanning.
3. Run the complete manifest-backed local suite with Python 3.10.11.
4. Scan tracked files for secret patterns and active `/rhome` paths.
5. Document that full server tests remain a separate required validation step.
