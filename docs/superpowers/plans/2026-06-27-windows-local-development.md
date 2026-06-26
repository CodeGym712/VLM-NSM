# Windows Local Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a secret-free, path-portable, reproducible Windows development workflow for the pure-Python and symbolic portions of VLM-NSM.

**Architecture:** A manifest-backed local pytest profile collects only verified lightweight modules and assigns real `unit`/`symbolic` markers. Active paths are derived from the repository or explicit environment configuration, and repository-hygiene tests guard both decisions.

**Tech Stack:** Python 3.10, pytest, PyYAML, PowerShell/Windows, Bash/server scripts

---

### Task 1: Add local-profile regression tests

**Files:**
- Create: `tests/test_local_profile.py`
- Create: `tests/test_repository_hygiene.py`

- [ ] Add tests requiring a parseable local manifest, unique existing paths, marker categories, no key-shaped values in the migration report, and no `/rhome` paths in active files.
- [ ] Run the new tests and confirm they fail because the manifest/helper is absent and the hygiene findings still exist.

### Task 2: Implement the lightweight pytest profile

**Files:**
- Create: `tests/local_test_manifest.yaml`
- Create: `tests/local_profile.py`
- Create: `tests/conftest.py`
- Create: `scripts/run_local_tests.py`
- Modify: `tests/deprecated/conftest.py`

- [ ] Add a curated set of already passing pure-Python unit and symbolic modules to the manifest.
- [ ] Load and validate the manifest in one helper shared by pytest and the runner.
- [ ] Apply `unit`/`symbolic` markers during collection.
- [ ] Make the runner enable UTF-8 mode and collect only manifest entries.
- [ ] Update the deprecated-suite hook to the pytest 9 `collection_path` signature.
- [ ] Run profile tests and a marker-selection smoke test.

### Task 3: Remove secrets and machine-specific paths

**Files:**
- Replace: `note/migration_audit_report.md`
- Modify: `tests/test_mock_embodied_executor.py`
- Modify: `src/vlm_nsm/planners/metric_ff_adapter.py`
- Modify: `src/vlm_nsm/experiments/run_language_matrix.py`
- Modify: `scripts/run_all_remaining.sh`
- Modify: `scripts/run_qwen14_hard_language.sh`
- Modify: `scripts/run_qwen32_hard_language.sh`
- Modify: `scripts/run_qwen_hanoi_language.sh`
- Modify: `scripts/run_qwen_language_all.sh`
- Modify: `scripts/run_qwen_model_comparison.sh`
- Modify: `scripts/run_qwen_stacking_language.sh`
- Modify: `README.md`

- [ ] Replace the sensitive historical report with a concise redacted audit record.
- [ ] Resolve the embodied test config from `PROJECT_ROOT`.
- [ ] Remove the private Metric-FF fallback.
- [ ] Make model and repository paths explicit environment inputs or script-relative paths.
- [ ] Run the repository-hygiene tests and confirm they pass.

### Task 4: Document and verify local development

**Files:**
- Create: `docs/local-development.md`
- Modify: `README.md`

- [ ] Document Python 3.10 setup, editable installation, local test command, server-only scope, and environment variables.
- [ ] Link the guide from the README and replace the server-specific quick-start path.
- [ ] Run `python scripts/run_local_tests.py` in `.venv-minimal`.
- [ ] Run targeted path and profile regression tests.
- [ ] Review `git diff`, verify no secret-shaped values remain, and amend the branch commit so the cleaned branch history does not retain the key.
