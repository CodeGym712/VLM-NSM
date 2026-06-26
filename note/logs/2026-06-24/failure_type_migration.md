# FailureType Migration Report

> 日期：2026-06-24

## Status

`FailureType` enum (19 values) defined in `core/failure_types.py` with `from_string()`, `is_recoverable()`, `recommended_repair()`.

## Migration Complete

| File | Status |
|------|:---:|
| `core/failure_types.py` | ✅ Source of truth |
| `neuro_symbolic/failure_summarizer.py` | ✅ Uses FailureType patterns internally |
| `neuro_symbolic/repair_proposer.py` | ✅ Uses `recommended_repair()` mapping |

## String Matching Retained (Acceptable)

These are in data analysis/legacy report scripts — not in the core execution pipeline:

| File | Reason to Keep |
|------|------|
| `experiments/stage19_d5_profile_report.py` | Legacy report, reads old trace files |
| `experiments/audit_vla_dataset.py` | Has its own string mapping for VLA audit |
| `experiments/analyze_robosuite_failures.py` | Post-hoc analysis, reads recorded strings |
| `experiments/stage27_cube2_failure_diagnostics.py` | Diagnostic tool, reads recorded strings |
| `experiments/export_embodied_report_tables.py` | Legacy table exporter |

These files read FAILURE STRINGS that were recorded in old trace/summary files. They cannot be migrated because the data was already saved with old string values. No new failures are produced by these files.

## Core Pipeline Status

The core pipeline (grasp_state_machine, place_state_machine, operator_monitor, task_runner) produces failure strings that are compatible with `FailureType.from_string()`. The migration to enum-based checks in NEW code is complete. Old code that reads legacy trace files retains string matching for backward compatibility.
