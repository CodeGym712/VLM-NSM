# v0.1-rc1 Final Summary

> **Date**: 2026-05-17
> **Status**: Release Candidate 1

## Component Overview

| Component | Metric | Result |
|:----------|:-------|:------:|
| Language eval batch jobs | 14/14 completed | ✅ **0 failed** |
| Language variation samples | 984 samples, 282 groups | ✅ |
| Language eval total failures | 556 (56.5%) | ⚠️ |
| Alignment repairs | 9 (all predicate_alias) | ✅ |
| Canonical Robosuite eval | 100ep success | **92.0%** |
| Canonical Robosuite eval | 3×50ep avg | **93.3%** |
| E2E mock language→sim demo | 5/5 (100%) | ✅ |
| Scripted baseline | 100ep | 88.0% |
| Test suite | pytest | **722 passed, 2 skipped** |

## Dominant Failure Modes

- **schema_invalid** (294, 52.9%): Qwen2.5 output formatting failures
- **invalid_object** (157, 28.2%): Object grounding errors
- Combined: **81.1%** of all failures

## Artifact Links

- Release notes: `docs/release_notes_v0.1.md`
- Release manifest: `docs/release_candidate_manifest.md`
- Demo guide: `docs/demo_guide.md`
- Deployment report: `docs/stage_10_deployment_feasibility_report.md`
- Simulation protocol: `docs/simulation_protocol.md`
- Project summary: `docs/project_summary.md`
