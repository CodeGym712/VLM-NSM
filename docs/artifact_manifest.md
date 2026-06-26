# Artifact Manifest

> Auto-generated table: `generated/report_tables/final/final_artifact_manifest.md`

## Overview

This manifest catalogs all important artifacts produced across Stages 6B–8:
configs, datasets, checkpoints, evaluation results, report tables, documentation,
and tests.

## Quick Reference

### Configs

| Artifact | Path |
|----------|:----:|
| Best execution config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |

### Expert Demos

| Artifact | Path |
|----------|:----:|
| Expert v2 strict_only (npz) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/trajectories_strict_only.npz` |
| Expert v2 strict_only (per-episode) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` |

### DAgger Datasets

| Artifact | Path |
|----------|:----:|
| DAgger round 1 | `generated/demos/stage_7B-4/dagger_round1_reach/trajectories_dagger_round1_reach.npz` |
| Aggregated reach dataset | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` |

### Checkpoints

| Artifact | Path | Notes |
|----------|:----:|-------|
| DAgger Reach BC (best) | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` | 93.3% multi-seed avg |
| bc_main (expert-only) | `generated/results/stage_7B-3/bc_main/best.pt` | 86.7% multi-seed avg |
| DAgger-only BC (ablative) | `generated/results/stage_7B-5/dagger_only_bc/best.pt` | 0% closed-loop |

### Embodied Evaluation Results

| Artifact | Path |
|----------|:----:|
| Multi-seed comparison | `generated/results/stage_7B-5/` (10 JSON files) |
| DAgger 50ep validation | `generated/results/stage_7B-5/dagger_reach_50ep.json` |
| Phase ablation | `generated/results/stage_7B-5/phase_*.json` |
| Data ablation | `generated/results/stage_7B-5/dagger_only_bc_eval_20ep.json` |
| Stage 8 final 100ep | `generated/results/stage_8/dagger_reach_final_100ep.json` |

### Language Evaluation Results

| Artifact | Path |
|----------|:----:|
| Summary (984 samples) | `generated/results/language_variation_summary.json` |
| Failure cases (556) | `generated/results/qwen_failure_cases.json` |
| Alignment repair stats | `generated/results/qwen_alignment_repair_stats.json` |

### Report Tables

| Artifact | Path |
|----------|:----:|
| Stage 7B-5 tables | `generated/report_tables/stage_7B-5/` |
| Final Stage 8 tables | `generated/report_tables/final/` |

### Documentation

| Artifact | Path |
|----------|:----:|
| Stage 7B-5 report | `docs/bc_stage7b5_report.md` |
| Final empirical report | `docs/final_empirical_report.md` |
| Evaluation protocol | `docs/evaluation_protocol.md` |
| Artifact manifest | `docs/artifact_manifest.md` |
| Project summary | `docs/project_summary.md` |

### Tests

| Artifact | Path |
|----------|:----:|
| Stage 7B-5 report export tests | `tests/test_stage7b5_report_export.py` |
| Policy comparison tests | `tests/test_policy_comparison_table.py` |
| Phase ablation tests | `tests/test_phase_ablation_table.py` |
| Failure audit tests | `tests/test_dagger_failure_audit.py` |
| Final report tests | `tests/test_final_report_tables.py` |
| Eval protocol tests | `tests/test_evaluation_protocol_doc.py` |
| Artifact manifest tests | `tests/test_artifact_manifest.py` |

---

## Reproduction Commands

```bash
# Final report tables
python3 -m vlm_nsm.experiments.export_final_report_tables

# Language eval pipeline
python3 -m vlm_nsm.experiments.summarize_language_variation
python3 -m vlm_nsm.experiments.analyze_qwen_failures
python3 -m vlm_nsm.experiments.analyze_alignment_repairs
python3 -m vlm_nsm.experiments.export_report_tables

# Embodied eval
# See docs/evaluation_protocol.md for commands

# Tests
python3 -m pytest tests/ -q --tb=short
```
