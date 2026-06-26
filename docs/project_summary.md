# VLM-NSM Project Summary

Vision-Language Model + Neural Symbolic Motion for embodied stacking tasks.

---

## Pipeline Overview

```
NL Instruction → Qwen2.5 TextGoalParser → Alignment Layer → PDDL Planner → Scripted/BC Executor
```

- **Language**: Qwen2.5-32B/14B parse natural language to structured goal
- **Alignment**: Schema validation, object resolution, completeness check, alias repair
- **Planning**: PDDL internal stacking planner (optimal action sequences)
- **Execution**: Robosuite scripted policy + MLP Behavior Cloning (DAgger hybrid)

---

## Key Results

### Language Evaluation (984 samples)

| Metric | Value |
|--------|:-----:|
| Batch jobs | **14/14 completed, 0 failed** |
| Groups | 282 |
| Easy sets | 100% goal success |
| Hard set goal success | ~60% |
| Total failures | 556/984 (56.5%) |
| Alignment repairs | 9 (predicate_alias only) |

#### Failure Taxonomy

| Failure Type | Count | % of Failures | % of Total |
|:-------------|:-----:|:-------------:|:----------:|
| schema_invalid | 294 | 52.9% | 29.9% |
| invalid_object | 157 | 28.2% | 16.0% |
| goal_not_satisfied | 81 | 14.6% | 8.2% |
| json_parse_error | 14 | 2.5% | 1.4% |
| incomplete_goal | 10 | 1.8% | 1.0% |
| **Total** | **556** | **100%** | **56.5%** |

**Dominant failure modes**: schema_invalid + invalid_object = **81.1%** of failures. Hard language generalization is bottlenecked by Qwen2.5 output formatting and object grounding, not by planning or execution.

### Embodied Evaluation (Stacking)

| Policy | Multi-seed Avg | Min | Mean Steps |
|--------|:-------------:|:---:|:----------:|
| Scripted expert | 91.7% | 80% | 785 |
| bc_main (expert-only) | 86.7% | 80% | 744 |
| **DAgger Reach BC (best)** | **93.3%** | **90%** | **738** |
| DAgger (50ep) | 92.0% | — | 734 |

### Canonical Robosuite Deployment Policy

| Parameter | Value |
|:----------|:------|
| Policy | **DAgger reach_pick-only** (hybrid BC + scripted) |
| Checkpoint | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| Config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |
| BC phases | `reach_pick` only (grasp/lift/reach_place/release/retreat = scripted) |
| 100ep success | **92.0%** (92/100) |
| 3×50ep avg | **93.3%** (96/90/94%) |
| E2E language→sim demo | **5/5 (100%)** |
| Preplace failures | 0 (100ep) / 0.7 avg (3×50ep) |

**Key decision**: reach_pick-only is the canonical default. DAgger reach_pick+reach_place (83.3% avg, 43.7/150ep preplace) is a research variant. Scripted baseline (88.0%) serves as simulation ceiling reference.

### Quickstart

```bash
# E2E language → simulation demo (5 demos)
bash scripts/run_canonical_demo.sh

# Canonical BC evaluation (20 episodes)
bash scripts/run_canonical_eval_20ep.sh

# Run all tests
python3 -m pytest tests/ -q --tb=short
```

### Key Constraints

| Constraint | Status |
|:-----------|:------:|
| Shared runner is canonical eval path | ✅ |
| BC for reach_pick only (default) | ✅ |
| Offline MSE does not predict success | ✅ |
| DAgger data requires expert data | ✅ |
| No scale mismatch (all actions in same space) | ✅ |
| No `on(bottom, table)` in E2E stacking goals | ✅ |

### Deprecated / Do Not Use

| Item | Replacement |
|:-----|:------------|
| `eval_bc.py` manual env loop (horizon//6 phase division, zero actions) | `eval_bc_hybrid.py` + `run_episode()` |
| Synthetic phase_id from step budget | Real phase_id from env state machine |
| BC-controlled grasp phase | Scripted grasp (BC reach_pick only) |
| reach_pick+reach_place as default | reach_pick-only default |
| `on(bottom, table)` in stacking goals | Omit table predicate (unreliable after stacking) |

---

## How to Reproduce Key Tables

```bash
# Final report tables (all 8)
python3 -m vlm_nsm.experiments.export_final_report_tables

# Language eval pipeline
python3 -m vlm_nsm.experiments.summarize_language_variation
python3 -m vlm_nsm.experiments.analyze_qwen_failures
python3 -m vlm_nsm.experiments.analyze_alignment_repairs
python3 -m vlm_nsm.experiments.export_report_tables

# Hybrid BC eval
python3 -m vlm_nsm.learning.eval_bc_hybrid --help
```

---

## Canonical Artifact Paths

| Artifact | Path |
|----------|:----:|
| Best execution config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |
| Expert demos | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` |
| DAgger aggregated dataset | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` |
| Best BC model | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| Multi-seed eval results | `generated/results/stage_7B-5/` |
| Language eval summary | `generated/results/language_variation_summary.json` |
| Final report tables | `generated/report_tables/final/` |

---

## Documentation Index

| Doc | Description |
|:----|:------------|
| `docs/bc_stage7b5_report.md` | DAgger robust validation (Stage 7B-5) |
| `docs/final_empirical_report.md` | Integrated language + embodied report |
| `docs/evaluation_protocol.md` | Canonical and deprecated eval paths |
| `docs/artifact_manifest.md` | Complete artifact listing |

---

## System Requirements

- Python 3.10+
- PyTorch (for BC training/eval)
- Robosuite 1.5.2 (for embodied eval)
- MUJOCO_GL=egl (for headless rendering)
- CUDA-capable GPU (recommended)
- ~65GB disk for model checkpoints

---

## Tests

```bash
# Run all tests (no robosuite/qwen dependency)
python3 -m pytest tests/ -q --tb=short

# Expected: 722+ passed, 2 skipped
```
