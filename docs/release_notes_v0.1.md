# Release Notes — v0.1-rc1

> **Date**: 2026-05-17
> **Status**: Release Candidate 1
> **Project**: VLM-NSM (Vision-Language Model + Neural Symbolic Motion)

---

## Overview

VLM-NSM extends the original NSM pipeline with language-aligned symbolic planning and
symbol-conditioned embodied skill execution for robotic stacking tasks. This release
candidate freezes the canonical deployment policy and artifacts for external validation.

---

## Canonical Deployment Policy

| Parameter | Value |
|:----------|:------|
| Policy | **DAgger reach_pick-only** (hybrid BC + scripted) |
| BC phases | `reach_pick` only |
| Scripted phases | grasp, lift, reach_place, release, retreat |
| Checkpoint | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| Config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |

---

## Key Metrics

### Robosuite Stacking

| Metric | Value |
|:-------|:-----:|
| 100ep success | **92.0%** (92/100) |
| 3×50ep avg | **93.3%** (96/90/94%) |
| Preplace failures (100ep) | **0** |
| Scripted baseline | 88.0% |
| DAgger reach+place (research) | 83.3% avg |

### End-to-End Language → Simulation

| Demo | Result |
|:-----|:------:|
| 5 diverse NL instructions | **5/5 (100%)** |
| BC steps per demo | 96-105 |
| Total steps per demo | 774-798 |

### Language Evaluation

| Metric | Value |
|:-------|:-----:|
| Total samples | 984 |
| Batch jobs | **14/14 completed, 0 failed** |
| Groups | 282 |
| Easy set goal success | 100% |
| Hard set goal success | ~60% |
| Total failures | 556/984 (56.5%) |
| Alignment repairs | 9 (predicate_alias) |

#### Failure Taxonomy

| Failure Type | Count | % of Failures | Description |
|:-------------|:-----:|:-------------:|:------------|
| schema_invalid | 294 | 52.9% | Qwen output did not match expected JSON schema |
| invalid_object | 157 | 28.2% | Referenced object not in valid object list |
| goal_not_satisfied | 81 | 14.6% | Parsed goal did not match expected goal |
| json_parse_error | 14 | 2.5% | Qwen output was not valid JSON |
| incomplete_goal | 10 | 1.8% | Goal missing required predicates |
| **Total** | **556** | **100%** | |

**Conclusion**: Hard language failures are dominated by **schema validity** and **object grounding** issues (schema_invalid + invalid_object = 81.1% of all failures). This suggests the Qwen2.5 parser's output formatting and object name resolution are the primary bottlenecks for hard language variations.

---

## Artifacts

| Artifact | Path |
|:---------|:-----|
| Best BC model | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| Execution config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |
| Expert demos | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` |
| DAgger dataset | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` |
| E2E demo results | `generated/results/stage_10/e2e_sim_demo/summary.json` |
| Language eval summary | `generated/results/language_variation_summary.json` |
| Language eval CSV | `generated/results/language_variation_summary.csv` |
| Failure cases | `generated/results/qwen_failure_cases.json` / `.csv` |
| Repair stats | `generated/results/qwen_alignment_repair_stats.json` / `.csv` |
| Model comparison table | `generated/results/report_table_model_comparison.csv` |
| Prompt comparison table | `generated/results/report_table_prompt_comparison.csv` |
| Alignment benefit table | `generated/results/report_table_alignment_benefit.csv` |
| By-category table | `generated/results/report_table_by_category.csv` |
| Release manifest | `docs/release_candidate_manifest.md` |
| Demo guide | `docs/demo_guide.md` |
| Deployment report | `docs/stage_10_deployment_feasibility_report.md` |
| Simulation protocol | `docs/simulation_protocol.md` |

---

## Reproduction Commands

```bash
# E2E language → simulation demo
bash scripts/run_canonical_demo.sh

# Canonical BC eval (20 episodes)
bash scripts/run_canonical_eval_20ep.sh

# Run all tests (no GPU required)
python3 -m pytest tests/ -q --tb=short
```

---

## Known Limitations

1. **Robosuite 1.5.2 seed not deterministic**: `env.seed()` unsupported; random init varies per run.
2. **Mock parser by default**: E2E demo uses `MockGoalParser`. Full NL requires Qwen2.5 parser (separate GPU).
3. **No BC-controlled grasp**: BC controls only reach_pick. Grasp/lift/reach_place/release/retreat are scripted.
4. **~12% intrinsic sim failure rate**: Scripted baseline 88.0% — some failures are simulation artifacts.
5. **reach_place BC demoted**: DAgger reach_pick+reach_place (83.3%) is research variant only (preplace bottleneck).
6. **Stacking task only**: Only 2-block stacking validated. Other tasks (pick-place, 3-block) not validated.
7. **eval_bc.py deprecated**: Old manual env loop is invalid; use `eval_bc_hybrid.py` + `run_episode()`.

---

## Semantic Constraints

- `build_stacking_goal` uses `on(top, bottom)` — do not add `on(bottom, table)`
- `InternalStackingPlanner` action order: `[obj, "table", target]` where obj=top block, target=bottom block
- Canonical bc_phases: `reach_pick` only — do not default to `reach_pick+reach_place`
- Goal predicates use bottom-to-top stacking convention
