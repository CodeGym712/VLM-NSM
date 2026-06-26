# v0.1-rc1 Freeze

> **Date**: 2026-05-17
> **Status**: ❄️ **Frozen Release Candidate**
>
> All artifacts, policies, and documentation are frozen as of this date.
> No further changes to the canonical deployment pipeline will be made
> without a new release candidate version.

---

## 1. Canonical Deployment Policy

| Parameter | Value |
|:----------|:------|
| Policy | **DAgger reach_pick-only** (hybrid BC + scripted) |
| BC phases | `reach_pick` only |
| Scripted phases | grasp, lift, reach_place, release, retreat |
| Checkpoint | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| Config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |
| Eval entry point | `eval_bc_hybrid.py` + `run_episode()` shared runner |
| E2E demo entry point | `run_e2e_language_to_sim.py` |

### Semantic Constraints (frozen)

- `build_stacking_goal` → `on(top, bottom)` only — do **not** add `on(bottom, table)`
- `InternalStackingPlanner` action order → `[obj, "table", target]` (obj = top block, target = bottom block)
- Canonical `bc_phases` → `reach_pick` only
- `eval_bc.py` manual loop → **deprecated**, use `eval_bc_hybrid.py`

---

## 2. Language Variation Evaluation

| Metric | Value |
|:-------|:------|
| Batch jobs | **14/14 completed, 0 failed** |
| Total samples | 984 (282 groups) |
| Easy set goal success | 100% |
| Hard set goal success | ~60% |
| Total failures | 556 (56.5%) |
| Alignment repairs | 9 (all predicate_alias) |

### Failure Taxonomy

| Type | Count | % of Failures |
|:-----|:-----:|:-------------:|
| schema_invalid | 294 | 52.9% |
| invalid_object | 157 | 28.2% |
| goal_not_satisfied | 81 | 14.6% |
| json_parse_error | 14 | 2.5% |
| incomplete_goal | 10 | 1.8% |
| **Total** | **556** | **100%** |

**Key finding**: 81.1% of hard-set failures stem from Qwen2.5 output formatting
(schema validity) and object grounding — not from planning or execution.

---

## 3. Robosuite Stacking Evaluation

| Metric | Value |
|:-------|:------|
| Scripted baseline (100ep) | 88.0% |
| DAgger reach+place (3×50ep avg) | 83.3% (research variant) |
| **DAgger reach_pick-only (100ep)** | **92.0%** |
| **DAgger reach_pick-only (3×50ep avg)** | **93.3%** |
| Preplace failures (100ep) | 0 |
| Preplace failures (3×50ep) | 0.7 avg |

**Key decision**: reach_pick-only is the canonical default. reach_place BC is
demoted to research variant due to preplace bottleneck (avg 43.7/150ep).

---

## 4. End-to-End Language → Simulation Demo

| Demo | Instruction | Result |
|:-----|:------------|:------:|
| 1 | Put the red cube on the green cube | ✅ |
| 2 | Stack cube A on cube B | ✅ |
| 3 | Place the cube on top of the other cube | ✅ |
| 4 | Move the block onto the target block | ✅ |
| 5 | Stack the red block on top of the green one | ✅ |

**5/5 (100%)** — all demos successful with BC reach_pick-only control.

---

## 5. Test Status

| Suite | Result |
|:------|:------:|
| Full pytest | **732 passed, 2 skipped** |
| Skipped (robosuite-required) | 2 tests |
| Smoke tests (Stage 12) | 23 tests ✅ |
| Release manifest tests | 9 tests ✅ |
| Release notes tests | 13 tests ✅ |
| Demo guide tests | 8 tests ✅ |
| Cached Qwen schema tests | 6 tests ✅ |

---

## 6. Frozen Artifacts

### Core Artifacts

| Artifact | Path |
|:---------|:------|
| Best BC model | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| Execution config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |
| Expert demos | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` |
| DAgger aggregated dataset | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` |

### Language Evaluation Artifacts

| Artifact | Path |
|:---------|:------|
| Language eval summary | `generated/results/language_variation_summary.json` |
| Language eval CSV | `generated/results/language_variation_summary.csv` |
| Failure cases | `generated/results/qwen_failure_cases.json` |
| Failure cases CSV | `generated/results/qwen_failure_cases.csv` |
| Repair stats | `generated/results/qwen_alignment_repair_stats.json` |
| Repair stats CSV | `generated/results/qwen_alignment_repair_stats.csv` |
| Model comparison table | `generated/results/report_table_model_comparison.csv` |
| Prompt comparison table | `generated/results/report_table_prompt_comparison.csv` |
| Alignment benefit table | `generated/results/report_table_alignment_benefit.csv` |
| By-category table | `generated/results/report_table_by_category.csv` |

### Robosuite Evaluation Artifacts

| Artifact | Path |
|:---------|:------|
| 100ep results | `generated/results/stage_9/dagger_reach_pick_only_100ep.json` |
| 3×50ep results | `generated/results/stage_10/reach_pick_only_seed{0,1,2}_50ep.json` |
| 100ep summary | `generated/report_tables/stage_10/reach_pick_only_100ep_summary.md` |
| Deployment comparison | `generated/report_tables/stage_10/deployment_policy_comparison.md` |

### E2E Demo Artifacts

| Artifact | Path |
|:---------|:------|
| E2E demo results | `generated/results/stage_10/e2e_sim_demo/summary.json` |
| E2E demo summary | `generated/report_tables/stage_10/e2e_demo_summary.md` |
| Cached Qwen inputs | `generated/results/stage_12/cached_qwen_e2e_inputs.json` |

### Scripts

| Script | Purpose |
|:-------|:--------|
| `scripts/run_canonical_demo.sh` | One-command E2E language→sim demo |
| `scripts/run_canonical_eval_20ep.sh` | One-command canonical BC eval |

### Documentation

| Doc | Path |
|:----|:------|
| Release notes | `docs/release_notes_v0.1.md` |
| Release manifest | `docs/release_candidate_manifest.md` |
| Freeze documentation | `docs/v0_1_rc1_freeze.md` (this file) |
| Demo guide | `docs/demo_guide.md` |
| Project summary | `docs/project_summary.md` |
| Deployment feasibility report | `docs/stage_10_deployment_feasibility_report.md` |
| Simulation protocol | `docs/simulation_protocol.md` |
| Evaluation protocol | `docs/evaluation_protocol.md` |
| Final empirical report | `docs/final_empirical_report.md` |
| Final summary table | `generated/report_tables/final/final_v0_1_rc1_summary.md` |

---

## 7. Known Limitations (frozen)

1. **Robosuite 1.5.2 seed not deterministic**: `env.seed()` unsupported; random init varies per run.
2. **Mock parser by default**: E2E demo uses `MockGoalParser`. Full NL requires Qwen2.5 parser (separate GPU).
3. **No BC-controlled grasp**: BC controls only reach_pick. All other phases are scripted.
4. **~12% intrinsic sim failure rate**: Scripted baseline 88.0% indicates simulation ceiling.
5. **reach_place BC demoted**: DAgger reach_pick+reach_place (83.3%) is research variant only.
6. **Stacking task only**: Only 2-block stacking validated. Pick-place, 3-block, etc. not validated.
7. **eval_bc.py deprecated**: Old manual env loop is invalid.
8. **Language eval hard set**: 56.5% failure rate on hard language variations, dominated by schema/grounding issues.

---

## 8. Post-Release Tasks (optional)

These tasks are explicitly **not required** for the v0.1-rc1 freeze and should
be considered for a future release:

1. **Cached Qwen simulation replay**: Run `run_e2e_language_to_sim` with
   `--parser cached_qwen --cached_outputs generated/results/stage_12/cached_qwen_e2e_inputs.json`
   to validate the full pipeline with pre-computed Qwen parse results.
2. **Demo videos**: Record 1-2 screen captures of the E2E demo with
   `render_mode="screen"` and store in `assets/demo/`.
3. **Qwen2.5 online parser integration**: Replace `MockGoalParser` with
   `QwenTextGoalParser` for true NL understanding in the demo pipeline.
4. **Multi-task generalization**: Extend evaluation to pick-place and 3-block stacking.
5. **DAgger round 2**: Targeted data collection if reach_pick failures become dominant.
