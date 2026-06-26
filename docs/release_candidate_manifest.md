# Release Candidate Manifest

> **Date**: 2026-05-17
> **Status**: ✅ Release Candidate
> **Canonical Deployment Policy**: DAgger reach_pick-only hybrid control

---

## Canonical Artifacts

### Checkpoint

- **Path**: `generated/results/stage_7B-4/dagger_reach_bc/best.pt`
- **Architecture**: MLP(17 → 256 → 256 → 7), ~72K params
- **Training**: Phase-conditioned DAgger Reach BC
- **Training data**: Expert v2 strict_only (100ep, 61,618 steps) + DAgger round 1 (30ep, 18,820 steps) = 80,438 steps aggregated

### Execution Config

- **Path**: `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`
- **Purpose**: Controller gains, thresholds, retry limits for Robosuite stacking

### Deployment Policy

| Parameter | Value |
|:----------|:------|
| Policy | DAgger reach_pick-only |
| BC phases | `reach_pick` only |
| Grasp/Lift/Reach_place/Release/Retreat | Scripted controllers |
| `--fixed_zero_action_dims` | `3 4 5` (orientation dims) |
| `--condition_on_phase` | Enabled |

### Datasets

| Dataset | Path | Samples |
|:--------|:-----|:--------|
| Expert demos (strict) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` | 100ep, 61,618 steps |
| DAgger aggregated | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` | 80,438 steps |
| Expert v2 full | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/` | 100 episodes |

---

## Performance Summary

| Metric | Value |
|:-------|:-----:|
| 100ep success rate | **92.0%** (92/100) |
| 3×50ep avg success rate | **93.3%** (96/90/94%) |
| Preplace failures (100ep) | **0** |
| Preplace failures (3×50ep) | **0.7 avg** |
| E2E language→sim demo | **5/5 (100%)** |
| Scripted baseline | 88.0% |

### Failure Profile (reach_pick-only, 250 combined episodes)

| Failure Type | Count | Rate |
|:-------------|:-----:|:----:|
| Horizon (timeout) | 8 | 3.2% |
| Verify failed | 7 | 2.8% |
| Preplace | 1 | 0.4% |
| Grasp failed | 0 | 0.0% |

---

## Commands

### E2E Language → Simulation Demo

```bash
PYTHONUNBUFFERED=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --output_dir generated/results/release_candidate/e2e_demo
```

### Canonical Eval (20ep)

```bash
PYTHONUNBUFFERED=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
python3 -m vlm_nsm.learning.eval_bc_hybrid \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --num_episodes 20 \
  --bc_phases reach_pick \
  --fixed_zero_action_dims 3 4 5 \
  --condition_on_phase \
  --device cuda \
  --output generated/results/release_candidate/reach_pick_20ep.json
```

---

## Key Reports

| Report | Path |
|:-------|:-----|
| Stage 10 deployment feasibility report | `docs/stage_10_deployment_feasibility_report.md` |
| Simulation protocol | `docs/simulation_protocol.md` |
| Evaluation protocol | `docs/evaluation_protocol.md` |
| Final empirical report | `docs/final_empirical_report.md` |
| E2E demo summary | `generated/report_tables/stage_10/e2e_demo_summary.md` |
| Deployment policy comparison | `generated/report_tables/stage_10/deployment_policy_comparison.md` |
| Reach_pick-only 100ep summary | `generated/report_tables/stage_10/reach_pick_only_100ep_summary.md` |
| BC Stage 7B-5 report | `docs/bc_stage7b5_report.md` |
| Language variation summary | `generated/results/language_variation_summary.json` / `.csv` |
| Failure case analysis | `generated/results/qwen_failure_cases.json` / `.csv` |
| Repair statistics | `generated/results/qwen_alignment_repair_stats.json` / `.csv` |
| Model comparison table | `generated/results/report_table_model_comparison.csv` |
| Prompt comparison table | `generated/results/report_table_prompt_comparison.csv` |
| Alignment benefit table | `generated/results/report_table_alignment_benefit.csv` |
| By-category table | `generated/results/report_table_by_category.csv` |

---

## Known Limitations

1. **Robosuite 1.5.2 seed not fully deterministic**: `env.seed()` is not supported, so random initialization varies across runs. The `--seed` flag is metadata only.
2. **Mock parser by default**: `run_e2e_language_to_sim.py` uses `MockGoalParser` by default. For full NL evaluation, use Qwen2.5 parser (requires GPU + model checkpoint).
3. **Qwen/cached parser optional**: The Qwen2.5 text parser and cached output mode are optional additions, not part of the canonical demo.
4. **No BC-controlled grasp by default**: BC controls only the reach_pick phase. Grasp, lift, reach_place, release, retreat use scripted controllers. This is by design (BC grasp degrades reliability).
5. **Simulation intrinsic failure rate**: Scripted baseline achieves 88.0%, indicating ~12% intrinsic failure rate in Robosuite stacking due to physics randomization.
6. **reach_place BC demoted**: DAgger reach_pick+reach_place (83.3% avg, 43.7/150ep preplace) is a research variant, not suitable for deployment.
7. **Stacking task only**: All evaluations are for 2-block stacking. Other task types (pick-place, 3-block stacking) are not yet validated.
8. **Canonical eval path**: `eval_bc_hybrid.py` with `run_episode()` shared runner is the only valid eval path. The old `eval_bc.py` manual loop is deprecated.

---

## File Checklist

| File | Status |
|:-----|:-------|
| `generated/results/stage_7B-4/dagger_reach_bc/best.pt` | ✅ |
| `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` | ✅ |
| `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` | ✅ |
| `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` | ✅ |
| `scripts/run_canonical_demo.sh` | ✅ |
| `scripts/run_canonical_eval_20ep.sh` | ✅ |
| `docs/stage_10_deployment_feasibility_report.md` | ✅ |
| `docs/simulation_protocol.md` | ✅ |
| `docs/evaluation_protocol.md` | ✅ |
| `generated/results/language_variation_summary.json` | ✅ |
| `generated/results/qwen_failure_cases.json` | ✅ 556 cases |
| `generated/results/qwen_alignment_repair_stats.json` | ✅ 9 repairs |
| `generated/results/report_table_model_comparison.csv` | ✅ |
| `generated/results/report_table_prompt_comparison.csv` | ✅ |
| `generated/results/report_table_alignment_benefit.csv` | ✅ |
| `generated/results/report_table_by_category.csv` | ✅ |
