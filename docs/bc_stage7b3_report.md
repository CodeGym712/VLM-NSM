# BC Stage 7B-3 Report: Phase-Specific and Residual Hybrid Behavior Cloning

## Overview

Stage 7B-3 extends the Stage 7B-2R BC baseline with:
1. **Phase-filtered BC training** (`--phase-filter`): BC models trained on individual phase data
2. **bc_blend_alpha**: residual blending between BC and scripted actions
3. **Multi-seed evaluation**: hybrid reach-only BC robustness across 3 seeds
4. **Endpoint diagnostics**: EEF-to-target distance error per phase

## Phase-Specific BC Models

### Training

Phase-specific BC models trained with the `--phase-filter` flag on individual phase data:

```bash
# reach_pick model (seed 42)
python3 -m vlm_nsm.learning.train_bc \
  --demo_dir generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only \
  --obs_key filtered_obs \
  --condition_on_phase --phase_key phase_id \
  --phase-filter reach_pick \
  --epochs 100 --batch-size 64 --lr 1e-3 \
  --hidden_dims 256 256 \
  --seed 42 --device cuda \
  --save generated/results/stage_7B-3/bc_reach_pick/best.pt

# reach_place model (seed 42)
python3 -m vlm_nsm.learning.train_bc \
  --demo_dir generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only \
  --obs_key filtered_obs \
  --condition_on_phase --phase_key phase_id \
  --phase-filter reach_place \
  --epochs 100 --batch-size 64 --lr 1e-3 \
  --hidden_dims 256 256 \
  --seed 42 --device cuda \
  --save generated/results/stage_7B-3/bc_reach_place/best.pt
```

| Model | Train Steps | Val Steps | Best Val Loss | Epochs | Training Time |
|-------|:----------:|:---------:|:-------------:|:------:|:------------:|
| reach_pick | 9,236 | 2,213 | 4.8e-5 | 100 | ~210s |
| reach_place | 40,258 | 9,744 | 0.0295 | 100 | ~329s |

The reach_pick model achieves extremely low loss (4.8e-5) because reaching is a simple
regression task (move EEF toward cube1). The reach_place model has higher loss (0.0295)
because the target position varies (cube2 position depends on placement success).

### Per-Phase MSE (Phase-Specific Models)

**Reach_pick model:**
| Phase | Train MSE | Val MSE | N Samples (Train) |
|-------|:--------:|:-------:|:-----------------:|
| reach_pick | 0.00008 | 0.00005 | 9,236 |
| grasp | 2.71 | 2.71 | 9,166 |
| lift | 3.65 | 3.68 | 9,535 |
| reach_place | 7.72 | 7.99 | 40,258 |
| release | — | — | 0 |
| retreat | — | — | 0 |

**Reach_place model:**
| Phase | Train MSE | Val MSE | N Samples (Train) |
|-------|:--------:|:-------:|:-----------------:|
| reach_pick | 3.10 | 3.10 | 9,236 |
| grasp | 3.19 | 3.19 | 9,166 |
| lift | 6.86 | 8.34 | 9,535 |
| reach_place | 0.027 | 0.028 | 40,258 |
| release | — | — | 0 |
| retreat | — | — | 0 |

Both models achieve near-zero MSE on their target phase and high MSE on other phases
(the model has no incentive to learn other phases). This confirms the phase filter
correctly isolates training data.

## bc_blend_alpha: Residual Blending

### Concept

`bc_blend_alpha` controls the blend between BC and scripted actions in hybrid mode:

```
final_action = (1 - alpha) * scripted_action + alpha * bc_action
```

- **alpha = 0.0**: pure scripted (proportional controller)
- **alpha = 1.0**: pure BC
- **alpha = 0.5**: equal blend

### Alpha Sweep Results (Stage 7B-2R model on strict_success data)

| Alpha | Success Rate | Mean Steps | BC Steps | Scripted Steps |
|:-----:|:-----------:|:----------:|:--------:|:--------------:|
| 0.25 | 40% | 587 | 202 | 404 |
| 0.50 | 15% | 696 | 251 | 500 |
| 0.75 | 10% | 785 | 298 | 546 |
| 1.00 | 5% | 735 | 272 | 483 |

**Interpretation:** Higher alpha = worse performance, suggesting the BC model
actions are actively harmful in closed loop. Even 25% BC degrades the scripted
baseline (88-91%) to 40%. Consistent with compounding error from endpoint diag.

## Multi-Seed Hybrid Reach-Only Evaluation

### Config
- **Model**: Phase-conditioned MLP (17->256->256->7, 72K params)
- **Control mode**: Hybrid — BC for `reach_pick` + `reach_place`, scripted for others
- **Training data**: Expert v2 strict_only (100 episodes, 80/20 split)
- **Action norm**: min_action_std=1e-3, masked MSE for dims 3-5
- **Variation**: Random seed changes the train/val split only

### Results (20 episodes per seed)

| Seed | Success | Mean Steps | BC Steps | Scripted Steps |
|:----:|:-------:|:----------:|:--------:|:--------------:|
| 0 | 95% | 985.5 | 585.7 | 415.2 |
| 1 | 90% | 983.5 | 583.2 | 417.3 |
| 2 | 100% | 1000.0 | 609.4 | 391.6 |
| **Mean ± Std** | **95.0 ± 5.0%** | **989.7 ± 9.4** | **592.8 ± 14.1** | **408.1 ± 13.9** |

### Key Findings
- **95% average success rate** across 3 seeds (σ=5%) was observed in earlier
  evaluation rounds (model checkpoint not preserved).
- **Independent reproduction**: strict_only model (newly trained) achieves only
  **10%** in hybrid reach-only eval with the same config.
- Stage 7B-2R model (strict_success data) also achieves **0-5%**.
- The discrepancy suggests the seed evals used a different checkpoint, different
  eval parameters (e.g., action denormalization), or had different env config.
- **Overall finding: BC currently does NOT work reliably** in closed-loop for this task.

## Endpoint Diagnostics

<!-- Results from generated/report_tables/stage_7B-3/*_diag/bc_endpoint_errors.md -->

Endpoint diagnostics measure EEF-to-target distance at the end of each phase,
comparing ground-truth (dataset) vs BC policy rollout (chained predictions).

### Main BC Model Endpoint Errors (Val Set, 20 episodes)

| Phase | N | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 |
|------|:-:|:-----------:|:----------:|:---------------:|:--------------:|
| reach_pick | 20 | 0.1163 | 0.1209 | 10.6615 | 14.2326 |
| grasp | 20 | 0.0198 | 0.0200 | 7.8198 | — |
| lift | 20 | 0.1301 | 0.1306 | 9.0790 | — |
| reach_place | 20 | 0.0567 | 0.0649 | 10.6030 | — |

**Key Finding:** Dataset endpoint errors are small (0.02-0.13m), meaning the
demonstrator reliably reaches each target. But BC policy rollout errors are
extremely large (7-10m) — orders of magnitude larger than the workspace (~0.3m).

This is classic **BC compounding error** (covariate shift). The model was trained
with teacher forcing (one-step prediction from correct states), but during rollout
errors accumulate, driving the EEF into out-of-distribution states where the model
produces increasingly incorrect actions.

## Conclusions

1. **Phase-filtered training works**: reach_pick and reach_place models achieve
   near-zero MSE on their target phase (4.8e-5 and 0.0295 respectively).
2. **BC compounding error is severe**: Endpoint diagnostics show 10m+ drift in
   closed-loop rollout for all models — orders of magnitude above workspace size.
   This is a fundamental limitation of behavioral cloning with teacher forcing.
3. **bc_blend_alpha degrades performance**: Any BC blend (even 25%) reduces
   success rate below the scripted baseline, suggesting BC actions are actively
   harmful in the current implementation.
4. **Phase-filtered models are unsuitable for hybrid eval**: They have no closed-loop
   stability outside their training distribution (non-target phases).
5. **Scripted control remains more reliable** than BC for this stacking task.

### Recommended Next Steps

- **DAgger or interactive correction** to address the covariate shift problem
- **Regularize BC actions** to be more conservative (reduce action magnitude)
- **Train on longer rollouts** instead of single-step teacher forcing
- **Use BC only for low-level refinement** (not full reach phases)

## Post-Stage 7B-5 Corrections

The following conclusions from Stage 7B-3 have been revised by Stage 7B-4/7B-5 findings:

1. **Scale mismatch hypothesis revised**: The Stage 7B-3 claim of "scale mismatch" between BC and scripted actions is not supported by the action space audit in Stage 7B-4. All action sources (dataset, BC denormalized, scripted) operate in the same [-1, 1] normalized space with similar distributions. The real problem is covariate shift.

2. **Old eval path deprecated**: The `eval_bc.py` manual env loop (horizon//6 phase division, zero actions for non-BC phases) produces invalid results. The shared `run_episode()` in `eval_bc_hybrid.py` is the canonical eval path. Same bc_main model achieves 83% via shared runner vs 0-10% via old eval.

3. **DAgger resolves the covariate shift**: Stage 7B-4 DAgger training (on-policy data collection + aggregated training) achieves 90% closed-loop success. Stage 7B-5 multi-seed validation confirms 93.3% avg across 3 seeds and 92.0% over 50 episodes.

4. **Offline MSE not predictive**: Stage 7B-4 confirms that DAgger model has higher offline MSE but better closed-loop performance. Do not use offline MSE as primary success metric.

## Files Changed

| File | Change |
|------|--------|
| `src/vlm_nsm/learning/train_bc.py` | `--phase-filter` CLI arg for phase-specific training |
| `src/vlm_nsm/learning/eval_bc.py` | `bc_blend_alpha` parameter for residual blending |
| `src/vlm_nsm/learning/bc_diagnostics.py` | Endpoint error computation (EEF-to-target distance) |
| `src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py` | `bc_blend_alpha` in recording wrapper |
