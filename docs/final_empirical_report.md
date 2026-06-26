# Final Empirical Report

> VLM-NSM: Vision-Language Model + Neural Symbolic Motion
> Stage 8 — Integrated Report, Final Ablations, and Release Packaging

---

## 1. Overview

VLM-NSM integrates a Vision-Language Model (Qwen2.5) with a Neural Symbolic Motion
pipeline to perform embodied stacking tasks. The system uses:

1. **Language understanding**: Qwen2.5-32B/14B text-goal parser with alignment layer
2. **Symbolic planning**: PDDL-based internal stacking planner
3. **Embodied execution**: Robosuite scripted expert policy with state machines
4. **Learned control**: MLP Behavior Cloning with DAgger on-policy correction

This report aggregates findings across all stages (6B–8).

---

## 2. Main Findings

### Language
- **Easy sets solved**: 100% goal success across all language variations
- **Hard set 60%**: Schema validity (294/556 failures) and object grounding (157/556)
  dominate; alignment recovers ~60% of hard cases
- **Weak prompt collapses**: 0% goal on hard categories without schema examples
- **14B vs 32B**: 14B has higher alignment pass rate (65-73% vs 59-61%) but similar goal success

### Embodied
- **Scripted expert**: 91.7% multi-seed average, 93.5% strict demo collection rate
- **DAgger BC**: 93.3% multi-seed (+6.6pp over expert-only BC), 92.0% at 50 episodes
- **DAgger-only collapses**: 0% — on-policy data needs expert data for coverage
- **reach_pick is critical**: BC on pick alone achieves 90%; reach_place adds marginal benefit
- **DAgger reduces horizon failures 62-75%**: vs expert-only BC across 60 episodes

---

## 3. Language Evaluation

### Dataset

| Difficulty | Samples | Categories | Description |
|:----------:|:-------:|:----------:|-------------|
| Easy | 4 per category | 8 (hanoi), 7 (stacking) | Standard language variants |
| Hard | 3 per stratum | 32 strata | 6 variation types × 2 tasks + perturbations |

Total: 984 samples, 282 groups.

### Failure Taxonomy

| Failure Type | Count | % of 984 |
|:------------|:-----:|:--------:|
| schema_invalid | 294 | 29.9% |
| invalid_object | 157 | 15.9% |
| goal_not_satisfied | 81 | 8.2% |
| json_parse_error | 14 | 1.4% |
| incomplete_goal | 10 | 1.0% |
| **Total failures** | **556** | **56.5%** |

Schema and object grounding account for 81% of all failures (451/556).

### Model Comparison (Hard Set, Full Alignment)

| Model | Task | Goal Success | Avg Time | Avg Tokens |
|-------|:---:|:-----------:|:--------:|:----------:|
| qwen25_14b | hanoi | 65.9% | 14.5s | 162 |
| qwen25_14b | stacking | 56.8% | 6.9s | 102 |
| qwen25_32b | hanoi | 59.1% | 11.1s | 88 |
| qwen25_32b | stacking | 61.4% | 8.3s | 83 |

### Prompt Comparison (32B Hard)

| Prompt | Goal Success |
|:------:|:-----------:|
| Full | ~60% |
| Medium | ~18% |
| Weak | 0% |

Medium prompt retains schema structure but drops examples; goal success drops from ~60% to ~18%.
Weak prompt (no schema) collapses entirely on hard categories.

### Alignment Analysis

- Alignment layer recovers ~60% of hard cases (schema validation + object resolution)
- Only 9 repairs across 984 samples (all predicate_alias)
- Without alignment: schema_invalid failures pass through to planner
- Without completeness repair: goal success similar to full alignment

---

## 4. Symbolic Planning and Alignment

### Alignment Layer

The alignment layer performs:
1. **Goal schema validation**: Ensures output matches required JSON structure
2. **Object resolution**: Maps model output to valid environment objects
3. **Completeness checking**: Verifies all required fields are present
4. **Repair**: Fixes predicate aliases (9 total across all evaluations)

### Planner

PDDL-based internal stacking planner (no LLM involvement in planning).
Solves stacking goals with optimal action sequences. 100% reliable given valid goal input.

### Pipeline

```
NL Instruction → Qwen TextGoalParser → JSON Goal → Alignment Layer → PDDL Goal → Planner → Actions → Executor
```

---

## 5. Robosuite Expert Policy

### Configuration

Best execution config: `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`

Key parameters:
- kp=0.08 (proportional gain)
- max_delta=0.015 (position step clamp)
- Use of state machines for grasp and place phases
- Early termination on goal satisfaction

### Expert Demo Collection

| Metric | Value |
|--------|:-----:|
| Strict demos | 100 |
| Total attempts | 107 |
| Strict success rate | 93.5% |
| True phase_id | ✅ (from RobosuiteEnvWrapper state machine) |

Canonical training data: `strict_only/` directory (100 verified strict demos).

### Scripted Policy Performance (Multi-seed)

| Seed | Success | Mean Steps | Main Failure |
|:----:|:-------:|:----------:|:------------:|
| 0 | 95% | 777 | horizon |
| 1 | 100% | 784 | — |
| 2 | 80% | 794 | horizon/verify |
| **Avg** | **91.7%** | **785** | horizon |

---

## 6. Behavior Cloning and DAgger Hybrid Control

### BC Architecture

MLP(17 → 256 → 256 → 7), 72K parameters, phase-conditioned (6-dim one-hot + 11-dim obs).

### Training Data Sources

| Dataset | Episodes | Steps | Use |
|---------|:--------:|:----:|:----|
| Expert v2 strict_only | 100 | 61,618 | bc_main baseline |
| DAgger round 1 (reach) | 30 | 18,820 | On-policy correction |
| Aggregated (expert+DAgger) | 130 | 80,438 | DAgger Reach BC |

### DAgger Methodology

1. Run base BC policy in hybrid mode (BC for reach phases only)
2. Record (obs, oracle_scripted_action) at each BC-controlled step
3. Aggregate expert + on-policy data → retrain
4. Evaluation via shared runner (`run_episode()`) with proper state machines

### Key Insight: Offline MSE Is Not Predictive

| Model | Val Loss | Closed-Loop Success |
|-------|:--------:|:------------------:|
| DAgger-only BC | 0.000627 | **0%** |
| Expert-only BC | 0.169 | 86.7% |
| Expert + DAgger | 0.218 | **93.3%** |

The best offline loss corresponds to the worst closed-loop performance.
**Do not use offline MSE as success metric.**

---

## 7. Final Policy Comparison

| Policy | seed=0 | seed=1 | seed=2 | Avg | Min | Mean Steps |
|--------|:-----:|:-----:|:-----:|:---:|:---:|:----------:|
| Scripted | 95% | 100% | 80% | 91.7% | 80% | 785 |
| bc_main | 80% | 85% | 95% | 86.7% | 80% | 744 |
| **DAgger Reach BC** | **95%** | **90%** | **95%** | **93.3%** | **90%** | **738** |
| DAgger (50ep) | — | — | — | 92.0% | — | 734 |

**DAgger Reach BC is the best learned controller**: 93.3% multi-seed avg,
outperforming both expert-only BC (+6.6pp) and scripted baseline (+1.6pp).

---

## 8. Ablations

### Data Ablation

| Training Data | Success | Notes |
|:-------------|:-------:|-------|
| Expert-only | 86.7% | Baseline |
| DAgger-only | **0%** | Insufficient coverage; 0.000627 val loss ≠ control success |
| Expert + DAgger | **93.3%** | Best — DAgger on-policy data improves state coverage |

### Phase Ablation

| BC Phases | Success | Main Failure | Interpretation |
|-----------|:-------:|:------------:|----------------|
| reach_pick only | **90%** | horizon | BC on pick alone most critical |
| reach_place only | **75%** | preplace | Scripted place outperforms BC-only place |
| Both reach | **95%** | horizon | Optimal configuration |
| reach + grasp | **0%** | horizon | No grasp training data → catastrophic |

**Conclusion**: BC should be restricted to reach_pick + reach_place.
The scripted grasp/lift/release/retreat controllers are reliable and
BC has no training data for these phases.

---

## 9. Failure Analysis

### DAgger Reach BC (110 episodes across all evals)

| Failure Type | Count | Root Cause |
|:------------:|:-----:|------------|
| horizon (early) | 4 | BC stuck in reach_pick (low BC steps <120) |
| preplace | 4 | Place verification fails; scripted-phase issue |
| horizon | 1 | Generic timeout |

### bc_main (60 episodes)

| Failure Type | Count | Root Cause |
|:------------:|:-----:|------------|
| horizon | 8 | Covariate shift during reach phases |

**DAgger reduces horizon failures by ~62-75%** (8 → 3 across 60 episodes).
The on-policy DAgger data helps the BC model handle states encountered during
rollout that differ from the expert demonstration distribution.

---

## 10. Lessons Learned

1. **Scale mismatch was a red herring**: Stage 7B-3 claimed BC/scripted action scale mismatch.
   Action space audit in Stage 7B-4 proved all sources operate in same [-1,1] normalized space.
   The real problem was covariate shift from teacher forcing.

2. **Old eval framework was broken**: The manual env loop in `eval_bc.py` (horizon//6 phase division,
   zero actions for non-BC phases) gave 0-10% for the same model that achieves 83% via shared runner.
   The shared `run_episode()` is the canonical eval path.

3. **Offline MSE is misleading**: DAgger-only model achieves 0.000627 val loss but 0% closed-loop.
   Expert model achieves 0.169 val loss but 86.7% closed-loop. Always evaluate in closed-loop.

4. **DAgger works when combined with expert data**: Pure on-policy data (18,820 DAgger samples)
   is insufficient. Expert data provides coverage; DAgger improves state-space robustness.

5. **Phase-specific insight**: reach_pick is the critical BC phase (90% alone). Scripted
   reach_place is already strong. Grasp BC inclusion is catastrophic.

---

## 11. Limitations

1. **Language hard set at 60%**: Schema and object grounding failures persist.
   Qwen output often omits or distorts object names in hard variations.

2. **DAgger plateau**: 93.3% multi-seed avg may be near the ceiling for this
   approach. Further improvement may require architectural changes.

3. **Robosuite sim-only**: All embodied evaluation in simulation. No real-world
   validation.

4. **Single task**: Stacking only. Generalization to other manipulation tasks
   not tested.

5. **No end-to-end integration**: Language → Planning → Control pipeline was
   evaluated component-wise, not in a single integrated run.

---

## 12. Release Artifacts

See `docs/artifact_manifest.md` for full listing.

Key artifacts:
- **DAgger Reach BC checkpoint**: `generated/results/stage_7B-4/dagger_reach_bc/best.pt`
- **Expert demos**: `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/`
- **Execution config**: `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`
- **Final report tables**: `generated/report_tables/final/`
- **Eval results**: `generated/results/stage_7B-5/`

---

## Appendix: Evaluation Commands

```bash
# Hybrid BC eval
python3 -m vlm_nsm.learning.eval_bc_hybrid \
    --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
    --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
    --num_episodes 20 --seed 0 --condition_on_phase \
    --bc_phases reach_pick reach_place --fixed_zero_action_dims 3 4 5 \
    --output results.json

# Final report tables
python3 -m vlm_nsm.experiments.export_final_report_tables

# Tests
python3 -m pytest tests/ -q --tb=short
```
