# DAgger 100ep Failure Audit v3

**Stage 8 result**: `dagger_reach_final_100ep.json`
- **Total episodes**: 100
- **Success**: 80 (80.0%)
- **Failures**: 20 (20.0%)
- **Truly clean** (success + no failure_type): 8
- **Elapsed**: 894.5s

---
## Failure Type Summary

| Type | Count | % of Total | % of Failures | Mean Steps | Mean BC Steps |
|:---:|:----:|:--------:|:------------:|:--------:|:------------:|
| horizon | 7 | 7.0% | 35.0% | 706 | 128 |
| preplace | 82 | 82.0% | 410.0% | 734 | 518 |
| verify_failed | 3 | 3.0% | 15.0% | 990 | 777 |

---
## Per-Episode Failure Detail

| Ep | Failure Type | Steps | BC Steps | Scripted | Likely Cause |
|:--:|:-----------:|:-----:|:--------:|:--------:|:-------------|
| 17 | horizon | 697 | 119 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 25 | preplace | 728 | 516 | 212 | Preplace verification never satisfied during place SM |
| 31 | preplace | 752 | 533 | 219 | Preplace verification never satisfied during place SM |
| 32 | preplace | 742 | 523 | 219 | Preplace verification never satisfied during place SM |
| 33 | horizon | 721 | 143 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 35 | verify_failed | 1000 | 787 | 213 | Place SM dropped block off-target |
| 45 | preplace | 783 | 489 | 294 | Preplace verification never satisfied during place SM |
| 46 | verify_failed | 1000 | 786 | 214 | Place SM dropped block off-target |
| 47 | preplace | 1000 | 785 | 215 | Preplace verification never satisfied during place SM |
| 53 | preplace | 1000 | 787 | 213 | Preplace verification never satisfied during place SM |
| 58 | preplace | 711 | 495 | 216 | Preplace verification never satisfied during place SM |
| 61 | horizon | 695 | 117 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 68 | horizon | 697 | 119 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 72 | horizon | 686 | 108 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 75 | preplace | 751 | 532 | 219 | Preplace verification never satisfied during place SM |
| 81 | preplace | 724 | 509 | 215 | Preplace verification never satisfied during place SM |
| 82 | horizon | 754 | 176 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 89 | verify_failed | 971 | 757 | 214 | Place SM dropped block off-target |
| 92 | horizon | 695 | 117 | 578 | Policy collapse in reach_pick (early-HZ, low BC steps) |
| 95 | preplace | 1000 | 783 | 217 | Preplace verification never satisfied during place SM |


---
## Root Cause Analysis

### Hypothesis 1: Policy collapse in reach_pick (early-horizon, bc_steps < 200)

Failures with `horizon` type and very low BC steps indicate the policy outputs near-zero or invalid actions early, causing the episode to stall in reach_pick phase. If the block is never approached, the state machine cannot advance.

### Hypothesis 2: reach_place divergence (horizon, bc_steps > 400)

Failures where bc_steps are high (>400) but still hit horizon suggest the BC policy oscillates or diverges near the target during the place phase, preventing the place state machine from converging.

### Hypothesis 3: Preplace verification failure (preplace type)

Even in successful episodes, `failure_type: preplace` is common — it means the preplace heuristic triggered at least once before eventual success. In failed episodes, the place SM simply could not converge.

### Hypothesis 4: verify_failed (block dropped off-target)

The block was released but the final predicate check fails. This can happen when release happens while the gripper is still over the target but the block bounces off, or when release position is imprecise.

---
## Comparison: Stage 7B-5 (93.3%) vs Stage 8 (80.0%)

| Metric | Stage 7B-5 (3×20ep) | Stage 8 (100ep) | Delta |
|:------|:------------------:|:--------------:|:----:|
| Episodes | 60 | 100 | — |
| Success rate | 93.3% | 80.0% | -13.3% |
| Mean steps | ~720 | 752.3 | — |
| Mean BC steps | ~520 | 510.9 | — |

The 13.3pp drop suggests either:
1. **Seed variance**: seed=0 underperforms the multi-seed average
2. **Small-sample bias**: 20ep runs happen to land on easier env initializations
3. **Policy degradation**: Large-N rollouts accumulate distributional shift

To distinguish: DAgger canonical 3×50ep (S9-C) at different seeds will show whether seed=0 is unlucky or the policy degrades at scale.