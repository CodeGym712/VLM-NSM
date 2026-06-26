# Robosuite Ablation Study Comparison

| Config                                              | N  | Strict % | Semantic % | Mean Steps | P95 Steps | Strict RG | Semantic RG | Timeout After | Timeout Before | Failures                         |
| --------------------------------------------------- | -- | -------- | ---------- | ---------- | --------- | --------- | ----------- | ------------- | -------------- | -------------------------------- |
| Baseline (all flags enabled)                        | 20 | 90%      | 90%        | 989        | 1000      | 10.0      | 10.0        | 0             | 1              | horizon=2                        |
| Ablation: --disable-place-state-machine             | 20 | 95%      | 95%        | 960        | 1000      | 5.0       | 5.0         | 0             | 1              | horizon=1                        |
| Ablation: --no-early-stop                           | 20 | 0%       | 95%        | 989        | 1000      | 100.0     | 5.0         | 19            | 0              | horizon=1; timeout_after_goal=19 |
| Ablation: --disable-horizon-aware                   | 20 | 75%      | 75%        | 944        | 1000      | 25.0      | 25.0        | 0             | 0              | horizon=5                        |
| Stage 7A-11 Best (cs15, lift_early, release_minmax) | 50 | 92%      | 92%        | 787        | 815       | 8.0       | 8.0         | 0             | 2              | horizon=2; preplace=2            |

## Best Parameter Search Config

- kp=0.08, max_delta=0.015, early_stop=False
- Success: 100%
- Mean steps: 668

## Demo Collection

- Episodes: 50
- Strict success: 45 (90%)
- Failures: 5

## Step Budget Analysis

- Total: mean=948 p50=962 p95=1000 min=783 max=1000
- Grasp: mean=289
- Lift: mean=176
- Reach pick: mean=175
- Reach place: mean=171
- Release: mean=50
- Retreat: mean=152