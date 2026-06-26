# Hybrid Reach-Only BC: Multi-Seed Evaluation

**Config:** `--control_mode hybrid --bc_phases reach_pick reach_place --condition_on_phase`
**Model:** phase-conditioned MLP (17->256->256->7), 72K params
**Training data:** expert v2 strict_only (80 train / 20 val episodes)
**Action norm:** min_action_std=1e-3, masked MSE loss for dims 3-5

## Per-Seed Results (20 episodes each)

| Seed | Success Rate | Mean Steps | BC Steps | Scripted Steps |
|------|:-----------:|:----------:|:--------:|:--------------:|
| 0    |    95%      |   985.5    |  585.7   |    415.2       |
| 1    |    90%      |   983.5    |  583.2   |    417.3       |
| 2    |   100%      |  1000.0    |  609.4   |    391.6       |

## Summary Statistics

| Metric      | Mean ± Std      | Min  | Max  |
|-------------|:--------------:|:----:|:----:|
| Success     | 95.0 ± 5.0%    | 90%  | 100% |
| Steps       | 989.7 ± 9.4    | 983.5| 1000.0|
| BC Steps    | 592.8 ± 14.1   | 583.2| 609.4|
| Script Steps| 408.1 ± 13.9   | 391.6| 417.3|

**Key Result:** Hybrid reach-only BC achieves **95% average success rate**
across 3 seeds (σ=5%), validating robustness of the phase-filtered approach.

All seeds use identical training config except random seed for train/val split.
Scripted phases (grasp, lift, release, retreat) use proportional control with
kp=0.08, max_delta=0.015.
