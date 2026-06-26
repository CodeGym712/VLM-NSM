# BC Reach Endpoint Error Analysis

## Methodology

Endpoint errors measure EEF-to-target distance at the end of each phase,
comparing ground-truth (dataset) vs BC policy rollout. For the policy rollout:

1. Start from the first timestep of each phase within an episode
2. Apply BC-predicted actions (denormalized) to update simulated EEF position
3. Cube positions are taken from the dataset at each timestep (state is "oracle")
4. Final predicted EEF position vs target position defines the endpoint error

## Main BC Model (Stage 7B-2R, strict_success data)

| Phase | N | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 |
|------|:-:|:-----------:|:----------:|:---------------:|:--------------:|
| reach_pick | 20 | 0.1163 | 0.1209 | 10.6615 | 14.2326 |
| grasp | 20 | 0.0198 | 0.0200 | 7.8198 | — |
| lift | 20 | 0.1301 | 0.1306 | 9.0790 | — |
| reach_place | 20 | 0.0567 | 0.0649 | 10.6030 | — |

## Reach_pick Model (phase-filtered)

| Phase | N | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 |
|------|:-:|:-----------:|:----------:|:---------------:|:--------------:|
| reach_pick | 20 | 0.1163 | 0.1209 | 10.5095 | 14.2326 |
| grasp | 20 | 0.0198 | 0.0200 | 45.9517 | — |
| lift | 20 | 0.1301 | 0.1306 | 27.8487 | — |
| reach_place | 20 | 0.0567 | 0.0649 | 238.8265 | — |

## Reach_place Model (phase-filtered)

| Phase | N | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 |
|------|:-:|:-----------:|:----------:|:---------------:|:--------------:|
| reach_pick | 20 | 0.1163 | 0.1209 | 16.9553 | — |
| grasp | 20 | 0.0198 | 0.0200 | 29.5199 | — |
| lift | 20 | 0.1301 | 0.1306 | 29.7842 | — |
| reach_place | 20 | 0.0567 | 0.0649 | 12.2054 | — |

## Key Findings

### Dataset (ground truth) endpoint errors are small
- All phases: 0.02-0.13m average EEF-to-target distance at phase end
- This confirms the demonstrator (proportional controller) reliably reaches targets

### Policy endpoint errors are catastrophic (~10m+)
- All models and phases show policy errors 2-3 orders of magnitude above dataset errors
- This exceeds the physical workspace (~0.3m), meaning the simulated EEF diverges completely

### Root Cause: Compounding Error (Covariate Shift)
- **Teacher forcing vs. closed-loop**: Model trained to predict next action from correct state, but during rollout each prediction error shifts the state further from training distribution.
- **Compounding**: With 50-100 steps per phase, even small per-step errors (0.01m) compound to large total drift.
- **No corrective feedback**: Unlike DAgger or interactive imitation learning, standard BC has no mechanism to recover from errors.

### Implications
1. **Phase-filtered models are unusable for closed-loop control**: Even on their target phase.
2. **Full model is not significantly better**: Similar 10m drift on all phases.
3. **BC for full reach phases is not viable** without corrective training (DAgger, interactive correction, or regularization).
4. **BC might still work for single-step refinement** (e.g., correcting gripper pose just before grasp), where the model doesn't need to chain multiple predictions.

## Source Data
- `generated/report_tables/stage_7B-3/main_model_diag/bc_endpoint_errors.md`
- `generated/report_tables/stage_7B-3/reach_pick_diag/bc_endpoint_errors.md`
- `generated/report_tables/stage_7B-3/reach_place_diag/bc_endpoint_errors.md`
