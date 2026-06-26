# BC Stage 7B-2 Report: Phase-Conditioned and Hybrid Behavior Cloning

## Overview

Stage 7B-2 extends the Stage 7B-1 BC baseline with:
1. Action normalization stability (min_action_std floor)
2. Phase-conditioned BC (phase one-hot observation encoding)
3. Per-phase MSE diagnostics
4. Hybrid BC eval (BC for selected phases, scripted for others)

## Action Normalization Fix

### Problem
Action dims 3-5 (orientation) are always zero in the demo data because the
proportional controller keeps gripper orientation fixed. Computing
`action_std = np.std(action, axis=0)` produces near-zero values for these
dims. Normalizing with `(action - mean) / std` causes numerical instability
(division by ~0), which the neural network amplifies during training.

### Fix
```python
action_std = np.maximum(action_std_raw, min_action_std)  # default 1e-3
```

- Near-zero dims tracked in `near_zero_std_dims`
- Raw std preserved in `action_std_raw` for diagnostics
- CLI: `--min_action_std 1e-3` (default)

### Masked MSE Loss
To prevent the model from wasting capacity predicting always-zero dims:

```python
loss_fn = _create_masked_mse_loss(mask_dims=[3, 4, 5], action_dim=7)
```

Excludes specified dims from gradient computation. Used with
`--mask_near_zero_action_dims` flag.

## Phase Conditioning Architecture

### Observation Encoding
```
Raw obs (11-dim):  [eef_xyz (3), gripper (1), has_obj (1), cube1_xyz (3), cube2_xyz (3)]
Phase one-hot (6): [reach_pick, grasp, lift, reach_place, release, retreat]
                  ↓ concatenate
Input (17-dim):    [obs_11d | one_hot_6d]
```

Phase one-hot is appended AFTER obs retrieval but BEFORE normalization.

### Policy
```python
MLPBCPolicy(obs_dim=17, action_dim=7, hidden_dims=[256, 256])
```

## Training Results

### Phase-Conditioned BC (obs_dim=17)

Trained on 100-episode demo dataset (80 train / 20 val episodes, 69.8K / 17.4K steps).

| Parameter | Value |
|-----------|-------|
| Model | MLPBCPolicy(obs_dim=17, action_dim=7, hidden_dims=[256, 256]) |
| Parameters | 72,199 |
| Epochs | 150 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Training time | 770s (12.8 min) on RTX 3090 |
| Best val loss | 0.0477 |

### Per-Phase MSE

| Phase | Train MSE | Val MSE | Relative Difficulty |
|-------|-----------|---------|-------------------|
| reach_pick | 0.0124 | 0.0107 | Easy |
| grasp | 0.0286 | 0.0239 | Moderate |
| lift | **0.1109** | **0.1008** | **Hard (9x reach_pick)** |
| reach_place | 0.0218 | 0.0337 | Easy |
| release | 0.0305 | 0.0406 | Moderate |
| retreat | 0.0365 | 0.0360 | Moderate |
| **Overall** | **0.0401** | **0.0409** | |

Key findings:
- **Lift phase is the hardest** — MSE is 3-10x higher than other phases. The lift phase
  requires precise gripper control and height detection, which is harder to clone.
- **Reach_pick is the easiest** — the proportional controller's reach behavior is
  well-captured by the BC policy (MSE=0.012/0.011).
- **Val-train gap is small** — no significant overfitting. Val MSE is close to train MSE
  for all phases.

### Action Distribution Validation

| Action Dim | Dataset Mean ± Std | Policy Mean ± Std | Status |
|------------|-------------------|-------------------|--------|
| dim_0 (pos x) | 0.00 ± 1.00 | 0.02 ± 0.96 | ✅ |
| dim_1 (pos y) | 0.00 ± 1.00 | 0.01 ± 0.92 | ✅ |
| dim_2 (pos z) | 0.00 ± 1.00 | 0.03 ± 0.95 | ✅ |
| dim_3 (ori) | 0.00 ± 0.00 | -0.00 ± 0.00 | ✅ near-zero variance |
| dim_4 (ori) | 0.00 ± 0.00 | -0.00 ± 0.00 | ✅ near-zero variance |
| dim_5 (ori) | 0.00 ± 0.00 | -0.00 ± 0.00 | ✅ near-zero variance |
| dim_6 (gripper) | 0.00 ± 1.00 | 0.04 ± 0.99 | ✅ |

The action distribution fix correctly handles zero-variance dims (3-5). The policy
outputs near-zero values for orientation while matching dataset distribution for
position (0-2) and gripper (6) dims.

## Hybrid BC Eval

### Full Mode
```bash
python3 -m vlm_nsm.learning.eval_bc \
    --checkpoint best.pt \
    --control_mode full \
    --condition_on_phase
```
BC controls all 6 phases.

### Hybrid Mode
```bash
python3 -m vlm_nsm.learning.eval_bc \
    --checkpoint best.pt \
    --control_mode hybrid \
    --bc_phases reach_pick reach_place \
    --fixed_zero_action_dims 3 4 5 \
    --condition_on_phase
```
BC controls only `reach_pick` and `reach_place`; other phases use scripted
(zero action + open gripper).

## Diagnostics

### Per-Phase MSE
```bash
python3 -m vlm_nsm.learning.bc_diagnostics \
    --checkpoint best.pt \
    --demo_npz trajectories.npz \
    --condition_on_phase \
    --output_dir generated/report_tables/stage_7B-2
```

### Action Distribution
The same command produces action distribution comparison (dataset vs policy).

## Training Command (Run on GPU)
```bash
python3 -m vlm_nsm.learning.train_bc \
    --npz generated/demos/stage_7B-1/robosuite_stage7b1_trajectories.npz \
    --obs_key filtered_obs \
    --phase_key phase_id \
    --condition_on_phase \
    --epochs 150 --batch-size 256 --lr 1e-3 \
    --hidden_dims 256 256 --episode_split \
    --min_action_std 1e-3 --fixed_zero_action_dims 3 4 5 \
    --seed 0 --device cuda \
    --save generated/results/stage_7B-2/bc_phase_conditioned/best.pt
```

## Files Changed

| File | Change |
|------|--------|
| `src/vlm_nsm/learning/dataset.py` | action_std floor, phase one-hot, near_zero_std_dims |
| `src/vlm_nsm/learning/train_bc.py` | MaskedMSE loss, phase conditioning, CLI args |
| `src/vlm_nsm/learning/eval_bc.py` | Hybrid mode, phase conditioning (rewrite) |
| `src/vlm_nsm/learning/bc_diagnostics.py` | New — per-phase MSE, action distribution |
| `docs/robosuite_demo_dataset.md` | Updated with BC trajectory schema |
| `docs/robosuite_empirical_report.md` | Stage 7B-2 section |
| `tests/test_bc_std_floor.py` | New — action norm, masked MSE (8 tests) |
| `tests/test_phase_conditioned_dataset.py` | New — phase dataset (7 tests) |
| `tests/test_bc_hybrid_eval_config.py` | New — hybrid eval config (10 tests) |
