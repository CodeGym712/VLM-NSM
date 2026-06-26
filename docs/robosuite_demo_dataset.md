# Robosuite Demo Dataset

## Overview

This document describes the robosuite demonstration dataset collected for
downstream Behavior Cloning (BC) training. Demos are collected using the
proportional control embodied pipeline with the best execution config.

## Collection Pipeline

Demos are collected by running the embodied stacking pipeline
(`collect_robosuite_demos.py`) in the robosuite 1.5.2 Stack environment:

1. Planner generates symbolic plan for 2-cube stacking
2. Embodied executor runs the plan with proportional control
3. Per-episode state snapshots are captured (initial + final)
4. Episodes are categorized by success type and saved as npz files

### Usage

```bash
# Best config (recommended)
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.collect_robosuite_demos \
    --num_episodes 50 --output_dir generated/demos/robosuite_best \
    --config generated/configs/robosuite_best_execution_config_stage7a9.json

# Collect until 100 strict successes
CUDA_VISIBLE_DEVICES=0,1,2,3 MUJOCO_GL=egl \
  python3 -m vlm_nsm.experiments.collect_robosuite_demos \
    --collect-until-strict 100 \
    --output_dir generated/demos/robosuite_100strict \
    --config generated/configs/robosuite_best_execution_config_stage7a9.json

# Record full trajectories for BC training
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.collect_robosuite_demos \
    --num_episodes 50 --output_dir generated/demos/robosuite_best \
    --record
```

## Directory Structure

```
generated/demos/robosuite_best/
├── strict_success/          # Clean execution + goal achieved, no timeout
│   ├── episode_0001.npz
│   ├── episode_0002.npz
│   └── ...
├── semantic_only_success/   # Goal achieved but execution not clean
│   └── ...
├── all_semantic_success/    # Union of strict_success + semantic_only_success
│   └── ...
├── failure_debug/           # Goal not achieved
│   └── ...
└── collection_summary.json   # Metadata and per-episode results
```

### Category Definitions

| Category | Criterion | Use Case |
|----------|-----------|----------|
| `strict_success` | episode_success=1 AND goal_ok=1 AND timeout_type=None | High-quality training data |
| `semantic_only_success` | goal_ok=1 but timeout_type is not None or execution had errors | Additional training data |
| `all_semantic_success` | All episodes where goal was achieved (strict + semantic_only) | Training with max coverage |
| `failure_debug` | Goal NOT achieved | Debugging failure modes |

## Per-Episode npz Schema

Each npz file in `strict_success/`, `semantic_only_success/`, etc. contains
per-episode initial and final state snapshots:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `initial_object_poses` | (N, 3) | float32 | Object positions [x, y, z] per cube |
| `initial_eef` | (3,) | float32 | End-effector position [x, y, z] |
| `initial_gripper_open` | (1,) | float32 | Gripper open state [-1 or 1] |
| `final_object_poses` | (N, 3) | float32 | Object positions at episode end |
| `final_eef` | (3,) | float32 | End-effector position at episode end |
| `final_gripper_open` | (1,) | float32 | Gripper state at episode end |
| `strict_success` | (1,) | bool | Strict physical success flag |
| `semantic_success` | (1,) | bool | Semantic goal success flag |
| `failure_type` | (8,) | uint8 | Failure type (padded to 8 chars) |
| `num_steps` | (1,) | int32 | Total steps taken |

N = number of cubes (2 for Stack task).

## Full Trajectory npz Schema (BC Training)

The single-file format produced by `DemonstrationRecorder` (used with `--record`
flag) contains per-timestep data for all episodes:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `obs` | (T, obs_dim) | float32 | Raw observation vectors (11-dim) |
| `filtered_obs` | (T, 11) | float32 | Filtered observation vectors |
| `action` | (T, 7) | float32 | Action vectors |
| `done` | (T,) | bool | Per-step done flag |
| `episode_id` | (T,) | int32 | Episode index per timestep |
| `step_id` | (T,) | int32 | Global step counter across all episodes |
| `success` | (T,) | bool | Per-step success flag |
| `phase_id` | (T,) | int64 | Phase index (0-5) for phase-conditioned BC |
| `skill_id` | (T,) | int32 | Skill index (deprecated, use phase_id) |

### Observation Vector (11-dim)

| Index | Component | Description |
|-------|-----------|-------------|
| 0-2 | EEF xyz | End-effector position |
| 3 | gripper | Gripper openness (-1 open, 1 closed) |
| 4 | has_object | Binary: whether gripper holds an object |
| 5-7 | cube1 xyz | Cube 1 position |
| 8-10 | cube2 xyz | Cube 2 position |

### Action Vector (7-dim)

| Index | Component | Description |
|-------|-----------|-------------|
| 0-2 | delta xyz | Position delta |
| 3-5 | delta orientation | Orientation delta (always zero in demo data) |
| 6 | gripper | Gripper command (-1 open, 1 close) |

Note: Dims 3-5 (orientation) are always zero because the proportional
controller keeps gripper orientation fixed. These dims are excluded from
MSE loss via `--fixed_zero_action_dims 3 4 5` and `--mask_near_zero_action_dims`.

### Phase IDs

| ID | Name | Description |
|----|------|-------------|
| 0 | reach_pick | Move EEF to pre-grasp position |
| 1 | grasp | Close gripper on cube |
| 2 | lift | Lift cube to clearance height |
| 3 | reach_place | Move EEF to place position |
| 4 | release | Open gripper to release cube |
| 5 | retreat | Move EEF away after placement |

## collection_summary.json Schema

```json
{
  "output_dir": "str",
  "num_episodes": 50,
  "collect_until_strict": null,
  "total_attempts": 50,
  "strict_success_count": 45,
  "semantic_only_success_count": 3,
  "all_semantic_success_count": 48,
  "failure_count": 2,
  "strict_success_rate": 0.9,
  "semantic_success_rate": 0.96,
  "efficiency_ratio": null,
  "per_episode": [
    {
      "episode": 1,
      "strict_success": true,
      "semantic_success": true,
      "failure_type": "success",
      "num_steps": 650,
      "timeout_type": null,
      "early_stopped": true,
      "initial_cube1_z": 0.0,
      "initial_cube2_z": 0.0,
      "final_cube1_z": 0.05,
      "final_cube2_z": 0.10
    }
  ]
}
```

When `--collect-until-strict N` is used, `total_attempts` reports the actual
number of episodes run, and `efficiency_ratio` reports strict/total.

## meta.json Schema (BC Trajectories)

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | Task name (e.g., "stacking") |
| `n_blocks` | int | Number of blocks/cubes |
| `env_type` | str | Environment type ("robosuite") |
| `controller_type` | str | Controller type ("proportional") |
| `num_episodes` | int | Total episodes |
| `num_success` | int | Total successful steps |
| `success_rate` | float | Fraction of successful steps |
| `obs_dim` | int | Observation dimension (11) |
| `action_dim` | int | Action dimension (7) |
| `step_names` | list[str] | Phase names in order |
| `observation_type` | str | Observation type ("raw") |
| `created_at` | str | ISO timestamp |

## Best Config (Stage 7A-9)

The recommended execution config for demo collection:

- **kp**: 0.08 (faster convergence)
- **max_delta**: 0.015 (smoother motion)
- **reach_threshold**: 0.02 (standard convergence)
- **grasp_close_steps**: 50 (reliable grasping)
- **release_open_steps**: 50 (reliable release)
- **early_stop_on_goal**: true (stop when goal achieved)

## Dataset Usage

### Per-episode npz files

```python
import numpy as np

data = np.load("episode_0001.npz")
initial_obs = np.concatenate([
    data["initial_object_poses"].flatten(),
    data["initial_eef"],
    data["initial_gripper_open"],
])
success = bool(data["strict_success"])
```

### Full trajectory npz (BC training)

```python
from vlm_nsm.learning.dataset import RobosuiteDemoDataset

train_ds, val_ds = RobosuiteDemoDataset.train_val(
    "robosuite_stage7b1_trajectories.npz",
    val_ratio=0.2, seed=42,
    obs_key="filtered_obs",
    condition_on_phase=True,
    phase_key="phase_id",
    min_action_std=1e-3,
)
```

## Expert v2 Dataset (Stage 7A-14)

**Path:** `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/`

| Item | Value |
|:---|---:|
| Collection method | `collect_until_strict` |
| Target strict demos | 100 |
| Attempts | 107 |
| Strict success | 100 (93.5%) |
| Total steps | 85,396 |
| True phase_id | Yes (recorded natively, not synthetic) |

### Subdirectories

| Directory | Contents | Usage |
|:---|---|:---|
| `strict_only/` | 100 strict-only per-episode npz files | **Recommended for training** |
| `strict_success/` | 107 npz (100 strict + 7 semantic-only) | Analysis only |
| `failure_debug/` | 24 npz | Debugging |

### Merged trajectory file

**Path:** `trajectories_strict_only.npz`
- 85,396 steps, 100 episodes
- Keys: `filtered_obs`, `action`, `phase_id`, `episode_id`, `step_id`, `skill_id`, `done`, `success`
- obs_dim=11, action_dim=7
- True phase_id available (values 0-3 for Stack task)

### Recommended BC training command

```bash
python3 -m vlm_nsm.learning.train_bc \
  --demo_dir generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only \
  --obs_key filtered_obs \
  --condition_on_phase --phase_key phase_id \
  --epochs 200 --batch-size 64 --lr 1e-3 \
  --hidden_dims 256 256 \
  --mask_near_zero_action_dims \
  --seed 42 --device cuda \
  --save generated/models/stage_7B-2R/bc_phase_conditioned.pt
```
