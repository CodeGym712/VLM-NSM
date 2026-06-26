# Robosuite Adapter Checklist

> Stage 7A-3: Preparation for connecting embodied pipeline to real Robosuite environment.
> Target: Replace MockEnvWrapper with RobosuiteEnvWrapper while keeping the rest of
> the EmbodiedSkillExecutor pipeline unchanged.

## 1. RobosuiteEnvWrapper API

Must implement the same interface as `BaseEnvWrapper`:

| Method | Returns | Purpose |
|--------|---------|---------|
| `reset()` | dict | Reset simulation, return initial obs dict |
| `get_object_pose(name)` | `{x, y, z}` dict | Get pose of a named object |
| `get_target_pose(name)` | `{x, y, z}` dict | Get target placement pose |
| `get_all_object_poses()` | `dict[name → {x, y, z}]` | Get all object poses |
| `get_objects_on(target_name)` | `list[str]` | Get objects stacked on target |

**Status:** ⬜ Not implemented in RobosuiteEnvWrapper

## 2. Object Pose Extraction

Robosuite provides poses via `env.sim.data.body_xpos` or `env.sim.data.site_xpos`.

Required mappings for stacking cubes:

| Object | Robosuite body/site name | Expected key in obs dict |
|--------|--------------------------|------------------------|
| Cube i | `cube{i}_body` or `cube{i}_joint1` | `cube{i}_pos`, `cube{i}_quat` |
| Table | `table_body` or `robot0_table` | `table_pos`, `table_quat` |
| Gripper | `robot0_gripper_center` or `gripper_site` | `gripper_pos`, `gripper_quat` |

> Robosuite naming depends on the specific environment (e.g., `Stack`, `NutAssembly`).
> For `Stack` environment: cubes are named `Cube`, `Cube2`, `Cube3`.

**Status:** ⬜ Need to verify body/site names for chosen environment

## 3. Cube Name Mapping

Current mock convention:
```
cube1, cube2, ..., cubeN  (ordered left-to-right on table)
```

Robosuite `Stack` environment convention (for `StackThree`):
```
Cube, Cube2, Cube3  (positioned at fixed locations)
```

Mapping needed:
```python
CUBE_NAME_MAP = {
    "cube1": "Cube",
    "cube2": "Cube2",
    "cube3": "Cube3",
}
```

The planner uses `cube1..cubeN` naming. The env wrapper must translate.

**Status:** ⬜ Mapping not yet implemented

## 4. OSC_POSE Action Format

Robosuite uses OSC_POSE (Operational Space Control) for end-effector control.

Action vector (7-dim for OSC_POSE):
```
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

| Index | Range | Description |
|-------|-------|-------------|
| 0 | [-0.1, 0.1] | Delta x (forward) |
| 1 | [-0.1, 0.1] | Delta y (left) |
| 2 | [-0.1, 0.1] | Delta z (up) |
| 3 | [-0.1, 0.1] | Delta roll |
| 4 | [-0.1, 0.1] | Delta pitch |
| 5 | [-0.1, 0.1] | Delta yaw |
| 6 | [-1, 1] | Gripper: -1 = open, 1 = close |

Our `action_dim=4` in the mock (dx, dy, dz, dgripper) maps to indices [0, 1, 2, 6].
The remaining orientation deltas [3, 4, 5] must be provided (typically 0).

**Status:** ⬜ ScriptedController needs Robosuite-compatible action mode

## 5. ActionGrounder Pose Convention

ActionGrounder.ground_move() outputs:

```python
{
    "object": "cube1",
    "source": "table",
    "target": "cube2",
    "object_pose": {"x": X, "y": Y, "z": Z},     # Current object position
    "target_pose": {"x": X, "y": Y, "z": Z},     # Target placement position
    "grasp_pose": {"x": X, "y": Y, "z": Z + 0.08},  # Pregrasp above object
    "place_pose": {"x": X, "y": Y, "z": Z + block_height},  # Place height
}
```

Must verify that Robosuite coordinate conventions match:
- Z-up world frame (Robosuite default: Z-up)
- Origin at table center
- Units in meters (Robosuite default)

**Status:** ⬜ Coordinate system not yet verified

## 6. TerminationFailureMonitor Thresholds

Current thresholds (designed for mock):

| Check | Threshold | Robosuite tuning notes |
|-------|-----------|----------------------|
| Grasp detection | obj_z - initial_z < 0.005 | May need adjustment based on actual lift height |
| Place failure | gripper_has_object after release | Works as-is |
| Timeout | 200 steps | Adjust based on Robosuite control frequency (e.g., 500 for slower motion) |

> Will need empirical calibration in Robosuite — the mock uses idealized dynamics.

**Status:** ⬜ Thresholds not yet calibrated for Robosuite

## 7. PhysicalPredicateExtractor Thresholds

Current thresholds:
```python
thresholds = {
    "xy_threshold": 0.02,       # 2 cm horizontal tolerance
    "z_min": -0.001,            # Allow small negative z
    "z_max": block_height * 1.5,  # Acceptable stack height range
}
```

These may be too strict for Robosuite physics. Recommended tuning:
- Start with 0.03-0.05 for xy_threshold (Robosuite placement error ~2-3 cm)
- Keep z_min at -0.001
- Increase z_max to block_height * 2.0 (allow settling variation)

**Status:** ⬜ Thresholds need Robosuite calibration

## 8. DemonstrationRecorder Dimensions

Current settings:

| Param | Mock value | Robosuite expected |
|-------|-----------|-------------------|
| obs_dim | 16 (filtered) | 16 (filtered, same filter logic) |
| action_dim | 4 (dx, dy, dz, dgripper) | 7 (OSC_POSE: 6 dof + gripper) |
| full_obs_dim | n_blocks * 7 + 15 | Same structure, verify dimension |

**Status:** ⬜ Action dimension differs for Robosuite (7 vs 4)

## 9. Replay Feasibility

ReplayController currently replays actions from collected npz demos.

Requirements for Robosuite replay:
1. npz contains action sequences with the same action_dim as the environment expects
2. Env is resettable to the same initial state for each replay episode
3. ReplayController correctly feeds actions to env.step()

> Note: If action_dim changes (4 → 7), old scripted demos cannot be replayed.
> A re-collection with Robosuite-compatible action space is required.

**Status:** ⬜ Replay not yet tested with Robosuite action format

## 10. Reality Gap Reporting Protocol

Reality Gap = (N_symbolic - N_physical) / N_symbolic × 100%

When transitioning from mock to Robosuite:

| Metric | Mock | Robosuite (expected) |
|--------|------|---------------------|
| step_success_rate | 1.0 | < 1.0 (physics failures) |
| skill_success_rate | 1.0 | < 1.0 |
| episode_success_rate | 1.0 | < 1.0 |
| predicate_match_rate | 1.0 | < 1.0 |
| reality_gap | 0.0 | > 0.0 |

The Reality Gap in Robosuite will be nonzero due to:
- Physics noise (release settling, friction)
- Grasp uncertainty (slippage, misalignment)
- Pose estimation error
- Control inaccuracy (OSC_POSE tracking error)

**Protocol:**
1. Run N episodes with ScriptedController in Robosuite
2. Record symbolic_success_count (ideal) and physical_success_count (observed)
3. Compute Reality Gap using `metrics.reality_gap()`
4. Compare against mock baseline (should be 0.0 in mock)

**Status:** ⬜ Robosuite Reality Gap not yet measured

---

## Implementation Order (Suggested)

1. [ ] **RobosuiteEnvWrapper** — Implement BaseEnvWrapper interface using robosuite Stack env
2. [ ] **Cube name mapping** — Map cube1..cubeN to Robosuite naming convention
3. [ ] **Action space** — Extend ScriptedController or create RobosuiteController with 7-dim OSC_POSE
4. [ ] **Observation extraction** — Verify object pose extraction from sim state
5. [ ] **Termination thresholds** — Calibrate grasp/place detection thresholds
6. [ ] **Predicate thresholds** — Calibrate xy_threshold and z_max for Robosuite
7. [ ] **Demo collection** — Run collect_stacking_demos with Robosuite backend
8. [ ] **Replay sanity** — Verify ReplayController works with Robosuite-collected demos
9. [ ] **Reality Gap** — Measure baseline Reality Gap in Robosuite
10. [ ] **Full pipeline** — End-to-end EmbodiedSkillExecutor with RobosuiteEnvWrapper

---

> Status indicators: ✅ Done, ⬜ Not started, 🔄 In progress
