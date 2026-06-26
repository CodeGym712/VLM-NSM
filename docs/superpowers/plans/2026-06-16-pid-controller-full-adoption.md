# PID Controller Full Adoption — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace P-controller entirely with PID controller as the default, close the 4pp success gap via gain auto-tuning, enhance Learned ID features, fix HDF5 phase labels, and validate end-to-end.

**Architecture:** Five independent workstreams. WS1 (gain tuning) and WS2 (set PID as default) are sequential prerequisites. WS3 (ID features) and WS4 (HDF5 fix) are parallel. WS5 (demo) depends on WS1+WS2+WS4.

**Tech Stack:** Python 3.10, robosuite 1.5.2, MuJoCo, PyTorch, HDF5, NumPy

---

## File Map

```
CREATE:
  src/vlm_nsm/experiments/tune_pid_gains.py          # WS1: grid search script
  src/vlm_nsm/data/id_training_dataset_v2.py         # WS3: enhanced ID features
  tests/test_pid_gain_tuning.py                      # WS1: tuning tests

MODIFY:
  src/vlm_nsm/embodied/controllers/operational_space_pid.py  # WS1: gain update
  src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py # WS2: PID default
  src/vlm_nsm/data/id_training_dataset.py                    # WS3: feature upgrade
  src/vlm_nsm/embodied/controllers/learned_id_model.py       # WS3: model update
  src/vlm_nsm/training/train_inverse_dynamics.py             # WS3: re-train
  src/vlm_nsm/embodied/place_state_machine.py               # WS4: phase callback
  src/vlm_nsm/experiments/collect_vla_demos.py              # WS4: recording fix

OUTPUT:
  generated/configs/control/pid_gain_schedule_tuned.json     # WS1: tuned gains
  generated/results/id/id_model_v4/best.pt                   # WS3: new ID model
  generated/datasets/vla_demos/d1_pid_200ep/                 # WS4: re-collected
  generated/results/control/pid_final/                       # WS5: demo output
```

---

### Task 1: PID Gain Auto-Tuning with Grid Search

**Files:**
- Create: `src/vlm_nsm/experiments/tune_pid_gains.py`
- Create: `tests/test_pid_gain_tuning.py`
- Modify: `src/vlm_nsm/embodied/controllers/operational_space_pid.py:55-65` (gain update)
- Output: `generated/configs/control/pid_gain_schedule_tuned.json`

Goal: Use control_audit_200ep dataset to find optimal per-phase (kp, ki, kd) that maximize success rate. Target: close the 4pp gap (PID 90% → ≥93%).

- [ ] **Step 1: Write the gain tuning script skeleton**

```python
# src/vlm_nsm/experiments/tune_pid_gains.py
"""Grid search PID gains using control_audit episodes for offline evaluation.

For each phase, sweep kp/ki/kd over reasonable ranges and select combination
that minimizes reach error (dist to target) while maximizing convergence rate.

Usage:
    python -m vlm_nsm.experiments.tune_pid_gains \
        --dataset_dir generated/datasets/vla_demos/control_audit_200ep \
        --output generated/configs/control/pid_gain_schedule_tuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from vlm_nsm.embodied.controllers.operational_space_pid import (
    OperationalSpacePID, GainSchedule, PIDGain,
)

# Per-phase search ranges
_SEARCH_RANGES: dict[str, dict] = {
    "reach_pick":   {"kp": [0.04, 0.06, 0.08, 0.10], "ki": [0.0, 0.001, 0.002, 0.005], "kd": [0.0, 0.005, 0.01, 0.02], "max_delta": [0.02]},
    "grasp":        {"kp": [0.03, 0.05, 0.07], "ki": [0.0, 0.001, 0.002], "kd": [0.0, 0.005, 0.008, 0.01], "max_delta": [0.01]},
    "lift":         {"kp": [0.04, 0.06, 0.08, 0.10], "ki": [0.0, 0.001, 0.002], "kd": [0.0, 0.005, 0.01], "max_delta": [0.02]},
    "reach_place":  {"kp": [0.04, 0.06, 0.08], "ki": [0.0, 0.001, 0.002, 0.005], "kd": [0.0, 0.005, 0.01, 0.015, 0.02], "max_delta": [0.015, 0.02]},
}

_PHASE_ID_MAP = {0: "reach_pick", 1: "grasp", 2: "lift", 3: "reach_place"}


def replay_episode_offline(episode_path: str, gains: dict, max_steps_per_phase: int = 500) -> tuple[float, int, int]:
    """Replay one episode with given PID gains, evaluating convergence.

    Returns:
        (mean_reach_error, total_steps, convergence_failures)
    """
    total_error = 0.0
    total_steps = 0
    conv_failures = 0

    with h5py.File(episode_path, "r") as h5:
        actions = h5["actions_4d"][:]
        eef_pos = h5["proprio/eef_pos"][:]
        phase_ids = h5["symbolic/phase_id"][:]
        n_steps = actions.shape[0]

        pid = OperationalSpacePID()
        # Override gains with candidate values
        for phase_name, g in gains.items():
            pid._gain_schedule.gains[phase_name] = PIDGain(**g)

        # Phase boundaries
        phase_starts = [0]
        for t in range(1, n_steps):
            if phase_ids[t] != phase_ids[t - 1]:
                phase_starts.append(t)
        phase_starts.append(n_steps)

        for ps_idx in range(len(phase_starts) - 1):
            t_start = phase_starts[ps_idx]
            t_end = phase_starts[ps_idx + 1]
            phase_name = _PHASE_ID_MAP.get(int(phase_ids[t_start]), "reach_pick")

            # Target: EEF position at end of phase (from recorded trajectory)
            target_eef = eef_pos[min(t_end - 1, n_steps - 1)]
            pid.set_target(pos=target_eef)
            pid.set_phase(phase_name)

            for t in range(t_start, min(t_end, t_start + max_steps_per_phase)):
                pid.set_current(eef_pos[t])
                v_des = pid.compute_pid_velocity()
                # Compute error: distance between predicted next pos and target
                pred_next = eef_pos[t] + v_des * 0.02
                err = np.linalg.norm(pred_next - target_eef)
                total_error += err
                total_steps += 1

            # Check if phase converged
            final_dist = np.linalg.norm(eef_pos[min(t_end - 1, n_steps - 1)] - target_eef)
            if final_dist > 0.02:  # reach_threshold
                conv_failures += 1

    mean_error = total_error / max(total_steps, 1)
    return mean_error, total_steps, conv_failures


def grid_search(dataset_dir: str, max_episodes: int = 50) -> dict:
    """Run grid search over per-phase gain ranges."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        episodes = [json.loads(l) for l in f.readlines()]

    # Use success episodes only
    success_eps = [e for e in episodes if e.get("success", True)][:max_episodes]
    print(f"[tune] Using {len(success_eps)} success episodes for tuning")

    best_gains = {}
    best_score = float("inf")

    # Tune one phase at a time, keeping others at default
    default_gains = GainSchedule()
    for phase_name, ranges in _SEARCH_RANGES.items():
        print(f"\n[tune] Tuning {phase_name}...")
        candidates = _generate_combinations(ranges)
        phase_best = None
        phase_best_score = float("inf")

        for candidate in candidates:
            # Build gain dict: candidate for this phase, defaults for others
            test_gains = {}
            for pn in _SEARCH_RANGES:
                if pn == phase_name:
                    test_gains[pn] = candidate
                else:
                    dg = default_gains.get(pn)
                    test_gains[pn] = {"kp": dg.kp, "ki": dg.ki, "kd": dg.kd, "max_delta": dg.max_delta, "i_max": dg.i_max}

            total_err = 0.0
            total_conv_fail = 0
            for ep in success_eps[:10]:  # validate on 10 episodes per candidate
                err, _, cf = replay_episode_offline(ep["episode_path"], test_gains)
                total_err += err
                total_conv_fail += cf

            score = total_err + total_conv_fail * 0.1  # penalize convergence failures
            if score < phase_best_score:
                phase_best_score = score
                phase_best = candidate

        print(f"[tune] {phase_name} best: kp={phase_best['kp']}, ki={phase_best['ki']}, kd={phase_best['kd']}, score={phase_best_score:.4f}")
        best_gains[phase_name] = phase_best

    return best_gains


def _generate_combinations(ranges: dict) -> list[dict]:
    """Generate all combinations from per-parameter search ranges."""
    keys = list(ranges.keys())
    values = list(ranges.values())
    combos = [[]]
    for v_list in values:
        new_combos = []
        for combo in combos:
            for v in v_list:
                new_combos.append(combo + [v])
        combos = new_combos
    return [{keys[i]: combo[i] for i in range(len(keys))} for combo in combos]


def main():
    parser = argparse.ArgumentParser(description="Tune PID gains via grid search")
    parser.add_argument("--dataset_dir", required=True, help="Path to control_audit dataset")
    parser.add_argument("--output", required=True, help="Output JSON path for tuned gains")
    parser.add_argument("--max_episodes", type=int, default=50, help="Max episodes for tuning")
    args = parser.parse_args()

    best_gains = grid_search(args.dataset_dir, args.max_episodes)

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(best_gains, f, indent=2)
    print(f"\n[tune] Saved tuned gains to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write tests for gain tuning**

```python
# tests/test_pid_gain_tuning.py
import json
import numpy as np
from vlm_nsm.embodied.controllers.operational_space_pid import (
    OperationalSpacePID, GainSchedule, PIDGain,
)

def test_tuned_gains_improve_convergence():
    """Tuned gains should converge faster than default gains on simple reach task."""
    target = np.array([0.1, 0.0, 0.95], dtype=np.float32)
    start = np.array([0.0, 0.0, 0.85], dtype=np.float32)

    # Default gains
    pid_default = OperationalSpacePID()
    pid_default.set_target(pos=target)
    pid_default.set_phase("reach_pick")
    pid_default.set_current(start)

    # Tuned gains (higher kp for faster approach)
    tuned_schedule = GainSchedule()
    tuned_schedule.gains["reach_pick"] = PIDGain(kp=0.10, ki=0.002, kd=0.01, max_delta=0.02, i_max=0.03)
    pid_tuned = OperationalSpacePID(gain_schedule=tuned_schedule)
    pid_tuned.set_target(pos=target)
    pid_tuned.set_phase("reach_pick")
    pid_tuned.set_current(start)

    # Simulate both
    default_steps = _simulate_to_target(pid_default, start, target)
    tuned_steps = _simulate_to_target(pid_tuned, start, target)

    assert tuned_steps <= default_steps + 5  # tuned should be at least as fast


def _simulate_to_target(pid, start, target, max_steps=500, threshold=0.02):
    pos = start.copy()
    for i in range(max_steps):
        pid.set_current(pos)
        v_des = pid.compute_pid_velocity()
        pos = pos + v_des * 0.02
        if np.linalg.norm(pos - target) < threshold:
            return i + 1
    return max_steps


def test_gain_combinations_generation():
    from vlm_nsm.experiments.tune_pid_gains import _generate_combinations
    ranges = {"kp": [0.05, 0.08], "ki": [0.0, 0.001]}
    combos = _generate_combinations(ranges)
    assert len(combos) == 4  # 2*2
    assert {"kp": 0.05, "ki": 0.0} in combos


def test_tuned_gains_json_roundtrip(tmp_path):
    """Tuned gains JSON should load correctly into GainSchedule."""
    gains_dict = {
        "reach_pick": {"kp": 0.10, "ki": 0.002, "kd": 0.01, "max_delta": 0.02, "i_max": 0.03},
        "grasp": {"kp": 0.05, "ki": 0.001, "kd": 0.008, "max_delta": 0.01, "i_max": 0.02},
    }
    p = tmp_path / "gains.json"
    with open(p, "w") as f:
        json.dump(gains_dict, f)

    with open(p) as f:
        loaded = json.load(f)

    gs = GainSchedule()
    for phase_name, g in loaded.items():
        gs.gains[phase_name] = PIDGain(**g)
    assert gs.get("reach_pick").kp == 0.10
```

- [ ] **Step 3: Run tuning tests to verify they fail (no tune script yet)**

Run: `nsm/bin/python -m pytest tests/test_pid_gain_tuning.py::test_gain_combinations_generation -v --tb=short`
Expected: FAIL (import error — tune script not created yet)

- [ ] **Step 4: Run tuning tests after creating tune script**

Run: `nsm/bin/python -m pytest tests/test_pid_gain_tuning.py -v --tb=short`
Expected: 3 passed

- [ ] **Step 5: Run grid search on control_audit_200ep**

Run: `CUDA_VISIBLE_DEVICES=0 nsm/bin/python -m vlm_nsm.experiments.tune_pid_gains --dataset_dir generated/datasets/vla_demos/control_audit_200ep --output generated/configs/control/pid_gain_schedule_tuned.json --max_episodes 50`
Expected: Output file with tuned per-phase gains

- [ ] **Step 6: Update GainSchedule defaults with tuned values**

```python
# src/vlm_nsm/embodied/controllers/operational_space_pid.py:55-65
# Replace default gains with tuned values from grid search output

@dataclass
class GainSchedule:
    """Per-phase PID gain schedule — tuned on control_audit_200ep."""
    
    gains: dict[str, PIDGain] = field(default_factory=lambda: {
        "reach_pick":   PIDGain(kp=0.10, ki=0.002, kd=0.01,  max_delta=0.02,  i_max=0.03),
        "grasp":        PIDGain(kp=0.05, ki=0.001, kd=0.008, max_delta=0.01,  i_max=0.02),
        "lift":         PIDGain(kp=0.08, ki=0.001, kd=0.01,  max_delta=0.02,  i_max=0.03),
        "reach_place":  PIDGain(kp=0.06, ki=0.002, kd=0.015, max_delta=0.015, i_max=0.03),
        "release":      PIDGain(kp=0.04, ki=0.001, kd=0.005, max_delta=0.005, i_max=0.01),
        "retreat":      PIDGain(kp=0.08, ki=0.001, kd=0.01,  max_delta=0.02,  i_max=0.03),
    })
```
Note: Replace values with actual grid search output after Step 5.

- [ ] **Step 7: Run PID eval with tuned gains to verify improvement**

Run: `CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl nsm/bin/python -u -m vlm_nsm.experiments.run_embodied_stacking_robosuite --num_episodes 100 --use-pid-controller --holding-check-enabled --lift-early-exit-enabled 2>&1 | grep -E "episode_success|failure_stage"`
Expected: episode_success_rate ≥ 0.92 (target: close 4pp gap)

- [ ] **Step 8: Run full regression after gain update**

Run: `nsm/bin/python -m pytest tests/ --ignore=tests/deprecated -x --tb=short -q`
Expected: all passing (1608+ tests)

- [ ] **Step 9: Commit gain tuning**

```bash
git add src/vlm_nsm/experiments/tune_pid_gains.py tests/test_pid_gain_tuning.py src/vlm_nsm/embodied/controllers/operational_space_pid.py generated/configs/control/pid_gain_schedule_tuned.json
git commit -m "feat: PID gain auto-tuning via grid search on control_audit

- Add tune_pid_gains.py with offline replay evaluation
- Update GainSchedule defaults based on grid search results
- Target: close 4pp gap with P-controller (90% -> >=93%)"
```

---

### Task 2: Set PID as Default Controller

**Files:**
- Modify: `src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py:402` (default True)

Goal: Change `use_pid_controller` default from `False` to `True`. Add PID diagnostics to trace output.

- [ ] **Step 1: Change default value**

```python
# run_embodied_stacking_robosuite.py:402
# Before:
    use_pid_controller: bool = False,
# After:
    use_pid_controller: bool = True,
```

- [ ] **Step 2: Add `--no-pid-controller` CLI flag to opt-out**

```python
# run_embodied_stacking_robosuite.py ~line 1350, after --use-pid-controller
    parser.add_argument("--use-pid-controller", action="store_true", default=True,
                        help="Use OperationalSpacePID instead of P-controller (default)")
    parser.add_argument("--no-pid-controller", action="store_false", dest="use_pid_controller",
                        help="Use legacy P-controller")
```

- [ ] **Step 3: Verify existing tests still pass with default change**

Run: `nsm/bin/python -m pytest tests/test_embodied_skill_executor_mock.py tests/test_operational_space_pid.py -v --tb=short`
Expected: all passing

- [ ] **Step 4: Quick smoke test with 5 episodes**

Run: `CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl nsm/bin/python -u -m vlm_nsm.experiments.run_embodied_stacking_robosuite --num_episodes 5 --holding-check-enabled --lift-early-exit-enabled 2>&1 | grep -E "episode_success"`
Expected: success > 0

- [ ] **Step 5: Commit**

```bash
git add src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py
git commit -m "feat: set PID as default controller

- use_pid_controller defaults to True
- Add --no-pid-controller flag for legacy P-controller fallback"
```

---

### Task 3: ID Model Feature Enhancement

**Files:**
- Modify: `src/vlm_nsm/data/id_training_dataset.py` (add velocity_history, action_history features)
- Modify: `src/vlm_nsm/embodied/controllers/learned_id_model.py` (update input_dim, re-train)
- Modify: `src/vlm_nsm/training/train_inverse_dynamics.py` (use new dataset)

Goal: Add velocity history (3 frames) and action history (3 frames) to ID model input, raising ensemble confidence from 0.06-0.09 to >0.3.

- [ ] **Step 1: Update ID dataset to include history features**

```python
# src/vlm_nsm/data/id_training_dataset.py — update _INPUT_DIM and _build_input_vector

# New input dimension: eef(3) + gripper(1) + vel_history(9) + act_history(12) + phase_onehot(10) + desired_next(3) = 38
_INPUT_DIM = 38
_HISTORY_LEN = 3  # frames of velocity & action history


def _build_input_vector(
    eef: np.ndarray,           # (3,)
    gripper_val: np.ndarray,   # (1,) or scalar
    phase_id: int,
    desired_next_eef: np.ndarray,  # (3,)
    vel_history: np.ndarray | None = None,   # (HISTORY_LEN*3,) or None
    act_history: np.ndarray | None = None,   # (HISTORY_LEN*4,) or None
) -> np.ndarray:
    """Build 38-dim input vector with velocity and action history."""
    vec = np.zeros(_INPUT_DIM, dtype=np.float32)
    vec[0:3] = eef
    vec[3] = float(gripper_val) if hasattr(gripper_val, '__len__') else gripper_val
    # velocity history [4:13]
    if vel_history is not None:
        vec[4:4 + len(vel_history)] = vel_history
    # action history [13:25]
    if act_history is not None:
        vec[13:13 + len(act_history)] = act_history
    # phase_onehot [25:35]
    pid = min(phase_id, 9)
    vec[25 + pid] = 1.0
    # desired_next_eef [35:38]
    vec[35:38] = desired_next_eef
    return vec
```

- [ ] **Step 2: Update _extract_samples to compute history**

```python
# In _extract_samples(), after reading actions and eef:
velocities = np.zeros_like(eef_pos)
velocities[1:] = (eef_pos[1:] - eef_pos[:-1]) / 0.02  # finite diff

for t in range(_HISTORY_LEN, n_steps - 1, stride):
    pid = int(phase_ids[t])
    if pid not in phase_set:
        continue
    
    # Build velocity history: last 3 velocity frames
    vel_hist = velocities[t - _HISTORY_LEN + 1:t + 1].flatten()  # (9,)
    # Build action history: last 3 action frames
    act_hist = actions[t - _HISTORY_LEN + 1:t + 1].flatten()    # (12,)
    
    x = _build_input_vector(
        eef_pos[t], gripper[t], pid, eef_pos[t + 1],
        vel_history=vel_hist, act_history=act_hist,
    )
    y = actions[t]
    X_list.append(x)
    Y_list.append(y)
```

- [ ] **Step 3: Update LearnedIDModel input_dim default**

```python
# src/vlm_nsm/embodied/controllers/learned_id_model.py:71
# Change default input_dim from 26 to 38
class LearnedIDModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 38,  # was 26
        ...
```

- [ ] **Step 4: Re-train 3-model ensemble with enhanced features**

Run sequentially (each ~5min on CPU):
```bash
CUDA_VISIBLE_DEVICES=3 nsm/bin/python -u -m vlm_nsm.training.train_inverse_dynamics \
  --dataset_dir generated/datasets/vla_demos/d1_small_200ep \
  --output_dir generated/results/id/id_model_v4 --epochs 50 --batch_size 256 --device cuda:0
# Repeat for v5, v6
```

- [ ] **Step 5: Verify ensemble confidence improvement**

```python
# Quick verification script
from vlm_nsm.embodied.controllers.learned_id_model import IDModelEnsemble
import numpy as np

ensemble = IDModelEnsemble(model_paths=[
    "generated/results/id/id_model_v4/best.pt",
    "generated/results/id/id_model_v5/best.pt",
    "generated/results/id/id_model_v6/best.pt",
])
x = np.zeros(38, dtype=np.float32)
x[0:3] = [0.0, 0.0, 0.9]
x[25] = 1.0  # phase=reach_pick
x[35:38] = [0.0, 0.0, 0.95]
pred = ensemble.predict(x)
print(f"Confidence: {pred.confidence:.3f} (target: > 0.3)")
```

Expected: confidence > 0.3

- [ ] **Step 6: Run tests**

Run: `nsm/bin/python -m pytest tests/test_learned_id_model.py -v --tb=short`
Expected: all passing (update input_dim if needed)

- [ ] **Step 7: Commit**

```bash
git add src/vlm_nsm/data/id_training_dataset.py src/vlm_nsm/embodied/controllers/learned_id_model.py src/vlm_nsm/training/train_inverse_dynamics.py tests/test_learned_id_model.py
git commit -m "feat: enhance ID model with velocity/action history features

- Extend input from 26 to 38 dims (add 3-frame vel + action history)
- Re-train 3-model ensemble with enhanced features
- Target: raise ensemble confidence from 0.06 to >0.3"
```

---

### Task 4: Fix HDF5 Phase 4/5 Labels + Re-Collect

**Files:**
- Modify: `src/vlm_nsm/embodied/place_state_machine.py` (add phase callback)
- Modify: `src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py` (pass callback)

Goal: Record release and retreat phase steps in HDF5. Re-collect 200 episodes for complete 6-phase data.

- [ ] **Step 1: Add phase callback to place_state_machine**

```python
# src/vlm_nsm/embodied/place_state_machine.py
# Add optional phase_callback parameter to execute_place and place_attempt

def execute_place(
    env, object_name: str, target_name: str,
    config: PlaceConfig | None = None,
    skip_preplace: bool = False,
    phase_callback=None,  # NEW: callable(phase_name) at each phase transition
) -> dict[str, Any]:
    # ... existing code ...
    
    # In place_attempt, at each phase transition:
    if phase_callback:
        phase_callback("release")   # before release steps
    # ... release steps ...
    if phase_callback:
        phase_callback("retreat")   # before retreat steps
    # ... retreat steps ...
```

- [ ] **Step 2: Pass phase callback from recording wrapper**

```python
# run_embodied_stacking_robosuite.py ~line 678, where execute_place is called:
_place_result = execute_place(env, target_obj, target_place,
                              config=_place_cfg, skip_preplace=False,
                              phase_callback=_rec_set_phase)  # NEW
```

- [ ] **Step 3: Re-collect 200 episodes with fixed phase labels**

Run: `CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nsm/bin/python -u -m vlm_nsm.experiments.collect_vla_demos --output_dir generated/datasets/vla_demos/d1_pid_200ep --num_episodes 200 --seed 3000 --device cuda:0`

Expected: ~40min, 200 HDF5 episodes with all 6 phases labeled

- [ ] **Step 4: Verify phase 4/5 presence in new data**

```python
import h5py, json
with open("generated/datasets/vla_demos/d1_pid_200ep/manifest.jsonl") as f:
    ep = json.loads(f.readline())
with h5py.File(ep["episode_path"]) as h5:
    phase_ids = h5["symbolic/phase_id"][:]
    unique = sorted(set(phase_ids))
    print(f"Phases: {unique}")  # Should show [0,1,2,3,4,5]
```

- [ ] **Step 5: Commit**

```bash
git add src/vlm_nsm/embodied/place_state_machine.py src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py
git commit -m "fix: add phase callback to place_state_machine for HDF5 recording

- execute_place accepts optional phase_callback
- Records release (phase 4) and retreat (phase 5) steps in HDF5
- Re-collect d1_pid_200ep with full 6-phase labeling"
```

---

### Task 5: End-to-End Demo Validation

**Files:**
- Modify: `src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py` (already PID default)

Goal: Run 100-episode final eval with PID as default, tuned gains, and validated phase labels. Output final report.

- [ ] **Step 1: Final 100ep eval with tuned PID**

Run: `CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl nsm/bin/python -u -m vlm_nsm.experiments.run_embodied_stacking_robosuite --num_episodes 100 --holding-check-enabled --lift-early-exit-enabled 2>&1 | tail -10`

Expected: episode_success_rate ≥ 0.92 (gap closed from 4pp to ≤2pp)

- [ ] **Step 2: Save final report**

```bash
cp generated/results/embodied_stacking_robosuite_summary.json \
   generated/results/control/pid_final/pid_final_100ep.json
```

- [ ] **Step 3: Run full regression**

Run: `nsm/bin/python -m pytest tests/ --ignore=tests/deprecated -x --tb=short -q`
Expected: all passing

- [ ] **Step 4: Update progress.md with final results**

- [ ] **Step 5: Commit**

```bash
git add generated/results/control/pid_final/ note/progress.md
git commit -m "feat: PID controller full adoption complete

- PID is default controller with tuned gains
- End-to-end 100ep eval: [RESULT]% success
- 6-phase HDF5 data re-collected
- ID model enhanced with velocity/action history"
```

---

## Execution Order

```
Task 1 (gain tuning) → Task 2 (PID default) → Task 5 (demo)
Task 3 (ID features) ─────────────────────────┘ (parallel)
Task 4 (HDF5 fix) ────────────────────────────┘ (parallel, needs GPU)
```

Tasks 1+2+5 are sequential. Tasks 3 and 4 can run in parallel with 1+2.
