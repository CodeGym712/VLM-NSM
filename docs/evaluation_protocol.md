# Evaluation Protocol

## Canonical Evaluation Paths

### 1. Language Evaluation Pipeline

```bash
# Step 1: Run language variation eval (Qwen models)
python3 -m vlm_nsm.experiments.run_language_variation \
    --task hanoi --n_blocks 3 --parser qwen_text \
    --model_path /path/to/Qwen2.5-32B-Instruct --model_name qwen25_32b \
    --difficulty easy --prompt_variant full --alignment_mode full

# Step 2: Summarize results
python3 -m vlm_nsm.experiments.summarize_language_variation

# Step 3: Analyze failures
python3 -m vlm_nsm.experiments.analyze_qwen_failures

# Step 4: Analyze alignment repairs
python3 -m vlm_nsm.experiments.analyze_alignment_repairs

# Step 5: Export report tables
python3 -m vlm_nsm.experiments.export_report_tables
```

**Batch script**: `scripts/run_all_remaining.sh` runs all 14 eval combinations (12 × 32B + 2 × 14B) sequentially.

### 2. Embodied Expert Evaluation

```bash
# Run expert policy with shared episode runner
python3 -m vlm_nsm.experiments.run_embodied_stacking_robosuite \
    --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
    --num_episodes N --output results.json
```

The shared runner (`robosuite_episode_runner.py`) is the canonical evaluation path.
It uses proper state machines (grasp, place), termination logic, and goal verification.

### 3. Hybrid BC Evaluation

```bash
# Evaluate BC model in hybrid reach-only mode
python3 -m vlm_nsm.learning.eval_bc_hybrid \
    --checkpoint path/to/checkpoint.pt \
    --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
    --num_episodes 20 --seed 0 --device cuda \
    --condition_on_phase \
    --bc_phases reach_pick reach_place \
    --fixed_zero_action_dims 3 4 5 \
    --output results.json

# Scripted-only baseline
python3 -m vlm_nsm.learning.eval_bc_hybrid \
    --checkpoint path/to/checkpoint.pt \
    --config generated/configs/stage_7A-12/.../robosuite_best_execution_config_stage7a12_final.json \
    --num_episodes 20 --bc_phases none --output results.json
```

### 4. BC Training

```bash
# Train on per-episode npz files (canonical)
python3 -m vlm_nsm.learning.train_bc \
    --demo_dir generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/ \
    --obs_key filtered_obs --condition_on_phase \
    --epochs 200 --batch-size 64 --lr 1e-3 --device cuda \
    --fixed_zero_action_dims 3 4 5 \
    --save model.pt

# Train on single aggregated npz
python3 -m vlm_nsm.learning.train_bc \
    --npz path/to/dataset.npz --obs_key filtered_obs --condition_on_phase \
    --epochs 200 --batch-size 64 --lr 1e-3 --device cuda \
    --save model.pt
```

### 5. DAgger Data Collection

```bash
python3 -m vlm_nsm.learning.collect_dagger_data \
    --checkpoint path/to/base_bc.pt \
    --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
    --num_episodes 30 --device cuda \
    --condition_on_phase --bc_phases reach_pick reach_place \
    --output dagger_data.npz
```

### 6. Final Report Generation

```bash
python3 -m vlm_nsm.experiments.export_final_report_tables
```

Output: `generated/report_tables/final/*.md`

---

## Canonical Data Sources

| Dataset | Path | Usage |
|---------|:----:|-------|
| Expert v2 strict_only (dir) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` | BC training |
| Expert v2 strict_only (npz) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/trajectories_strict_only.npz` | Aggregated training |
| DAgger round 1 | `generated/demos/stage_7B-4/dagger_round1_reach/trajectories_dagger_round1_reach.npz` | On-policy DAgger data |
| Aggregated reach | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` | Expert + DAgger combined |

Only **true phase_id** data should be used. Datasets with synthetic phase_id are deprecated.

---

## Deprecated Paths

| Path | Reason | Replacement |
|:----:|--------|:------------|
| `eval_bc.py` manual env loop | Uses horizon//6 phase division, zero actions for non-BC phases, no state machines | `eval_bc_hybrid.py` with shared `run_episode()` |
| Synthetic phase_id | Artificially divided phases inconsistent with true state machine transitions | True phase_id from `RobosuiteEnvWrapper` |
| `strict_success/` mixed directory | Contains episodes that only pass semantic check, not strict goal check | `strict_only/` directory with verified strict demos |
| Offline MSE as primary metric | Does not correlate with closed-loop control performance | Closed-loop success rate via `eval_bc_hybrid.py` |
| BC-controlled grasp as default | Model has no grasp training data → catastrophic 0% | Keep grasp/lift/release/retreat scripted; BC only for reach phases |

---

## Key Constraints

1. **Shared runner is canonical**: `run_episode()` in `robosuite_episode_runner.py` is the only valid evaluation path.
2. **BC phases = reach_pick + reach_place**: Do not extend to grasp/lift/release/retreat without retraining.
3. **No offline MSE as success metric**: Validation loss does not predict closed-loop success (DAgger-only: 0.000627 val loss ↔ 0% success).
4. **DAgger data alone insufficient**: Must be combined with expert data.
