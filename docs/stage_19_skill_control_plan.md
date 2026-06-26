# Stage 19 Skill Control Plan

## Motivation

The upper stack is already parser-driven:

```text
instruction -> Qwen parser -> alignment -> target_stack -> symbolic plan
```

Stage 19 starts the control-side refactor without replacing the frozen
controller. The first step is to insert a stable skill interface between the
symbolic planner and the existing Robosuite runner.

## Current Implementation

New files:

```text
src/vlm_nsm/skills/skill_types.py
src/vlm_nsm/skills/skill_plan_from_symbolic.py
src/vlm_nsm/execution/scripted_skill_executor.py
scripts/run_e2e_skill_executor5.sh
```

Core schema:

```text
SkillCommand(skill_name="pick_place", object_id="cube1",
             source="table", target_object_id="cube2", relation="on")
SkillPlan(commands=[...], source_plan=[...])
SkillExecutionResult(success, steps, failure_type, trajectory_path, metrics)
```

Symbolic conversion:

```text
move(cube1, table, cube2)
  -> pick_place(object_id=cube1, source=table, target_object_id=cube2, relation=on)
```

The E2E runner now has an opt-in control interface:

```bash
--control_interface legacy          # default, existing behavior
--control_interface skill_executor  # new SkillPlan -> ScriptedSkillExecutor path
```

## Constraints Preserved

- The canonical checkpoint is not modified:
  `generated/results/stage_7B-4/dagger_reach_bc/best.pt`
- The canonical config is not modified:
  `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`
- The default E2E path remains `legacy`.
- The learned skill policy is not a default.
- Physical success still uses `on(top, bottom)` and does not add the table
  support predicate.
- DashScope is not required for the skill executor smoke path.

## Validation

Unit tests:

```bash
nsm/bin/python -m pytest \
  tests/test_skill_types.py \
  tests/test_skill_plan_from_symbolic.py \
  tests/test_scripted_skill_executor.py \
  tests/test_e2e_skill_control_interface.py \
  -q --tb=short
```

Robosuite smoke:

```bash
MUJOCO_GL=egl PYTHON_BIN=nsm/bin/python bash scripts/run_e2e_skill_executor5.sh
```

## Next Steps

1. Run the skill executor E2E5 smoke and compare against the legacy result.
2. Add `collect_skill_demonstrations.py` using `ScriptedSkillExecutor` as the
   teacher.
3. Store per-skill manifest rows and `.npz` trajectories.
4. Implement `SkillDataset` for `obs_history -> action_chunk`.
5. Train ACT-lite as the first learned skill baseline.
6. Add diffusion only after the skill dataset and ACT baseline are stable.
