# VLM-NSM: Language-Aligned Symbolic Planning with Symbol-Conditioned Embodied Skill Execution

## System Overview

```
VLM-NSM v2 Architecture
========================

Three-Layer Design:

Layer 1: Language-Conditioned Symbolic Goal Planning
  NL Instruction → GoalParser → Alignment → PDDL → Planner → Symbolic Plan
  ─────────────────────────────────────────────────────────────
  Responsible for: understanding natural language, validating goals,
  generating optimal symbolic plans. All symbolic, all verifiable.

Layer 2: Language-Aligned Symbolic Planning Evaluation
  Benchmarks → Ablations → Metrics → Report Tables
  ─────────────────────────────────────────────────
  Responsible for: quantifying language robustness, alignment value,
  prompt sensitivity, model comparison.

Layer 3: Embodied Skill Execution with Learnable Controller Interface
  Symbolic Plan → SkillStepDecomposer → Controller → Env → Monitors
  ─────────────────────────────────────────────────────────────────
  Responsible for: decomposing symbolic actions into physical steps,
  executing via scripted/learned controllers, recording demonstrations,
  detecting execution failures, measuring reality gap.
```

---

## Layer 1: Language-Conditioned Symbolic Goal Planning

### Pipeline

```
Natural Language Instruction / Object List
  │
  ▼
GoalParser (NL → JSON goal)
  │  MockGoalParser (test only)
  │  QwenTextGoalParser (Qwen2.5-Instruct)
  │
  ▼
Alignment Layer
  │  Object whitelist validation
  │  Predicate whitelist + alias repair (above→on etc.)
  │  Type signature checking
  │  Goal completeness checking (Hanoi / Stacking)
  │
  ▼
PDDL Problem Generator
  │  Dynamic domain + problem generation
  │  Smaller predicate inference
  │
  ▼
Planner Adapter
  │  InternalHanoiPlanner (optimal recursive)
  │  InternalStackingPlanner (optimal greedy)
  │  Metric-FF (external)
  │  FastDownward (reserved)
  │
  ▼
Abstract Symbolic Plan
  move(cube2, table, cube3)
  move(cube1, table, cube2)
```

### Key Design Decisions

1. **LLM outputs goals, not plans.** Long-horizon planning is delegated to symbolic planners, avoiding LLM hallucination in action sequences.
2. **Alignment layer as safety gate.** Every parsed goal passes through validation before reaching the planner. This catches LLM errors like invalid objects, wrong predicate names, type mismatches, and incomplete specifications.
3. **Dynamic PDDL generation.** Supports Hanoi 3/4/5 and Stacking 3/4/5 without hardcoded templates. Smaller predicates are inferred from type hierarchies.

### Components

| Module | File | Purpose |
|--------|------|---------|
| GoalParser | `core/goal_parser.py` | NL → JSON goal |
| Alignment | `core/alignment.py` | Validation + repair |
| PDDLGenerator | `core/pddl_generator.py` | Dynamic PDDL |
| PlannerAdapter | `core/planner_adapter.py` | Planner routing |
| HanoiPlanner | `planners/internal_hanoi_planner.py` | Hanoi optimal planner |
| StackingPlanner | `planners/internal_stacking_planner.py` | Stacking optimal planner |
| SymbolicExecutor | `executors/symbolic_executor.py` | Symbolic state execution |
| PredicateMonitor | `core/predicate_monitor.py` | Goal satisfaction check |

---

## Layer 2: Language-Aligned Symbolic Planning Evaluation

### Benchmarks

| Dataset | Samples | Categories | Purpose |
|---------|---------|------------|---------|
| Hanoi Easy | 8 | 8 categories | Basic language understanding |
| Stacking Easy | 7 | 7 categories | Basic language understanding |
| Hanoi Hard | 22 | 6 categories | Stress test: aliases, underspec, synonyms, mixed lang, noise, misleading |
| Stacking Hard | 22 | 6 categories | Stress test: aliases, underspec, synonyms, mixed lang, noise, partial |

### Evaluation Dimensions

- **Model comparison**: Qwen2.5-32B-Instruct vs Qwen2.5-14B-Instruct
- **Prompt comparison**: full (detailed schema) vs medium (minimal schema) vs weak (no schema)
- **Alignment ablation**: full vs no_alignment vs no_repair vs no_alias_repair vs no_completeness_repair

### Metrics

| Metric | Definition |
|--------|------------|
| json_parse_rate | Proportion of samples producing parseable JSON |
| schema_validity_rate | Proportion of samples with valid goal_predicates structure |
| alignment_success_rate | Proportion of samples passing alignment (N/A for no_alignment mode) |
| repair_rate | Proportion of samples where alignment applied repairs |
| planner_success_rate | Proportion of samples where planner found a plan |
| execution_success_rate | Proportion of samples where executor succeeded |
| goal_success_rate | Proportion of samples satisfying the goal |

### Report Tables

| Table | Content |
|-------|---------|
| model_comparison | Per model/task/difficulty (hard, full+medium prompts) |
| prompt_comparison | 32B per prompt variant (full/medium/weak) |
| alignment_benefit | 32B medium prompt, 5 alignment modes |
| by_category | Per category breakdown |

### Baselines

**fixed_goal_nsm:**
Uses hardcoded goal predicates, bypasses GoalParser entirely, goes directly to planner.
Purpose: Quantify the value of the language interface. If fixed_goal_nsm outperforms the full pipeline, the language parser is adding error.

**llm_direct_plan:**
Let LLM output action sequences directly, skipping planner entirely.
Purpose: Prove that symbolic planner is necessary. If the LLM cannot produce valid action sequences, the planner is justified.

---

## Layer 3: Embodied Skill Execution with Learnable Controller Interface

### Pipeline

```
Abstract Symbolic Plan
  move(cube2, table, cube3)
  move(cube1, table, cube2)
  │
  ▼
SymbolConditionedObservationFilter
  │  Uses symbolic action context (action name, target object, target location)
  │  to select task-relevant observations from the environment.
  │  Does NOT use privileged ground-truth labels.
  │  Output: filtered observation (obs_dim,)
  │
  ▼
SkillStepDecomposer
  │  Decomposes move(obj, source, target) into fixed step sequence:
  │  1. reach_pick  2. grasp  3. lift  4. reach_place  5. release  6. retreat
  │  Current version: fixed task-specific template for stacking pick-and-place.
  │  Future work: dynamic step generation.
  │
  ▼
Controller Interface
  │  BaseController (abstract)
  │  ├── ScriptedController (hardcoded poses, no learning)
  │  ├── ReplayController (recorded trajectories → replay)
  │  └── Future: MLP BC, Temporal BC, Diffusion Policy, Hybrid
  │
  ▼
Environment (MockEnvWrapper / RobosuiteEnvWrapper)
  │
  ▼
Monitors
  │  TerminationFailureMonitor → StepSignal
  │  PhysicalPredicateExtractor → PredicateState
  │  PredicateMonitor → GoalSignal
  │
  ▼
DemonstrationRecorder
     Saves (obs, action, done, episode_id, skill_id, step_id, success)
     as npz + metadata JSON for downstream BC training.
```

### Component Details

#### SymbolConditionedObservationFilter

```
File: src/vlm_nsm/embodied/symbol_conditioned_observation_filter.py
Class: SymbolConditionedObservationFilter
```

Uses symbolic action context available from the planner to select task-relevant observations. Does not use privileged ground-truth labels unavailable at test time.

Input:
- `action`: Action(name, args) — symbolic action from planner
- `raw_observation`: dict — full env observation (all object poses)

Output:
- `filtered_obs`: np.ndarray shape (obs_dim,) — task-relevant observation vector

Design principle: The filter uses only information available at test time — the symbolic action that the planner produced. This distinguishes it from an oracle that would use ground-truth labels.

#### SkillStepDecomposer

```
File: src/vlm_nsm/embodied/skill_step_decomposer.py
```

Current version: Fixed task-specific template for Stacking pick-and-place style move actions.

Decomposition of `move(obj, source, target)`:

| Step | Name | Description |
|------|------|-------------|
| 1 | reach_pick | Move to grasp pose above object |
| 2 | grasp | Close gripper |
| 3 | lift | Lift object |
| 4 | reach_place | Move to place pose above target |
| 5 | release | Open gripper |
| 6 | retreat | Move away |

Future work: Dynamic step generation based on action type, object properties, and scene context.

#### Controller Interface

```
BaseController (src/vlm_nsm/embodied/controllers/base_controller.py)
├── ScriptedController (src/vlm_nsm/embodied/controllers/scripted_controller.py)
├── ReplayController (src/vlm_nsm/embodied/controllers/replay_controller.py)
├── MLP BC (future)
├── Temporal BC (future)
├── Diffusion Policy (future)
└── Hybrid Controller (future)
```

**ScriptedController:** Hardcoded pose sequences for each skill step. No learning, used as a sanity check for the pipeline.

**ReplayController:** Records demonstrations and replays them. Replay success rate serves as data quality sanity check before training BC models. If replay_success_rate is low, do not proceed to BC training.

#### Monitor Interfaces

Three distinct types with separate responsibilities:

**StepSignal** (from TerminationFailureMonitor):
```python
@dataclass
class StepSignal:
    done: bool              # Episode complete
    success: bool           # Step succeeded
    failed: bool            # Step failed
    timeout: bool           # Step timed out
    failure_type: str | None  # e.g., "grasp_failed", "collision", "precondition_violation"
```

**PredicateState** (from PhysicalPredicateExtractor):
```python
@dataclass
class PredicateState:
    predicates: list[Predicate]  # Extracted predicates
    source: str                  # "symbolic" or "physical"
```

**GoalSignal** (from PredicateMonitor):
```python
@dataclass
class GoalSignal:
    goal_satisfied: bool
    missing_predicates: list[str]
```

These three types must not be conflated. TerminationFailureMonitor handles step-level termination. PhysicalPredicateExtractor produces symbolic state from physical observations. PredicateMonitor evaluates goal satisfaction.

#### DemonstrationRecorder

```
File: src/vlm_nsm/embodied/demonstration_recorder.py
```

Saves data in npz format compatible with downstream BC training (MLP BC, Temporal BC, Diffusion Policy).

**npz schema:**

| Field | Shape | Description |
|-------|-------|-------------|
| `obs` | (T, obs_dim) | Observation sequence |
| `action` | (T, action_dim) | Action sequence |
| `done` | (T,) | Episode termination flags |
| `episode_id` | (T,) | Episode index per timestep |
| `skill_id` | (T,) | Skill index (0=reach_pick, 1=grasp, etc.) |
| `step_id` | (T,) | Step index within episode |
| `success` | (T,) | Success flag per timestep |

**metadata JSON schema:**

| Field | Type | Description |
|-------|------|-------------|
| task | str | "hanoi" or "stacking" |
| n_blocks | int | Number of blocks |
| env_type | str | "mock" or "robosuite" |
| controller_type | str | "scripted" or "replay" |
| num_episodes | int | Total episodes |
| num_success | int | Successful episodes |
| success_rate | float | num_success / num_episodes |
| obs_dim | int | Observation dimension |
| action_dim | int | Action dimension |
| step_names | list[str] | ["reach_pick", "grasp", "lift", ...] |
| observation_type | str | e.g. "symbol_conditioned" |
| created_at | str | ISO timestamp |

---

## Reality Gap

### Definition

The Reality Gap measures the systematic discrepancy between symbolic planning success rate and physical execution success rate:

$$
\text{Reality Gap} =
\frac{
N_{\text{symbolic\_success}} - N_{\text{physical\_success}}
}{
N_{\text{symbolic\_success}}
}
\times 100\%
$$

Where:
- `N_symbolic_success`: Number of episodes where the symbolic planner and symbolic executor produce a successful plan execution.
- `N_physical_success`: Number of episodes where the physical (embodied) executor successfully executes the same plans.

If `N_symbolic_success = 0`, the Reality Gap is recorded as N/A.

### Interpretation

| Reality Gap | Meaning |
|-------------|---------|
| 0% | Perfect sim-to-real transfer |
| 0-20% | Minor execution noise |
| 20-50% | Significant embodiment challenges |
| >50% | Fundamental mismatch between symbolic assumptions and physics |
| N/A | Symbolic pipeline itself is not succeeding |

---

## Experiment C: Embodied Execution Evaluation

### Setup

- Use Stacking 3 task (3 cubes)
- ScriptedController for pick-and-place
- MockEnvWrapper (unit tests) / RobosuiteEnvWrapper (full eval)
- N episodes per condition

### Metrics

| Metric | Definition |
|--------|------------|
| step_success_rate | Fraction of individual steps that succeed |
| skill_success_rate | Fraction of complete skills (all 6 steps) that succeed |
| episode_success_rate | Fraction of full episodes that succeed |
| failure_stage_distribution | Distribution of failures across skill steps |
| predicate_match_rate | Fraction of predicted predicates matching physical state |
| goal_success_rate | Fraction of episodes satisfying the goal in physical execution |
| reality_gap | (N_symbolic - N_physical) / N_symbolic × 100% |

### Conditions

| Condition | Controller | Environment | Purpose |
|-----------|------------|-------------|---------|
| C1 | ScriptedController | MockEnv | Pipeline sanity check |
| C2 | ScriptedController | Robosuite | Baseline embodied execution |
| C3 | ReplayController | MockEnv | Replay quality check |
| C4 | ReplayController | Robosuite | Full replay evaluation |

---

## Stage Roadmap

### Stage 7A: Embodied Skill Execution Interface (Current)

Build the Layer 3 infrastructure:

1. **SymbolConditionedObservationFilter** — filter observations using symbolic action context
2. **SkillStepDecomposer** — fixed template for move → 6-step decomposition
3. **TerminationFailureMonitor** — step-level termination detection
4. **BaseController + ScriptedController** — hardcoded controller
5. **DemonstrationRecorder** — npz recording with defined schema
6. **Embodied execution pipeline** — integrate all components in a run script
7. **Experiment C.1** — ScriptedController + MockEnv evaluation
8. **Reality Gap calculation** — compare symbolic vs physical success

### Stage 7B: Controller Learning Pipeline

```
Demonstration Collection (ScriptedController)
  → ReplayController Sanity Check (data quality gate)
  → MLP Behavior Cloning
  → Temporal Behavior Cloning
  → Diffusion Policy
  → Hybrid Controller (learned + scripted)
```

Key principle: ReplayController success rate is a data quality gate. If replay_success_rate is low, do not proceed to BC training.

### Stage 8: VLM Integration and Full Evaluation

- Qwen2.5-VL image-based goal parsing
- Full Reality Gap analysis across all conditions
- Paper writing and final results

---

## File Structure

### New Code (embodied module)

```
src/vlm_nsm/embodied/
├── __init__.py
├── symbol_conditioned_observation_filter.py   # SymbolConditionedObservationFilter
├── skill_step_decomposer.py                    # SkillStepDecomposer (fixed template)
├── termination_failure_monitor.py              # TerminationFailureMonitor → StepSignal
├── demonstration_recorder.py                   # DemonstrationRecorder (npz schema)
└── controllers/
    ├── __init__.py
    ├── base_controller.py                      # BaseController (abstract)
    ├── scripted_controller.py                  # ScriptedController (hardcoded poses)
    └── replay_controller.py                    # ReplayController (recorded trajectories)
```

### Tests

```
tests/
├── test_symbol_conditioned_observation_filter.py
├── test_skill_step_decomposer.py
├── test_termination_failure_monitor.py
├── test_demonstration_recorder_schema.py
├── test_replay_controller.py
└── test_reality_gap_metric.py
```

### Baselines (current location, experimental)

```
src/vlm_nsm/baselines/
├── fixed_goal_nsm.py       # Hardcoded goal predicates, no language parsing
└── llm_direct_plan.py      # LLM outputs actions directly, no planner
```

Note: Baselines are defined as experimental baselines. Future migration to `src/vlm_nsm/experiments/baselines/` is possible but not yet scheduled.

---

## Testing Requirements

All new tests must:
1. Not depend on real Qwen models (use mock parsers)
2. Not depend on real Robosuite (use MockEnvWrapper)
3. Not modify stable symbolic planning mainline
4. Pass alongside existing 152 tests

---

## Appendix: Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-11 | v2.0 | Full architecture revision: three-layer design, renamed OracleFilter → SymbolConditionedObservationFilter, Layer 3 naming fix, Reality Gap definition, Experiment C metrics, baseline definitions, Stage 7B route adjustment |
