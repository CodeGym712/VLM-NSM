# VLM-NSM: Language-Aligned Symbolic Planning with Symbol-Conditioned Embodied Skill Execution

Independent extension project inspired by the original NSM pipeline. Does not modify the original NSM repository.

For a Python 3.10 Windows setup and the supported lightweight test command, see [Windows local development](docs/local-development.md).

## Three-Layer Architecture

```text
Layer 1: Language-Conditioned Symbolic Goal Planning
  NL Instruction → GoalParser → Alignment → PDDL → Planner → Symbolic Plan
  ─────────────────────────────────────────────────────────────
  Mock / Qwen2.5-Instruct text parser
  Alignment layer (object/predicate whitelist, type check, alias repair, completeness check)
  Dynamic PDDL problem generator
  Internal planner or Metric-FF adapter
  Symbolic state executor
  Predicate monitor

Layer 2: Language-Aligned Symbolic Planning Evaluation
  Benchmarks → Ablations → Metrics → Report Tables
  ─────────────────────────────────────────────────
  Language variation (easy + hard sets)
  Prompt comparison (full / medium / weak)
  Alignment mode ablation (5 modes)
  Model comparison (32B vs 14B)
  Failure case analysis & repair statistics

Layer 3: Embodied Skill Execution with Learnable Controller Interface
  Symbolic Plan → SkillStepDecomposer → Controller → Env → Monitors → Demo Recorder
  ────────────────────────────────────────────────────────────────────────────────
  SymbolConditionedObservationFilter
  Scripted / Replay / Learned controllers
  TerminationFailureMonitor, PhysicalPredicateExtractor, PredicateMonitor
  DemonstrationRecorder (npz for BC training)
```

## Quick Start

```bash
cd /path/to/VLM-NSM
# Activate environment (conda or venv with torch/transformers installed)
conda activate nsm

# Run tests
python tests/run_all.py

# Hanoi 3/4/5 symbolic experiments
bash scripts/run_hanoi_3.sh
bash scripts/run_hanoi_4.sh
bash scripts/run_hanoi_5.sh

# Language variation evaluations
# Easy set (8 Hanoi + 7 Stacking samples):
python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser mock --difficulty easy

# Hard set (22 Hanoi + 22 Stacking samples):
python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser mock --difficulty hard
```

On Windows with the minimal environment, use the curated local profile instead of `tests/run_all.py`:

```powershell
python scripts\run_local_tests.py
```

If Metric-FF is not auto-discovered, set:
```bash
export METRIC_FF_BIN=/path/to/metric-ff/ff
```

## Qwen2.5 Language Variation Evaluation

Evaluate Qwen2.5-32B-Instruct or Qwen2.5-14B-Instruct on language variation benchmarks:

```bash
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Single sample smoke test
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path /path/to/Qwen2.5-32B-Instruct \
  --model_name qwen25_32b --difficulty hard --max_samples 1

# Full hard set with prompt/alignment comparison
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path /path/to/Qwen2.5-32B-Instruct \
  --model_name qwen25_32b --difficulty hard \
  --prompt_variant weak --alignment_mode no_alignment

# Summarize results
python -m vlm_nsm.experiments.summarize_language_variation

# Export report tables
python -m vlm_nsm.experiments.export_report_tables
```

## Report Tables

After running evaluations:

- `report_table_model_comparison.csv` — per model/task/difficulty
- `report_table_prompt_comparison.csv` — full vs weak prompt
- `report_table_alignment_benefit.csv` — with/without alignment
- `report_table_by_category.csv` — per category breakdown
- `qwen_failure_cases.csv` — failed sample details
- `qwen_alignment_repair_stats.csv` — alignment repair statistics

## Results

All results in `generated/results/`.
Download in `generated/logs/qwen_outputs/`.

## Current Scope

- Hanoi 3/4/5 symbolic pipeline (internal planner + Metric-FF)
- Stacking 3/4/5 symbolic pipeline (internal planner)
- Dynamic PDDL problem generation
- Symbolic execution and predicate monitoring
- Language variation benchmark (easy + hard sets, 8 categories)
- Qwen2.5-32B-Instruct and Qwen2.5-14B-Instruct evaluation
- Alignment layer with object/predicate whitelist, type checking, alias repair, goal completeness
- Prompt variant comparison (full vs medium vs weak)
- Alignment mode ablation (full vs no_alignment vs no_repair vs no_alias_repair vs no_completeness_repair)
- Mock-first embodied stacking demo
- Alignment ablation and fixed-goal baseline

## Embodied Skill Execution (Stage 7A)

- SymbolConditionedObservationFilter: task-relevant observation selection using symbolic action context
- SkillStepDecomposer: fixed template for move → 6-step decomposition
- TerminationFailureMonitor: step-level termination detection
- ScriptedController and ReplayController
- DemonstrationRecorder: npz recording for BC training
- Reality Gap metric: (N_symbolic - N_physical) / N_symbolic × 100%

## Future Scope

- Real robosuite-backed stacking wrapper refinement
- Qwen2.5-VL image parser
- MLP Behavior Cloning
- Temporal Behavior Cloning
- Diffusion Policy
- Hybrid Controller (learned + scripted)
- Fast Downward adapter
