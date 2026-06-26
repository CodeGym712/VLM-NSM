# Dataset Card — VLM-NSM Dataset v0.2

> **Version**: 0.2
> **Date**: 2026-05-21
> **Project Stage**: Stage 15 (DashScope Qwen Parser-Driven Integration)

---

## 1. Dataset Overview

The VLM-NSM dataset v0.2 aggregates experimental artifacts from the
language-to-symbolic-plan-to-robosuite-execution pipeline. It covers
parse-only evaluations, end-to-end (E2E) demonstrations, and control
policy benchmarks.

| Field | Value |
|:------|:------|
| Dataset Name | VLM-NSM Dataset Manifest |
| Version | 0.2 |
| Project Stage | Stage 15 |
| Controller | DAgger BC reach_pick + scripted precision phases |
| Parser | DashScope qwen3.6-plus (OpenAI-compatible API) |
| Task | 2-block stacking |

---

## 2. Dataset Components

| Component | Type | Samples | Parser |
|:----------|:-----|:-------:|:-------|
| language_eval_v0_1 | language_parse_eval | ~200+ | Qwen2.5-32B (text) |
| cached_qwen_e2e_v0_1_1 | e2e_demo | 5 | cached Qwen2.5 |
| dashscope_qwen_parse5_stage15 | parse_eval | 5 | online_qwen |
| dashscope_qwen_parse50_stage15 | parse_eval | 51 | online_qwen |
| dashscope_qwen_e2e5_stage15 | e2e_demo | 5 | online_qwen |
| dashscope_qwen_e2e10_stage15 | e2e_demo | 10 | online_qwen |
| control_learning_reach_pick_v0_1 | control_policy | 150 (3×50ep) | — |
| demo_media_v0_1 | demo_media | — | mock |
| stage15_artifacts | stage_artifacts | — | online_qwen |

---

## 3. Task Scope

All results are for **2-block stacking** unless otherwise specified.

Supported instruction categories:
- Direct object naming ("Stack cube A on cube B")
- Color-based ("Put the red cube on the green cube")
- Positional ("Place the cube on the left onto the right one")
- Object + color mixed ("Move the red cube onto cube B")
- Underspecified ("Stack them up")
- Chinese ("把红色方块放到绿色方块上")

---

## 4. Parser Modes

| Mode | Description | API |
|:-----|:------------|:----|
| `mock` | Deterministic parser (no API call) | None |
| `cached_qwen` | Pre-computed Qwen2.5 results | None (replay) |
| `online_qwen` | DashScope qwen3.6-plus API call | DashScope OpenAI-compatible |

The `online_qwen` parser uses:
- Endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Model: `qwen3.6-plus`
- Temperature: 0.0 (deterministic)
- Max tokens: 512

---

## 5. Standard E2E Summary Schema

Each E2E demo outputs a JSON summary with these fields:

| Field | Type | Description |
|:------|:-----|:------------|
| `instruction` | str | Natural language instruction |
| `success` | bool | Overall episode success |
| `parser` | str | Parser class name |
| `parse_success` | bool | Whether parsing succeeded |
| `parse_error` | str\|null | Parse error message |
| `fallback_used` | bool | Whether fallback parser was activated |
| `fallback_parser` | str\|null | Fallback parser name |
| `parsed_goal` | dict\|null | Raw parsed JSON |
| `goal_json` | dict\|null | Same as parsed_goal (redundant for backward compat) |
| `aligned_goal_predicates` | list | Predicates after alignment (may include `on(bottom, table)`) |
| `parser_target_stack` | list | Bottom-to-top stack derived from alignment |
| `demo_target_stack` | list | Reference stack from demo definition |
| `target_stack_source` | str | `"parser_aligned_goal"` or `"demo_hardcoded"` |
| `target_stack_mismatch` | bool | Whether derived stack differs from demo |
| `raw_qwen_output_path` | str\|null | Path to saved raw model output |
| `alignment_status` | str | `"passed"` or error description |
| `symbolic_plan` | list | Structured plan: `[{"name": "move", "args": [...]}]` |
| `physical_goal_predicates` | list | Predicates for physical success (no `on(bottom, table)`) |
| `plan_status` | str | `"success"` or error description |
| `plan_actions` | list | Flat action list |
| `planner_error` | str\|null | Planner error message |
| `policy_used` | str | Phase name using BC policy |
| `execution_success` | bool | Whether Robosuite episode succeeded |
| `failure_type` | str\|null | Episode failure type (None on success) |
| `episode_result` | dict | Full episode metrics |

---

## 6. Language-to-Plan Fields

Fields specific to parse-eval components:

| Field | Type | Description |
|:------|:-----|:------------|
| `total` | int | Total instructions |
| `parse_ok` | int | Successful parses |
| `schema_valid` | int | Valid schema count |
| `alignment_success` | int | Alignment passed |
| `target_stack_success` | int | Target stack derived |
| `planner_success` | int | Planner solved |
| `fully_successful` | int | All above passed |
| `failure_taxonomy` | dict | Failure type breakdown |

---

## 7. E2E Execution Fields

Failure types (from `episode_result.failure_type` or `failure_type`):

| Type | Description |
|:-----|:------------|
| `None` | Episode succeeded |
| `grasp_failed` | Robot failed to grasp the target block |
| `horizon` | Episode exceeded max steps |
| `verify_failed` | Final goal predicate check failed |
| `phase_error` | Phase transition error |

---

## 8. Control Learning Metadata

| Metric | Value | Source |
|:-------|:-----:|:-------|
| Canonical eval (100ep) | **92.0%** (92/100) | Stage 10 |
| Canonical eval (3×50ep avg) | **93.3%** (96/90/94%) | Stage 10 |
| Scripted baseline | 88.0% | Stage 9 |
| Policy type | DAgger Behavior Cloning | Stage 7B |
| BC phases | `reach_pick` only | Frozen |
| Non-BC phases | Scripted precision (grasp/lift/reach_place/release/verify) | Frozen |
| Checkpoint | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` | Frozen |
| Config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` | Frozen |

---

## 9. Known Limitations

1. **2-block stacking only**: All results are for 2-block stacking. 3-block or
   generalisation to other tasks is not yet validated.
2. **No vision grounding**: Object bindings (red → cube1, green → cube2) rely
   on prompt conventions, not visual perception.
3. **Fixed object conventions**: Evaluations use `cube1`/`cube2` with known
   mappings. Open-world object recognition is not supported.
4. **100% success is not generalisation**: Perfect scores reflect controlled
   evaluation conditions and fixed instruction templates, not open-world
   language understanding.
5. **Not a VLA policy**: The pipeline is symbolic planner + scripted controller,
   not an end-to-end Vision-Language-Action model.
6. **API Model**: DashScope `qwen3.6-plus` differs from Qwen2.5-32B. Behaviour
   on edge cases (underspecified instructions, alias-heavy prompts) may differ.
7. **Demo target stacks are hardcoded**: Physical success uses `on(top, bottom)`
   only. The `on(bottom, table)` predicate appears in alignment for completeness
   but is excluded from physical goal checking.

---

## 10. Reproducibility Commands

### Smoke test
```bash
bash scripts/smoke_dashscope_qwen.sh
```

### Parse-only
```bash
# 5 instructions
bash scripts/run_dashscope_parse5.sh
# 50 instructions (actually 51)
bash scripts/run_dashscope_parse50.sh
```

### End-to-end demo
```bash
# 5 demos (standardized summary)
MUJOCO_GL=egl bash scripts/run_dashscope_e2e_demo.sh
# 10 demos
MUJOCO_GL=egl bash scripts/run_dashscope_e2e10.sh
```

### Test suite
```bash
python3 -m pytest tests/ -q --tb=short
# Expected: 873 passed, 4 skipped
```

---

## 11. Privacy / API Key Policy

- **No API keys are included** in this dataset.
- DashScope API keys (`QWEN_API_KEY`) are stored in `configs/dashscope_qwen.env`
  which is gitignored.
- Use `configs/dashscope_qwen.env.example` as a template — never commit real keys.
- The dataset manifest and card contain only aggregate metrics and metadata.

---

## 12. Not Included

The following are outside the scope of this dataset:

- Raw DashScope API request/response logs (not saved by default unless
  `--save_raw_qwen_outputs` is passed)
- Training data or model weights (BC policy checkpoint is versioned separately)
- 3-block or multi-task evaluations
- Vision-language grounding data
- Real robot deployment logs
- Alternative controller policies (scripted-only, full BC)
