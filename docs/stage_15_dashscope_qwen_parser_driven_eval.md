# Stage 15: DashScope Qwen Parser-Driven Integration

> **Version**: v0.2.0 (Dataset manifest + large-scale validation)
> **Date**: 2026-05-21
> **Model**: qwen3.6-plus (DashScope OpenAI-compatible API)

---

## 1. Motivation

The online Qwen parser was designed for API-based LLMs (Qwen2.5-32B). Stage 15
migrates from the official Qwen API to **DashScope** (Alibaba Cloud's unified
model API) for two reasons:

1. **Unified endpoint**: DashScope provides OpenAI-compatible chat completion,
   eliminating SDK duplication.
2. **qwen3.6-plus**: A newer, faster model than Qwen2.5-32B, with lower latency
   and comparable instruction-following quality for structured JSON output.
3. **API key isolation**: DashScope keys are managed via
   `configs/dashscope_qwen.env` (gitignored), preventing accidental commits.

---

## 2. API Configuration

| Parameter | Value |
|:----------|:------|
| Endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Model | `qwen3.6-plus` |
| Auth | API key in `QWEN_API_KEY` env var |
| Temperature | 0.0 (deterministic) |
| Max tokens | 512 |

Environment file: `configs/dashscope_qwen.env.example`

```
QWEN_API_KEY=sk-your-key-here
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.6-plus
```

Copy to `configs/dashscope_qwen.env` (gitignored) and fill in the real key.

---

## 3. Smoke Test

```bash
bash scripts/smoke_dashscope_qwen.sh
```

Expected output: a valid JSON response with `task_type`, `object_bindings`,
and `goal_predicates`. Latency is typically 1–3 seconds for a two-block
stacking instruction.

---

## 4. Parser Adaptation

### 4.1 Schema

The prompt was updated to match the `AlignmentLayer` schema:

```json
{
  "task_type": "stacking",
  "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
  "goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]]
}
```

### 4.2 Kwargs Compatibility

`OnlineQwenGoalParser.parse()` now accepts all E2E kwargs:

- `instruction`
- `available_objects` / `objects`
- `task_type`
- `n_blocks`
- `scene_description` (reserved for future vision grounding)

Extra kwargs are accepted via `**kwargs` without error.

---

## 5. Parser-Driven Plan-Only Fix

The `eval_online_qwen_parser.py` was updated to use the same parser-driven
target_stack derivation as the E2E pipeline.

**Old behavior**: `demo["target_stack"]` hardcoded → planner used fixed stack.

**New behavior**:
```
goal_json → AlignmentLayer → target_stack_from_predicates() → planner
```

This ensures the eval script and E2E script use the same derivation logic.

---

## 6. Parse-Only 5 Results

5 DashScope instructions through the full parse→align→plan pipeline:

| Metric | Value |
|:-------|:-----:|
| parse_ok | 5/5 (100%) |
| schema_valid | 5/5 (100%) |
| alignment_success | 5/5 (100%) |
| target_stack_success | 5/5 (100%) |
| planner_success | 5/5 (100%) |
| fully_successful | 5/5 (100%) |

---

## 7. E2E 5 Results

5 DashScope instructions through the full
parse→align→plan→Robosuite pipeline:

| Metric | Value |
|:-------|:-----:|
| E2E success | 5/5 (100%) |
| Controller | DAgger BC reach_pick + scripted precision phases |

All 5 demos succeeded with the parser-driven target_stack.

---

## 8. Standardized E2E 5 Re-Run

The E2E 5 demo was re-run after standardising the summary schema (S15.1-A).
All 5 demos now output the full 29-field standardised summary including
`goal_json`, `aligned_goal_predicates`, `parser_target_stack`,
`symbolic_plan`, `physical_goal_predicates`, and `failure_type`.

| Demo | Instruction | Success | Steps | BC Steps |
|:----:|:------------|:-------:|:-----:|:--------:|
| 1 | Put the red cube on the green cube. | ✅ | 750 | 95 |
| 2 | Stack cube A on cube B. | ✅ | 795 | 116 |
| 3 | Place the cube on top of the other cube. | ✅ | 782 | 111 |
| 4 | Move the block onto the target block. | ✅ | 780 | 97 |
| 5 | Stack the red block on top of the green one. | ✅ | 790 | 100 |

**Result**: 5/5 successful (100%) with standardised summary.

Output: `generated/results/stage_15/e2e_dashscope_qwen_summary_v2/`

---

## 9. Parse50 Large-Scale Parse Eval

50 instructions across 6 categories (direct, color, positional, mixed,
underspecified, Chinese) through the parse→align→plan pipeline.

**Note**: The input JSON file contains 51 records (one extra from the planned 50).
The benchmark is named "Parse50" and results are documented as 51/51.

| Category | Count | Example |
|:---------|:-----:|:--------|
| Direct object naming | 10 | "Stack cube A on cube B." |
| Color-based | 10 | "Put the red cube on the green cube." |
| Positional/referential | 10 | "Place the cube on the left onto the right one." |
| Object + color mixed | 8 | "Move the red cube onto cube B." |
| Underspecified | 7 | "Stack them up." |
| Noise / meta | 6 | "Ignore the previous instruction..." |

Results:

| Metric | Value |
|:-------|:-----:|
| total | 51 |
| parse_ok | 51/51 (100%) |
| schema_valid | 51/51 (100%) |
| alignment_success | 51/51 (100%) |
| target_stack_success | 51/51 (100%) |
| planner_success | 51/51 (100%) |
| fully_successful | 51/51 (100%) |
| failure_taxonomy | {} (no failures) |

Output: `generated/results/stage_15/dashscope_qwen_parse_50/`

---

## 10. E2E10 Validation

10 DashScope instructions through the full E2E pipeline using the new
`--instructions` JSON loading support (added to `run_e2e_language_to_sim.py`).

| Demo | Instruction | Success | Steps | BC Steps |
|:----:|:------------|:-------:|:-----:|:--------:|
| 1 | Put the red cube on the green cube. | ✅ | 807 | 108 |
| 2 | Stack cube A on cube B. | ✅ | 785 | 105 |
| 3 | Place the cube on top of the other cube. | ✅ | 784 | 101 |
| 4 | Move the block onto the target block. | ✅ | 793 | 123 |
| 5 | Stack the red block on top of the green one. | ✅ | 775 | 102 |
| 6 | Place cube1 onto cube2. | ✅ | 796 | 113 |
| 7 | Make a tower with the two blocks. | ✅ | 805 | 124 |
| 8 | Put the green block on top of the red one. | ✅ | 742 | 75 |
| 9 | Stack them up with cube B at the bottom. | ✅ | 805 | 100 |
| 10 | Put the red cube on the green cube, then done. | ✅ | 787 | 102 |

**Result**: 10/10 successful (100%).

Output: `generated/results/stage_15/e2e_dashscope_qwen_parser_driven_10/`

---

## 11. Test Suite

Full test suite after all Stage 15 additions:

```text
python3 -m pytest tests/ -q --tb=short
873 passed, 4 skipped
```

New Stage 15 tests:

| Test file | Tests |
|:----------|:-----:|
| `tests/test_e2e_summary_parser_driven_fields.py` | 12 |
| `tests/test_eval_online_qwen_metrics_outputs.py` | 6 |
| `tests/test_dataset_manifest_schema.py` | S16-B |
| `tests/test_build_dataset_manifest.py` | S16-C |
| `tests/test_dataset_card_v0_2.py` | S16-E |

---

## 12. Parser-Driven Summary Schema

The E2E summary now includes standardized fields for dataset reporting:

| Field | Description |
|:------|:------------|
| `goal_json` | Raw parsed JSON from the parser |
| `aligned_goal_predicates` | Predicates after alignment (may include `on(bottom, table)`) |
| `parser_target_stack` | Bottom-to-top stack derived from alignment |
| `demo_target_stack` | Reference stack from demo definition |
| `target_stack_source` | `"parser_aligned_goal"` or `"demo_hardcoded"` |
| `target_stack_mismatch` | Whether derived stack differs from demo |
| `symbolic_plan` | Structured plan: `[{"name": "move", "args": [...]}]` |
| `physical_goal_predicates` | Predicates for physical success (no table predicate) |
| `raw_qwen_output_path` | Path to saved raw model output (if saved) |
| `execution_success` | Whether the Robosuite episode succeeded |
| `failure_type` | Episode failure type (None on success) |

Example:
```json
{
  "aligned_goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]],
  "parser_target_stack": ["cube2", "cube1"],
  "target_stack_source": "parser_aligned_goal",
  "target_stack_mismatch": false,
  "symbolic_plan": [{"name": "move", "args": ["cube1", "table", "cube2"]}],
  "physical_goal_predicates": [["on", "cube1", "cube2"]]
}
```

---

## 13. Frozen Controller

The controller policy is unchanged from the canonical deployment:

| Phase | Policy |
|:------|:-------|
| reach_pick | DAgger BC (trained) |
| grasp | Scripted precision |
| lift | Scripted precision |
| reach_place | Scripted precision |
| release | Scripted precision |
| retreat / verify | Scripted precision |

No changes to:
- Checkpoint: `generated/results/stage_7B-4/dagger_reach_bc/best.pt`
- Config: `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`
- BC phases (reach_pick only)

---

## 14. Limitations

1. **Scale**: 5 + 51 parse-only and 5 + 10 E2E demos tested. Broader
   validation (e.g., 3-block, multi-task) has not been performed.
2. **Model delta**: qwen3.6-plus is not Qwen2.5-32B — behavior may differ
   on edge cases (underspecified instructions, alias-heavy prompts).
3. **Grounding**: Current grounding is purely prompt/object-convention based.
   No vision grounding or scene perception is used.
4. **Not a VLA policy**: This is a symbolic planner + scripted controller
   pipeline, not an end-to-end Vision-Language-Action model.
5. **100% success in controlled setting**: Perfect scores reflect controlled
   2-block stacking with fixed object conventions, not open-world
   language understanding or generalisation.
6. **Parse50 benchmark name**: The input file contains 51 records (not 50).
   The benchmark name "Parse50" is retained for historical consistency;
   results should be cited as 51/51.

---

## 15. Next Steps

### Completed in v0.2.0

- [x] Parse50: 51-instruction evaluation (51/51 fully successful)
- [x] E2E10: 10-demo E2E validation (10/10 successful)
- [x] Dataset manifest: Standardized schema v0.2 with 9 components
- [x] Dataset card: Comprehensive documentation v0.2
- [x] `--instructions` external JSON loading for E2E runner
- [x] Standardized E2E summary with 29 fields
- [x] Test suite: 873 passed, 4 skipped

### Next

- [ ] Dataset manifest: Schema v1.0 with full dataset versioning
- [ ] E2E20: Scale E2E validation further
- [ ] Oracle scene inventory: Perception package for scene-aware parsing
- [ ] Failure taxonomy analysis for targeted prompt improvement
- [ ] 3-block symbolic planning (Robosuite 3-block E2E deferred)
