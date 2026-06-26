# Stage 14: Online Qwen Parser Integration

> **Date**: 2026-05-18
> **Status**: Complete

---

## Motivation

Move from cached/replay-based Qwen evaluation to real online Qwen API
calls in the E2E language-to-simulation pipeline. This enables:

- End-to-end testing with actual Qwen2.5 reasoning (not cached results)
- Realistic latency and error behavior characterization
- Seamless integration with the existing Alignment → Planner → Robosuite flow

---

## Scope

**Parser integration only.** No changes to:
- Canonical checkpoint (`stage_7B-4/dagger_reach_bc/best.pt`) ❄️
- Canonical config (`stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`) ❄️
- Controller logic (DAgger reach_pick-only hybrid) ❄️
- Planner action order (`[obj, "table", target]`) ❄️
- `build_stacking_goal` semantics (`on(top, bottom)` only, no table predicate) ❄️
- BC phases (reach_pick only) ❄️

The table predicate (`on(bottom, table)`) may appear in the parser output
for alignment-layer completeness compatibility, but physical E2E success
verification still uses `on(top, bottom)` only.

---

## Parser Modes

Three parser modes are now available in `run_e2e_language_to_sim.py`:

| Mode | CLI Value | Description |
|:-----|:----------|:------------|
| Mock | `--parser mock` | Deterministic, no LLM (default) |
| Cached Qwen | `--parser cached_qwen` | Pre-computed Qwen results from JSON |
| Online Qwen | `--parser online_qwen` | Real Qwen2.5 API call per instruction |

---

## Prompt Schema

The online Qwen prompt (`STACKING_PROMPT` in `online_qwen_parser.py`):

```
You are a robot instruction parser. Convert natural language instructions
into symbolic goal JSON.

Task: stacking blocks
Available objects: cube1, cube2
Valid predicates: on(object, object_or_table)
Valid object IDs: cube1, cube2, table

Output JSON schema:
{
  "task": "stacking",
  "goal_predicates": [
    ["on", "<top_object>", "<bottom_object>"],
    ["on", "<bottom_object>", "table"]
  ]
}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no code fences.
- Do NOT invent object IDs beyond those listed above.
- If the instruction is ambiguous, choose the most likely mapping.
- For two-block stacking, always include bottom support predicate.

Examples:
...
```

### Completeness Note

The `on(bottom, table)` predicate is included in the prompt for
alignment-layer completeness checking. Physical goal detection in
`build_stacking_goal` does NOT use the table predicate — the bottom
block drifts during stacking, making physical detection unreliable.
This is consistent with the v0.1-rc1 frozen behavior.

---

## Failure Handling

1. **Parse failure** (invalid JSON response): retry up to `--qwen_retries` times.
2. **Schema failure** (missing required fields): caught by AlignmentLayer.
3. **API error** (connection, auth, timeout): raised immediately.
4. **Fallback**: if `--fallback_parser` is set, the system retries
   parsing with mock or cached_qwen when the primary parser fails.

The key step is read from `QWEN_API_KEY` env var by default (configurable
via `--qwen_api_key_env`).

---

## Commands

### Parse-only evaluation (no Robosuite)

```bash
python3 -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions data/language_variations/stacking_zh_en.jsonl \
  --output generated/results/online_qwen_eval_results.json \
  --model qwen2.5-32b-instruct \
  --available_objects cube1 cube2
```

### E2E language-to-simulation with online Qwen

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
QWEN_API_KEY=your_key_here \
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --parser online_qwen \
  --qwen_model qwen2.5-32b-instruct \
  --save_raw_qwen_outputs
```

### With fallback to cached Qwen

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
QWEN_API_KEY=your_key_here \
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --parser online_qwen \
  --fallback_parser mock
```

---

## CLI Reference

### New args for `run_e2e_language_to_sim.py`

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--parser` | `mock` | One of: `mock`, `cached_qwen`, `online_qwen` |
| `--qwen_model` | `qwen2.5-32b-instruct` | Model name for API |
| `--qwen_api_base` | None | Custom API base URL |
| `--qwen_api_key_env` | `QWEN_API_KEY` | Env var for API key |
| `--qwen_temperature` | 0.0 | Sampling temperature |
| `--qwen_max_tokens` | 512 | Max output tokens |
| `--qwen_timeout` | 60 | API timeout in seconds |
| `--qwen_retries` | 1 | Parse retry count |
| `--save_raw_qwen_outputs` | False | Save raw model outputs |
| `--fallback_parser` | `none` | Fallback: `none`, `mock`, `cached_qwen` |

### New args for `eval_online_qwen_parser.py`

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--instructions` | (required) | Path to instructions file |
| `--output` | `.../online_qwen_eval_results.json` | Output path |
| `--model` | `qwen2.5-32b-instruct` | Qwen model name |
| `--api_base` | None | API base URL |
| `--qwen_api_key_env` | `QWEN_API_KEY` | Env var name |
| `--temperature` | 0.0 | Sampling temperature |
| `--max_tokens` | 512 | Max output tokens |
| `--timeout` | 60 | API timeout |
| `--retries` | 1 | Parse retry count |
| `--available_objects` | `cube1 cube2` | Valid object IDs |
| `--save_raw_qwen_outputs` | False | Save raw outputs |
| `--no_alignment` | False | Skip alignment/planner validation |

---

## Implementation

### New files

| File | Purpose |
|:-----|:--------|
| `src/vlm_nsm/language/__init__.py` | Module init |
| `src/vlm_nsm/language/online_qwen_parser.py` | `OnlineQwenGoalParser` class |
| `src/vlm_nsm/experiments/eval_online_qwen_parser.py` | Parse-only eval script |
| `tests/test_json_extraction.py` | JSON extraction tests |
| `tests/test_online_qwen_prompt.py` | Prompt content tests |
| `tests/test_online_qwen_cli_args.py` | CLI arg tests |
| `tests/test_eval_online_qwen_parser_script.py` | Eval script tests |

### Modified files

| File | Change |
|:-----|:--------|
| `src/vlm_nsm/experiments/run_e2e_language_to_sim.py` | Added online_qwen parser mode, CLI args, fallback |

---

## Known Limitations

1. **API nondeterminism**: Unlike mock/cached parsers, online Qwen may
   produce different outputs across calls even with temperature=0.0.
2. **Latency**: Each instruction incurs API round-trip time (~1-10s).
3. **Schema errors**: Online Qwen may output malformed JSON or
   invalid predicates despite prompt instructions.
4. **Object grounding**: The parser has no visual input — it only
   uses object ID lists. Ambiguous references may be mis-resolved.
5. **API key required**: The `QWEN_API_KEY` env var must be set.
   Fail on missing key with actionable error.

---

## Test Status

Tests cover:
- JSON extraction (pure JSON, fenced, text, invalid)
- Prompt content (schema, predicates, objects, few-shot, completeness note)
- CLI args for both E2E script and eval script
- Eval script structure (functions, imports)
