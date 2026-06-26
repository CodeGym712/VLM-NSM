# VLM-NSM Demo Guide

> **Version**: v0.1.1 (Demo Validation Patch, based on v0.1-rc1)
> **Date**: 2026-05-17
> **Canonical Policy**: DAgger reach_pick-only hybrid control

---

## Quick Demo

Run the end-to-end language → simulation demo (5 instructions):

```bash
bash scripts/run_canonical_demo.sh
```

This runs:
1. Goal parsing (MockGoalParser)
2. Alignment and validation
3. Internal stacking planning
4. Robosuite execution with BC-controlled reach_pick + scripted grasp/lift/reach_place/release/retreat

**Expected output**: 5/5 demos successful.

**Output**: `generated/results/release_candidate/e2e_demo/summary.json`

---

## Quick Eval

Run a 20-episode canonical BC evaluation:

```bash
bash scripts/run_canonical_eval_20ep.sh
```

**Expected output**: Success rate consistent with 92.0% reference (90-96% range).

**Output**: `generated/results/release_candidate/reach_pick_20ep.json`

---

## Cached Qwen Replay

Run the demo using pre-computed Qwen2.5 parse results (no Qwen model needed):

```bash
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --parser cached_qwen \
  --cached_outputs generated/results/stage_12/cached_qwen_e2e_inputs.json \
  --output_dir generated/results/stage_12/e2e_cached_qwen_demo
```

**Expected output**: ≥4/5 sim success (cached Qwen outputs were validated offline).

---

## Recording Off-screen Demo Videos

Record MP4 videos of demo episodes without a display (requires GPU + Robosuite):

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --parser mock \
  --record_video \
  --video_fps 20
```

Options:

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--record_video` | off | Enable off-screen video recording |
| `--video_dir` | `{output_dir}/videos` | Output directory for MP4 files |
| `--video_fps` | 20 | Frames per second |
| `--video_max_episodes` | all | Limit number of episodes recorded |

Output filename format: `e2e_{parser}_{outcome}_ep{idx:03d}_v0_1_1.mp4`

Requires `MUJOCO_GL=egl` and OpenCV (`cv2.VideoWriter`). Video recording errors
do not affect episode execution (fails gracefully).

---

## Online Qwen Parser

Run the demo using real online Qwen2.5 API calls (requires API key):

```bash
QWEN_API_KEY=your_key_here \
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --parser online_qwen \
  --qwen_model qwen2.5-32b-instruct \
  --save_raw_qwen_outputs \
  --record_video
```

With fallback to mock parser if online Qwen parse fails:

```bash
QWEN_API_KEY=your_key_here \
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
python3 -m vlm_nsm.experiments.run_e2e_language_to_sim \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --bc_phases reach_pick \
  --parser online_qwen \
  --fallback_parser mock
```

### Parse-only evaluation (no Robosuite)

```bash
QWEN_API_KEY=your_key_here \
python3 -m vlm_nsm.experiments.eval_online_qwen_parser \
  --instructions data/language_variations/stacking_zh_en.jsonl \
  --output generated/results/online_qwen_eval_results.json \
  --model qwen2.5-32b-instruct \
  --available_objects cube1 cube2
```

Online Qwen parser notes:
- The `on(bottom, table)` predicate appears in parser output for alignment-layer
  completeness, but physical E2E success detection remains `on(top, bottom)` only.
- API key must be set via env var (default: `QWEN_API_KEY`).
- See `docs/stage_14_online_qwen_integration.md` for full documentation.

---

## DashScope Qwen Parser-Driven Demo

Run the demo using DashScope's OpenAI-compatible API (model `qwen3.6-plus`):

```bash
# Set up environment (gitignored — do not commit key)
cp configs/dashscope_qwen.env.example configs/dashscope_qwen.env
# Edit configs/dashscope_qwen.env with your real API key

# Smoke test: single parse
bash scripts/smoke_dashscope_qwen.sh

# Parse-only evaluation (5 instructions, no Robosuite)
bash scripts/run_dashscope_parse5.sh

# End-to-end demo (5 instructions, requires GPU + Robosuite)
MUJOCO_GL=egl bash scripts/run_dashscope_e2e_demo.sh

# Larger validation
bash scripts/run_dashscope_parse50.sh          # 50 parse-only
MUJOCO_GL=egl bash scripts/run_dashscope_e2e10.sh  # 10 E2E demos
```

The pipeline is identical to the online Qwen parser except for the API backend:

| Detail | Value |
|:-------|:------|
| Endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Model | `qwen3.6-plus` |
| Auth | `QWEN_API_KEY` env var |
| Temperature | 0.0 (deterministic) |
| Parser | `OnlineQwenGoalParser` (same class as Qwen2.5) |

**Key differences from Qwen2.5**:
- Uses DashScope's OpenAI-compatible endpoint — no separate SDK needed.
- `qwen3.6-plus` is faster with comparable structured JSON output quality.
- API key stored in `configs/dashscope_qwen.env` (gitignored).
- The parser-driven `target_stack` derivation (`target_stack_from_predicates`) is used, matching the E2E pipeline.

**Warning**: `configs/dashscope_qwen.env` is gitignored, but do not manually `git add` it.
Always use the `.env.example` template for reference. See `docs/stage_15_dashscope_qwen_parser_driven_eval.md` for full report.

---

## Reference Results

| Metric | Value | Source |
|:-------|:-----:|:-------|
| Canonical eval (100ep) | **92.0%** (92/100) | Stage 10 |
| Canonical eval (3×50ep avg) | **93.3%** (96/90/94%) | Stage 10 |
| E2E mock demo (5 demos) | **5/5 (100%)** | Stage 10 |
| Scripted baseline | 88.0% | Stage 9 |
| DAgger reach+place (research) | 83.3% avg | Stage 9 |

---

## Troubleshooting

### `MUJOCO_GL` not set

```bash
export MUJOCO_GL=egl    # headless server
export MUJOCO_GL=osmesa # alternative software rendering
```

### CUDA out of memory / no GPU

Set visible device and force CPU fallback:

```bash
export CUDA_VISIBLE_DEVICES=0
# or for CPU-only:
python3 ... --device cpu
```

Note: CPU inference is very slow. A GPU is recommended.

### `eval_bc.py` not found or wrong results

The old `eval_bc.py` manual loop is **deprecated**. Always use:

```bash
python3 -m vlm_nsm.learning.eval_bc_hybrid ...
```

The old script uses `horizon//6` phase division and zero-actions for non-BC phases — both are invalid.

### Tests fail

Run the full test suite (no GPU required):

```bash
python3 -m pytest tests/ -q --tb=short
```

Expected: **699+ passed, 2 skipped**. The 2 skipped tests require Robosuite.

### Robosuite seed limitation

`env.seed()` is not supported in Robosuite 1.5.2. The `--seed` flag is metadata only — environment initialization varies across runs.

### `on(bottom, table)` predicate

Do not add `on(bottom, table)` to stacking goals. Physical predicate extraction uses table center (0,0) as reference, and the bottom block drifts beyond the 0.05 xy threshold after stacking. The canonical `build_stacking_goal` does not include this predicate.

---

## Key Files

| File | Purpose |
|:-----|:--------|
| `scripts/run_canonical_demo.sh` | One-command E2E demo |
| `scripts/run_canonical_eval_20ep.sh` | One-command BC eval |
| `docs/release_candidate_manifest.md` | Frozen artifact manifest |
| `docs/stage_10_deployment_feasibility_report.md` | Full deployment report |
| `docs/simulation_protocol.md` | Canonical simulation protocol |
| `docs/release_notes_v0.1.md` | v0.1-rc1 release notes |
| `docs/release_notes_v0.1.1.md` | v0.1.1 demo validation patch |
| `docs/stage_14_online_qwen_integration.md` | Online Qwen parser integration |
| `assets/demo/v0_1_1/demo_media_manifest.md` | Demo media status |
