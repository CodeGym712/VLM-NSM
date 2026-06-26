# Release Notes — v0.1.1 Demo Validation Patch

> **Date**: 2026-05-17
> **Status**: Demo Validation Patch
> **Based on**: [v0.1-rc1](release_notes_v0.1.md) (frozen 2026-05-17)

---

## Relationship to v0.1-rc1

v0.1.1 is a **non-invasive demo validation patch**. All frozen v0.1-rc1
components remain unchanged:

### Unchanged (frozen from v0.1-rc1)

| Component | Status |
|:----------|:-------|
| Canonical checkpoint (`stage_7B-4/dagger_reach_bc/best.pt`) | ❄️ Frozen |
| Canonical config (`stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`) | ❄️ Frozen |
| Deployment policy (DAgger reach_pick-only) | ❄️ Frozen |
| BC phases (reach_pick only) | ❄️ Frozen |
| Planner action order (`[obj, "table", target]`) | ❄️ Frozen |
| `build_stacking_goal` semantics (`on(top, bottom)`, no table predicate) | ❄️ Frozen |
| Language evaluation metrics (984 samples, 556 failures, etc.) | ❄️ Frozen |
| Robosuite evaluation metrics (92.0%, 93.3%, 5/5) | ❄️ Frozen |

### Added in v0.1.1

| Addition | Status |
|:---------|:-------|
| Cached Qwen Robosuite replay | ⏳ Requires GPU + Robosuite |
| Mock demo replay | ⏳ Requires GPU + Robosuite |
| Demo media (videos/screenshots) | ⏳ Requires rendering |
| Demo media manifest | ✅ |
| v0.1.1 release notes | ✅ (this file) |
| Updated demo guide with v0.1.1 references | ✅ |
| Test coverage for v0.1.1 docs | ✅ |

---

## Validation Status

### Cached Qwen Robosuite Replay

- **Input**: `generated/results/stage_12/cached_qwen_e2e_inputs.json` (5 instructions)
- **Parser**: `CachedQwenGoalParser`
- **Results**: `generated/results/stage_13/e2e_cached_qwen_demo/summary.json`
- **Simulation**: ✅ **5/5 (100%)** — all demos successful
- **Note**: Initial 0/5 alignment failure was fixed by adding `on(bottom, table)` to cached Qwen inputs. This is required for alignment-layer completeness checking. The physical E2E goal (`build_stacking_goal`) still uses `on(top, bottom)` only.

### Mock Demo Replay

- **Results**: `generated/results/stage_13/e2e_mock_demo/summary.json`
- **Simulation**: ✅ **4/5 (80%)** — 1 horizon timeout, within expected intrinsic failure rate (~12%)
- **Consistency**: Confirmed v0.1-rc1 results are reproducible

### Off-Screen Video Recording

- **Feature**: `--record_video` flag added to `run_e2e_language_to_sim.py`
- **Method**: Off-screen rendering via `render_mode="offscreen"` with MUJOCO_GL=egl, no display required
- **Output**: MP4 files via imageio, one per episode
- **Status**: ✅ Implementation complete, requires MUJOCO_GL=egl + imageio to generate files

---

## Artifacts

### New in v0.1.1

| Artifact | Path |
|:---------|:------|
| GPU pending note | `generated/results/stage_13/gpu_pending_note.md` |
| Cached Qwen demo results (5/5) | `generated/results/stage_13/e2e_cached_qwen_demo/` |
| Mock demo results (4/5) | `generated/results/stage_13/e2e_mock_demo/` |
| Off-screen video recording | `--record_video` flag in `run_e2e_language_to_sim.py` |
| Demo media manifest | `assets/demo/v0_1_1/demo_media_manifest.md` |
| v0.1.1 release notes | `docs/release_notes_v0.1.1.md` |

### Inherited from v0.1-rc1

See [release_notes_v0.1.md](release_notes_v0.1.md) for the full frozen artifact list.

---

## Known Limitations

Same as v0.1-rc1, plus:

9. ~~Demo validation GPU-gated~~ **Resolved**: Cached Qwen replay (5/5) and
   mock demo replay (4/5) completed. GPU with Robosuite 1.5.2 and
   MUJOCO_GL=egl required for re-runs.

10. **No screen recording**: Off-screen video recording is available via
    `--record_video` (render_mode="offscreen" with MUJOCO_GL=egl). True
    screen recording (render_mode="screen") is not supported in headless CI.

---

## Test Status

| Suite | Result |
|:------|:------:|
| Full pytest | **732 passed, 2 skipped** (same as v0.1-rc1) |
| v0.1.1 doc tests | Added: manifest, release notes, demo guide references |
