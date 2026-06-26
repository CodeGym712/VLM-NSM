# Demo Media Manifest — v0.1.1

## Status

| Item | Status | Notes |
|:-----|:------|:------|
| Mock demo video | ✅ Supported via `--record_video` | Off-screen rendering, MUJOCO_GL=egl required |
| Cached Qwen demo video | ✅ Supported via `--record_video` | Off-screen rendering, MUJOCO_GL=egl required |
| Stacking result screenshot | ⏳ Pending | No screenshot capture implemented yet |

**Date**: 2026-05-17
**Method**: Off-screen video recording via `render_mode="offscreen"` with MUJOCO_GL=egl.
The `--record_video` flag on `run_e2e_language_to_sim.py` captures frames during
episode execution and writes MP4 files via imageio. See the [demo guide](../../docs/demo_guide.md)
for usage.

## Generated Files

When recorded with `--record_video`, MP4 files are written to `{output_dir}/videos/`:

```
{output_dir}/videos/
├── e2e_mock_success_ep001_v0_1_1.mp4         mock parser, episode 1
├── e2e_mock_success_ep002_v0_1_1.mp4         mock parser, episode 2
├── e2e_cached_qwen_success_ep001_v0_1_1.mp4  cached Qwen parser, episode 1
└── ...
```

Filename format: `e2e_{parser}_{outcome}_ep{idx:03d}_v0_1_1.mp4`

## Source Results

| Media | Parser | Episode | Expected Outcome |
|:------|:-------|:--------|:----------------|
| e2e_mock_success_ep001_v0_1_1.mp4 | mock | 1 | 5/5 reference |
| e2e_cached_qwen_success_ep001_v0_1_1.mp4 | cached_qwen | 1 | ≥4/5 expected |

## References

- GPU pending note: `generated/results/stage_13/gpu_pending_note.md`
- Release notes: `docs/release_notes_v0.1.1.md`
