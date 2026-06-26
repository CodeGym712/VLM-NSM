# Demo Assets

This directory stores demo videos and screenshots for the VLM-NSM release candidate.

## Expected Contents

| File | Description |
|:-----|:------------|
| `e2e_demo_success.mp4` | Screen recording of one successful E2E language→sim demo |
| `e2e_demo_steps.mp4` | Step-by-step breakdown of the 6 phases (reach_pick → grasp → lift → reach_place → release → retreat) |
| `stacking_result.png` | Final stacked state showing cube1 on cube2 |

## How to Generate

Videos can be recorded by setting `render_mode="screen"` in `RobosuiteEnvWrapper` and using a screen capture tool, or by adding `--record_video` support to `run_e2e_language_to_sim.py`.

Current `run_e2e_language_to_sim.py` uses `render_mode="none"` (headless). To record:

1. Change `render_mode` to `"screen"` in the demo script (line 278 of `run_e2e_language_to_sim.py`)
2. Use a screen recorder (e.g., `ffmpeg -f x11grab`) during execution
3. Place the resulting files in this directory

## References

- v0.1.1 demo media: `assets/demo/v0_1_1/demo_media_manifest.md`
- Demo results: `generated/results/stage_10/e2e_sim_demo/summary.json`
- Demo summary: `generated/report_tables/stage_10/e2e_demo_summary.md`
- Demo guide: `docs/demo_guide.md`
- Release notes v0.1-rc1: `docs/release_notes_v0.1.md`
- Release notes v0.1.1: `docs/release_notes_v0.1.1.md`
- Deployment report: `docs/stage_10_deployment_feasibility_report.md`
