# Artifact Manifest

| Category | Path | Description |
|----------|:----:|-------------|
| **Configs** | | |
| Execution config | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` | Best execution parameters (kp, max_delta, state machines) |
| **Expert Demos** | | |
| Expert v2 strict_only (npz) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/trajectories_strict_only.npz` | 100 strict expert demos with true phase_id |
| Expert v2 strict_only (dir) | `generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/` | Per-episode npz files for BC training |
| **DAgger Datasets** | | |
| DAgger round 1 | `generated/demos/stage_7B-4/dagger_round1_reach/trajectories_dagger_round1_reach.npz` | 18,820 on-policy DAgger samples |
| Aggregated reach dataset | `generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz` | Expert + DAgger merged (80,438 steps) |
| **Checkpoints** | | |
| DAgger Reach BC | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` | Best learned controller: DAgger-trained, phase-conditioned, 72K params |
| bc_main (expert-only) | `generated/results/stage_7B-3/bc_main/best.pt` | Expert-only BC baseline |
| DAgger-only BC | `generated/results/stage_7B-5/dagger_only_bc/best.pt` | DAgger-only ablative (0% closed-loop) |
| **Embodied Eval Results** | | |
| Multi-seed comparison | `generated/results/stage_7B-5/` | Scripted/bc_main/DAgger over 3 seeds × 20ep |
| DAgger 50ep validation | `generated/results/stage_7B-5/dagger_reach_50ep.json` | 92.0% over 50 episodes |
| Phase ablation | `generated/results/stage_7B-5/phase_*.json` | Phase ablation eval results |
| Data ablation | `generated/results/stage_7B-5/dagger_only_bc_eval_20ep.json` | DAgger-only BC eval (0%) |
| **Language Eval Results** | | |
| Summary | `generated/results/language_variation_summary.json` | 984 samples, 282 groups |
| Failure cases | `generated/results/qwen_failure_cases.json` | 556 failure cases with metadata |
| Alignment repair stats | `generated/results/qwen_alignment_repair_stats.json` | 9 repairs tracked |
| **Report Tables** | | |
| Stage 7B-5 tables | `generated/report_tables/stage_7B-5/` | Multi-seed, data ablation, phase ablation, failure audit |
| Final tables | `generated/report_tables/final/` | Integrated final report tables |
| **Documentation** | | |
| Stage 7B-5 report | `docs/bc_stage7b5_report.md` | DAgger robust validation |
| Final empirical report | `docs/final_empirical_report.md` | Integrated language + embodied report |
| Evaluation protocol | `docs/evaluation_protocol.md` | Canonical and deprecated eval paths |
| Artifact manifest | `docs/artifact_manifest.md` | This file |
| **Tests** | | |
| Stage 7B-5 tests | `tests/test_stage7b5_report_export.py` | Verify report tables and JSON results |
| Policy comparison tests | `tests/test_policy_comparison_table.py` | Multi-seed policy validation |
| Phase ablation tests | `tests/test_phase_ablation_table.py` | Phase ablation validation |
| Failure audit tests | `tests/test_dagger_failure_audit.py` | Failure analysis tests |
| Final report tests | `tests/test_final_report_tables.py` | Verify final Stage 8 tables |
