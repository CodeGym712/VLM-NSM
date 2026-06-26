# Expert v2 Dataset Recovery Summary

**Date:** 2026-05-15
**Collection method:** ``collect_until_strict``

| Metric | Value |
|:---|---:|
| Target strict demos | 100 |
| Attempts | 107 |
| Strict success | 100 |
| Strict success rate | 93.5% |
| Semantic-only success | 7 |
| Failures | 7 |
| **Efficiency ratio** | 93.5% |

### Failure Breakdown
| Type | Count |
|:---|---:|
| horizon | 4 |
| grasp_failed | 2 |
| preplace | 1 |

### Output Structure
**Root:** ``/rhome/zhaowm/my_project/VLM-NSM/generated/demos/stage_7A-14/robosuite_expert_v2_100strict``

| Directory | Contents |
|:---|---:|
| ``strict_success/`` | 107 npz files (100 strict + 7 semantic-only) |
| ``strict_only/`` | **100 strict-only npz (recommended for training)** |
| ``failure_debug/`` | 24 npz files (failures + intermediate retries) |
| ``collection_summary.json`` | Full episode-level summary |
| ``trajectories_strict_only.npz`` | Merged trajectory file (85,396 steps, 100 eps) |

### Dataset Schema (per-episode npz)
| Key | Shape | Description |
|:---|---:|:---|
| ``filtered_obs`` | (T, 11) | Filtered observations |
| ``obs`` | (T, 11) | Raw observations |
| ``action`` | (T, 7) | End-effector delta + gripper |
| ``phase_id`` | (T,) | True phase ID [0-5] |
| ``skill_id`` | (T,) | Skill ID [0-3] |
| ``episode_id`` | (T,) | Episode index |
| ``step_id`` | (T,) | Global step index |
| ``done`` | (T,) | Episode done flags |
| ``success`` | (T,) | Step success flags |

### True Phase ID Coverage
| Phase | ID | Samples |
|:---|---:|---:|
| reach_pick | 0 | 11616 |
| grasp | 1 | 11864 |
| lift | 2 | 11914 |
| reach_place | 3 | 50002 |
| release | 4 | — (not in Stack task) |
| retreat | 5 | — (not in Stack task) |