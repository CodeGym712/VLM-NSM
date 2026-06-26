# reach_pick-only 100ep Summary

**Source**: `generated/results/stage_9/dagger_reach_pick_only_100ep.json`
**Model**: `stage_7B-4/dagger_reach_bc/best.pt`
**bc_phases**: `reach_pick` (grasp/lift/release/retreat = scripted)
**Seed**: 0

## Results

| Metric | Value |
|:-------|:-----:|
| Success rate | **92.0%** (92/100) |
| Mean steps | 795 |
| Mean BC steps | 113 |
| Mean scripted steps | 702 |
| P95 steps | 856 |

## Failure Types

| Type | Count |
|:----|:-----:|
| horizon | 6 |
| verify_failed | 2 |
| preplace | **0** |

## Key Finding

**No preplace failures** — limiting BC to reach_pick eliminates the preplace failure mode entirely.
Compared to DAgger reach_pick+reach_place (83.3% avg), reach_pick-only gains **~9pp** by avoiding
BC-induced noise during the place phase.
