# Canonical Simulation Comparison

| Config | N | Success | Mean Steps | Mean BC | Mean Scripted | P95 Steps | HZ | PP | VF |
|:------|:-:|:------:|:---------:|:-------:|:------------:|:--------:|:--:|:--:|:--:|
| Stage 8 DAgger 100ep (seed=0) | 100 | 80.0% | 752 | 511 | 262 | 972 | 7 | 82 | 3 |
| Scripted Canonical 100Ep | 100 | 88.0% | 781 | 0 | 781 | 816 | 8 | 0 | 2 |
| Dagger Reach Seed0 50Ep | 50 | 82.0% | 744 | 522 | 228 | 1000 | 1 | 47 | 1 |
| Dagger Reach Seed1 50Ep | 50 | 82.0% | 735 | 483 | 277 | 911 | 4 | 41 | 1 |
| Dagger Reach Seed2 50Ep | 50 | 86.0% | 734 | 482 | 277 | 910 | 4 | 43 | 1 |

### Legend
- HZ = horizon (timeout)
- PP = preplace (place verification failed)
- VF = verify_failed (goal predicate mismatch)

### Key Questions
1. **Scripted baseline**: What is the ceiling for scripted-only execution?
2. **DAgger canonical 3x50ep**: Does multi-seed average match Stage 7B-5 (93.3%) or Stage 8 (80.0%)?
3. **reach_pick-only diagnostic**: Does limiting BC to reach_pick improve reach_place (preplace) outcomes?