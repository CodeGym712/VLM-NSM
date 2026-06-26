# Deployment Policy Comparison

| Policy | N | Success | Steps | BC | P95 | PP | HZ | VF | GF |
|:------|:-:|:------:|:----:|:--:|:---:|:--:|:--:|:--:|:--:|
| Scripted-only | 100 | 88.0% | 781 | 0 | 816 | 0 | 8 | 2 | 1 |
| DAgger reach+place (seed0) | 50 | 82.0% | 744 | 522 | 1000 | 47 | 1 | 1 | 0 |
| DAgger reach+place (seed1) | 50 | 82.0% | 735 | 483 | 911 | 41 | 4 | 1 | 1 |
| DAgger reach+place (seed2) | 50 | 86.0% | 734 | 482 | 910 | 43 | 4 | 1 | 0 |
| DAgger reach-only (100ep) | 100 | 92.0% | 795 | 113 | 856 | 0 | 6 | 2 | 0 |
| DAgger reach-only seed0 | 50 | 96.0% | 794 | 106 | 856 | 1 | 0 | 1 | 0 |
| DAgger reach-only seed1 | 50 | 90.0% | 802 | 120 | 1000 | 1 | 1 | 3 | 0 |
| DAgger reach-only seed2 | 50 | 94.0% | 799 | 108 | 848 | 0 | 1 | 1 | 0 |
| _DAgger reach+place avg_ | 150 | 83.3% | — | — | — | 43.7 | — | — | — |
| _DAgger reach-only avg_ | 150 | 93.3% | — | — | — | 0 | — | — | — |

### Legend
- PP = preplace (place verification failed)
- HZ = horizon (timeout)
- VF = verify_failed
- GF = grasp_failed

### Recommendation

| Policy | Verdict | Rationale |
|:------|:-------:|:----------|
| **DAgger reach_pick-only** | ✅ **Canonical default** | 92% on 100ep, 0 preplace failures |
| DAgger reach_pick+reach_place | ⚠️ Research variant | Higher BC step count but preplace failures degrade reliability |
| Scripted-only | 📊 Baseline | No learned control, used as ceiling reference |