# Dataset Summary

| Dataset | Episodes | Steps | Source | Used In |
|---------|:--------:|:----:|:------|:--------|
| Expert v2 strict_only | 100 (in 107 attempts) | 85396 | Scripted policy, true phase_id | bc_main, DAgger aggregation |
| DAgger round 1 (reach) | 30 | 18,820 | On-policy BC rollout + oracle actions | DAgger Reach BC training |
| Aggregated reach dataset | 130 | 80438 | Expert v2 + DAgger merged | DAgger Reach BC (best model) |

Expert v2 collection efficiency: 93.5% strict success rate
(93 strict + 7 semantic-only in 107 attempts = 100 valid strict demos).
