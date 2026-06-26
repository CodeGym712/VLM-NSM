# Embodied Policy Comparison

| Policy | seed=0 | seed=1 | seed=2 | Avg | Min | Mean Steps | Mean BC Steps |
|--------|:-----:|:-----:|:-----:|:---:|:---:|:----------:|:-------------:|
| Scripted (scripted-only) | 95% | 100% | 80% | 91.7% | 80% | 785 | 0 |
| bc_main (expert-only BC) | 80% | 85% | 95% | 86.7% | 80% | 744 | 463 |
| DAgger Reach BC (ours) | 95% | 90% | 95% | 93.3% | 90% | 738 | 498 |
| DAgger Reach BC (50ep) | — | — | — | 92.0% | — | 734 | 503 |
