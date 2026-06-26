# DAgger Data Ablation

| Training Data | Episodes | Samples | Success Rate (avg) | Mean Steps | Notes |
|:-------------|:--------:|:-------:|:-----------------:|:----------:|-------|
| Expert-only (bc_main) | 100 | 61,618 | 86.7% | 744 | Baseline expert BC |
| DAgger-only | 30 | 18,820 | 0% | 704 | Collapses — offline val loss 0.000627 but closed-loop 0% |
| Expert + DAgger (ours) | 130 | 80,438 | 93.3% | 738 | +6.6pp over expert-only — DAgger improves state coverage |
