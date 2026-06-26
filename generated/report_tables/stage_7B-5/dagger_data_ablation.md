# Data Ablation: Training Source Comparison

| Training Data | Episodes | Steps | Success Rate | Mean Steps |
|--------------|:--------:|:----:|:------------:|:----------:|
| Expert-only (bc_main) | 100 | 61,618 | 86.7% | 744 |
| DAgger-only | 30 | 18,820 | 0% | 704 |
| Expert + DAgger (ours) | 130 | 80,438 | 93.3% | 738 |
