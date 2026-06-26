# Language Failure Taxonomy

Total failures: 556 across 556 samples (428 successes).

### By Failure Type

| Failure Type | Count | % of Total | Description |
|:------------|:-----:|:----------:|-------------|
| schema_invalid | 294 | 52.9% | Output does not match required JSON schema |
| invalid_object | 157 | 28.2% | References object not present in the environment |
| goal_not_satisfied | 81 | 14.6% | Parsed goal does not satisfy task constraints |
| json_parse_error | 14 | 2.5% | Model output is not valid JSON |
| incomplete_goal | 10 | 1.8% | Goal is missing required fields |

### By Model

| Model | schema_invalid | invalid_object | goal_not_satisfied | json_parse_error | incomplete_goal | Total |
|-------|:---:|:---:|:---:|:---:|:---:|:----:|
| qwen25_14b | 84 | 21 | 42 | 4 | 6 | 157 |
| qwen25_32b | 210 | 136 | 39 | 10 | 4 | 399 |

### By Task

| Task | schema_invalid | invalid_object | goal_not_satisfied | json_parse_error | incomplete_goal | Total |
|------|:---:|:---:|:---:|:---:|:---:|:----:|
| hanoi | 145 | 87 | 40 | 9 | 0 | 281 |
| stacking | 149 | 70 | 41 | 5 | 10 | 275 |
