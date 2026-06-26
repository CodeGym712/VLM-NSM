# Phase Ablation

| BC-controlled Phases | Success Rate | Mean Steps | BC Steps | Scripted Steps | Main Failure |
|--------------------|:-----------:|:---------:|:--------:|:--------------:|:------------:|
| reach_pick only | 90% | 794 | 107 | 705 | verify_failed |
| reach_place only | 75% | 758 | 368 | 423 | horizon |
| Both reach (hybrid) | 95% | 741 | 510 | 248 | horizon |
| reach + grasp | 0% | 706 | 1002 | 0 | horizon |
| No BC (scripted-only) | 95% | 777 | 0 | 777 | horizon |
