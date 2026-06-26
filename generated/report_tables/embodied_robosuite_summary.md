# Embodied Robosuite Evaluation Summary

- Task: stacking_robosuite
- Environment: Stack (n_blocks=2)
- Robot: Panda
- Controller: proportional_osc_pose
- Episodes: 50
- Early stop: True

## Classic Metrics

- Episode success rate: 92.0%
- Step success rate: 0.7%
- Skill success rate: 0.0%
- Goal success rate: 92.0%
- Predicate match rate: 92.0%

## Stage 7A-7 Metrics

- Semantic goal success rate: 92.0%
- Strict physical success rate: 92.0%
- Strict reality gap: 8.0
- Semantic reality gap: 8.0
- Mean steps: 787
- P95 steps: 815
- Timeout after goal: 0
- Timeout before goal: 2
- Predicate thresholds: {'xy_threshold': 0.05, 'z_min': -0.005, 'z_max': 0.05}
- Failure stages: {"preplace": 2, "horizon": 2}
