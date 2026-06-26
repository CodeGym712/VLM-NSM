# BC Endpoint Errors (EEF-to-Target Distance)

Distance in meters at phase endpoints. Policy rollout uses chained BC predictions within each phase.

| Phase       | N Endpoints | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 | Delta Mean |
| ----------- | ----------- | ----------- | ---------- | --------------- | -------------- | ---------- |
| reach_pick  | 20          | 0.1163      | 0.1209     | 10.5095         | 14.2326        | +10.3932   |
| grasp       | 20          | 0.0198      | 0.0200     | 45.9517         | 52.3387        | +45.9318   |
| lift        | 20          | 0.1301      | 0.1306     | 27.8487         | 28.8877        | +27.7186   |
| reach_place | 20          | 0.0567      | 0.0649     | 238.8265        | 257.2863       | +238.7697  |

## Interpretation

- **DS Err** = ground-truth dataset distance (how close the demonstrator got)
- **Policy Err** = BC policy rollout distance (how close BC gets)
- **Delta** = Policy - Dataset (positive = BC worse than demonstrator)
- Large delta on reach phases indicates the policy fails to converge to the target object.
