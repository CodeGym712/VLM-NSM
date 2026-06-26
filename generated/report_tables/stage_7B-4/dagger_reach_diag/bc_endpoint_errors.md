# BC Endpoint Errors (EEF-to-Target Distance)

Distance in meters at phase endpoints. Policy rollout uses chained BC predictions within each phase.

| Phase       | N Endpoints | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 | Delta Mean |
| ----------- | ----------- | ----------- | ---------- | --------------- | -------------- | ---------- |
| reach_pick  | 20          | 0.1163      | 0.1209     | 10.8933         | 14.2548        | +10.7770   |
| grasp       | 20          | 0.0198      | 0.0200     | 32.5923         | 45.7500        | +32.5725   |
| lift        | 20          | 0.1301      | 0.1306     | 42.5380         | 49.7555        | +42.4079   |
| reach_place | 20          | 0.0567      | 0.0649     | 11.8891         | 15.5621        | +11.8324   |

## Interpretation

- **DS Err** = ground-truth dataset distance (how close the demonstrator got)
- **Policy Err** = BC policy rollout distance (how close BC gets)
- **Delta** = Policy - Dataset (positive = BC worse than demonstrator)
- Large delta on reach phases indicates the policy fails to converge to the target object.
