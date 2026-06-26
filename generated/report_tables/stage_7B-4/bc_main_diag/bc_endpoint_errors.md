# BC Endpoint Errors (EEF-to-Target Distance)

Distance in meters at phase endpoints. Policy rollout uses chained BC predictions within each phase.

| Phase       | N Endpoints | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 | Delta Mean |
| ----------- | ----------- | ----------- | ---------- | --------------- | -------------- | ---------- |
| reach_pick  | 20          | 0.1163      | 0.1209     | 11.6821         | 16.1333        | +11.5658   |
| grasp       | 20          | 0.0198      | 0.0200     | 8.3564          | 9.3553         | +8.3366    |
| lift        | 20          | 0.1301      | 0.1306     | 8.3219          | 8.6363         | +8.1918    |
| reach_place | 20          | 0.0567      | 0.0649     | 14.2559         | 19.6898        | +14.1991   |

## Interpretation

- **DS Err** = ground-truth dataset distance (how close the demonstrator got)
- **Policy Err** = BC policy rollout distance (how close BC gets)
- **Delta** = Policy - Dataset (positive = BC worse than demonstrator)
- Large delta on reach phases indicates the policy fails to converge to the target object.
