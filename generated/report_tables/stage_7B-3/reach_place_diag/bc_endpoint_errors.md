# BC Endpoint Errors (EEF-to-Target Distance)

Distance in meters at phase endpoints. Policy rollout uses chained BC predictions within each phase.

| Phase       | N Endpoints | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 | Delta Mean |
| ----------- | ----------- | ----------- | ---------- | --------------- | -------------- | ---------- |
| reach_pick  | 20          | 0.1163      | 0.1209     | 16.9553         | 24.5821        | +16.8390   |
| grasp       | 20          | 0.0198      | 0.0200     | 29.5199         | 41.0590        | +29.5001   |
| lift        | 20          | 0.1301      | 0.1306     | 29.7842         | 37.3815        | +29.6541   |
| reach_place | 20          | 0.0567      | 0.0649     | 12.2054         | 16.7361        | +12.1487   |

## Interpretation

- **DS Err** = ground-truth dataset distance (how close the demonstrator got)
- **Policy Err** = BC policy rollout distance (how close BC gets)
- **Delta** = Policy - Dataset (positive = BC worse than demonstrator)
- Large delta on reach phases indicates the policy fails to converge to the target object.
