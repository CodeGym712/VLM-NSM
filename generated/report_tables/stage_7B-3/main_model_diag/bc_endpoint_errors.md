# BC Endpoint Errors (EEF-to-Target Distance)

Distance in meters at phase endpoints. Policy rollout uses chained BC predictions within each phase.

| Phase       | N Endpoints | DS Err Mean | DS Err P95 | Policy Err Mean | Policy Err P95 | Delta Mean |
| ----------- | ----------- | ----------- | ---------- | --------------- | -------------- | ---------- |
| reach_pick  | 20          | 0.1163      | 0.1209     | 10.6615         | 14.3394        | +10.5453   |
| grasp       | 20          | 0.0198      | 0.0200     | 7.8198          | 8.8061         | +7.8000    |
| lift        | 20          | 0.1301      | 0.1306     | 9.0790          | 9.3471         | +8.9488    |
| reach_place | 20          | 0.0567      | 0.0649     | 10.6030         | 14.5968        | +10.5462   |

## Interpretation

- **DS Err** = ground-truth dataset distance (how close the demonstrator got)
- **Policy Err** = BC policy rollout distance (how close BC gets)
- **Delta** = Policy - Dataset (positive = BC worse than demonstrator)
- Large delta on reach phases indicates the policy fails to converge to the target object.
