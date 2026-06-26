# Robosuite Failure Audit

## Demo Collection Failures

- **Config**: kp=0.08, max_delta=0.015, close_steps=15
- **Total episodes**: 100
- **Failures**: 11

### Failure Type Breakdown

| Failure Type  | Count | Percentage |
| ------------- | ----- | ---------- |
| horizon       | 7     | 63.6%      |
| grasp_failed  | 2     | 18.2%      |
| preplace      | 1     | 9.1%       |
| verify_failed | 1     | 9.1%       |

### Per-Episode Detail

| Episode | Failure Type  | Steps | Initial Cube1 Z | Initial Cube2 Z |
| ------- | ------------- | ----- | --------------- | --------------- |
| 1       | horizon       | 657   | 0.830           | 0.835           |
| 6       | preplace      | 1000  | 0.830           | 0.835           |
| 11      | horizon       | 684   | 0.830           | 0.835           |
| 15      | horizon       | 687   | 0.830           | 0.835           |
| 19      | verify_failed | 1000  | 0.830           | 0.835           |
| 27      | horizon       | 665   | 0.830           | 0.835           |
| 29      | grasp_failed  | 755   | 0.830           | 0.835           |
| 55      | horizon       | 670   | 0.830           | 0.835           |
| 56      | horizon       | 677   | 0.830           | 0.835           |
| 91      | horizon       | 678   | 0.830           | 0.835           |
| 98      | grasp_failed  | 736   | 0.830           | 0.835           |

## Eval Run Failures

- **Total episodes**: 50
- **Failures**: 4

| Failure Type | Steps | Phase Trace                                       |
| ------------ | ----- | ------------------------------------------------- |
| preplace     | 1000  | grasp=113 lift=101 reach_pick=113 reach_place=673 |
| preplace     | 1000  | grasp=116 lift=101 reach_pick=96 reach_place=687  |
| horizon      | 677   | grasp=565 reach_pick=112                          |
| horizon      | 669   | grasp=565 reach_pick=104                          |

### Aggregate Failure Patterns

| Pattern  | Count | Typical Steps | Root Cause                         |
| -------- | ----- | ------------- | ---------------------------------- |
| preplace | 2     | 1000          | Place SM cannot converge to target |
| horizon  | 2     | ~565          | Grasp timeout (565 steps in CLOSE) |
