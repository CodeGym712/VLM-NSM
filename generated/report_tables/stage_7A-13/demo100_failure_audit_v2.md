# Demo 100 Failure Audit v2

- **Collection summary**: `generated/demos/stage_7A-12/collection_summary.json`
- **Total episodes**: 100
- **Failures**: 13
- **Success rate**: 87.0%

## Failure Type Summary

| Failure Type  | Count | Percentage |
| ------------- | ----- | ---------- |
| horizon       | 7     | 53.8%      |
| preplace      | 3     | 23.1%      |
| verify_failed | 2     | 15.4%      |
| grasp_failed  | 1     | 7.7%       |

## Phase Budget Analysis

## Per-Episode Failure Detail

| Episode | Failure Type  | Steps | Last Phase | Close Steps | Extended? | Confidence | Holding After Lift | Remaining At Place | Goal Once? | Likely Cause                      |
| ------- | ------------- | ----- | ---------- | ----------- | --------- | ---------- | ------------------ | ------------------ | ---------- | --------------------------------- |
| 23      | horizon       | 701   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
| 29      | preplace      | 1000  | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Place SM cannot converge          |
| 32      | verify_failed | 1000  | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Place SM dropped block off-target |
| 33      | grasp_failed  | 718   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Grasp contact/slip failure        |
| 34      | horizon       | 680   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
| 39      | preplace      | 1000  | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Place SM cannot converge          |
| 40      | horizon       | 694   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
| 45      | horizon       | 691   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
| 47      | verify_failed | 1000  | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Place SM dropped block off-target |
| 51      | horizon       | 695   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
| 62      | horizon       | 686   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
| 84      | preplace      | 1000  | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Place SM cannot converge          |
| 88      | horizon       | 671   | none       | ?           | No        | N/A        | N/A                | N/A                | No         | Horizon exceeded in later phases  |
