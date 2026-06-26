# Eval-Collect Config Parity Audit

- **Eval config**: `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`
- **Collect config**: `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json`
- **Total fields**: 29
- **Mismatches**: 0
- **All match**: ✓ Yes

| Field                                | Eval Value    | Collect Value | Match? |
| ------------------------------------ | ------------- | ------------- | ------ |
| kp                                   | 0.08          | 0.08          | ✓      |
| max_delta                            | 0.015         | 0.015         | ✓      |
| reach_threshold                      | 0.015         | 0.015         | ✓      |
| approach_height                      | 0.08          | 0.08          | ✓      |
| descend_offset                       | 0.005         | 0.005         | ✓      |
| lift_height                          | 0.05          | 0.05          | ✓      |
| grasp_close_steps                    | 15            | 15            | ✓      |
| close_mode                           | adaptive      | adaptive      | ✓      |
| close_steps_min                      | 10            | 10            | ✓      |
| close_steps_fast                     | 18            | 18            | ✓      |
| close_steps_max                      | 28            | 28            | ✓      |
| close_stabilize_steps                | 5             | 5             | ✓      |
| fallback_extend_close_if_not_holding | True          | True          | ✓      |
| holding_check_enabled                | True          | True          | ✓      |
| lift_early_exit_enabled              | True          | True          | ✓      |
| grasp_max_retries                    | 3             | 3             | ✓      |
| place_max_retries                    | 2             | 2             | ✓      |
| place_state_machine.mode             | simple        | simple        | ✓      |
| place_state_machine.enabled          | True          | True          | ✓      |
| terminate_on_goal                    | True          | True          | ✓      |
| terminate_after_safe_release         | False         | False         | ✓      |
| skip_retreat_if_goal_satisfied       | True          | True          | ✓      |
| horizon_aware.enabled                | True          | True          | ✓      |
| release_steps_min                    | 10            | 10            | ✓      |
| release_steps_max                    | 40            | 40            | ✓      |
| post_release_stable_steps            | 5             | 5             | ✓      |
| release_open_steps                   | N/A (default) | N/A (default) | ✓      |
| log_grasp_confidence                 | True          | True          | ✓      |
| mark_grasp_uncertain                 | True          | True          | ✓      |