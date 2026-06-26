====================================================================================================
                        Stage 7A-12 Candidate Comparison (20 episodes each)                         
====================================================================================================
Metric                         A (fixed15)        B (adaptive 10/15/25)    C (adaptive 10/18/28)   
----------------------------------------------------------------------------------------------------
Strict Success                                  85.0%                  85.0%                 100.0%
Semantic Success                                85.0%                  85.0%                 100.0%
Mean Steps                                      860.7                  854.3                  859.8
p95 Steps                                      1000.0                 1000.0                  876.9
Reality Gap                                   1500.0%                1500.0%                   0.0%
Strict RG                                     1500.0%                1500.0%                   0.0%
----------------------------------------------------------------------------------------------------

A (fixed15):
  Failures: {'preplace': 1, 'horizon': 1, 'horizon_preplace': 1}
  Termination: {'terminate_on_goal_count': 17, 'terminate_after_safe_release_count': 0, 'skip_retreat_if_goal_satisfied_count': 0, 'normal_completion_count': 1, 'horizon_termination_count': 3}
  Timeout before goal: 2

B (adaptive_10_15_25):
  Failures: {'horizon': 1, 'verify_failed': 2}
  Termination: {'terminate_on_goal_count': 17, 'terminate_after_safe_release_count': 0, 'skip_retreat_if_goal_satisfied_count': 0, 'normal_completion_count': 1, 'horizon_termination_count': 2}
  Timeout before goal: 2

C (adaptive_10_18_28):
  Failures: {}
  Termination: {'terminate_on_goal_count': 20, 'terminate_after_safe_release_count': 0, 'skip_retreat_if_goal_satisfied_count': 0, 'normal_completion_count': 0, 'horizon_termination_count': 0}
  Timeout before goal: 0

Best config: C (adaptive_10_18_28)
Decision: freeze as robosuite_best_execution_config_stage7a12_final.json