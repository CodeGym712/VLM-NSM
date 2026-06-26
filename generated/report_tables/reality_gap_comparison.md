# Reality Gap Comparison

Comparison between Track A (symbolic planning, perfect execution) and Track B configurations (embodied execution with increasing reality fidelity).

Stage 7A-7: Strict RG = episodes with clean execution + goal success. Semantic RG = episodes where goal was achieved (may include timeout_after_goal).

| Config                     | Episodes | Step Success | Skill Success | Episode Success | Goal Success | Predicate Match | Strict RG | Semantic RG |
| -------------------------- | -------- | ------------ | ------------- | --------------- | ------------ | --------------- | --------- | ----------- |
| Track A (Symbolic Planner) | N/A      | 100.0%       | 100.0%        | 100.0%          | 100.0%       | 100.0%          | 0%        | 0%          |
| Track B: Robosuite         | 50       | 0.7%         | 0.0%          | 92.0%           | 92.0%        | 92.0%           | 8.0       | 8.0         |