================================================================================
               Stage 7B-1: MLP Behavior Cloning — Summary Report                
================================================================================

--- Training Configuration ---

  Model:        MLP(11 -> 256 -> 256 -> 7)
  Parameters:   70,663
  Dataset:      generated/demos/stage_7B-1/robosuite_stage7b1_trajectories.npz
  Obs key:      filtered_obs
  Epochs:       150
  Batch size:   256
  Learning rate: 1e-3
  Train episodes: 90 (78095 steps)
  Val episodes:   10 (9136 steps)
  Train time:    473s

--- Training Loss ---

  Initial train loss: 0.3770
  Final train loss:   0.0410
  Initial val loss:   0.2260
  Best val loss:      0.0431

--- BC vs Scripted Policy Comparison (20 episodes) ---

Metric                         BC Policy       Scripted Policy   
---------------------------------------------------------------
Episode Success                0%              96%               
Mean Steps                     1000            856               
p95 Steps                      1000            892               

--- Analysis ---

The MLP BC policy (0% success) underperforms the scripted policy (96%) due to:
  1. Distribution shift: small action prediction errors compound over the
     long horizon (~860 steps), causing the policy to diverge.
  2. Constant action dimensions: dims 3-5 have zero variance in the training
     data (orientation held constant by proportional controller), creating
     numerical instability after normalization.
  3. Limited model capacity: 70k params may be insufficient to capture the
     full state-to-action mapping for this task.

Next steps for Stage 7C:
  - Use observation filtering (symbol-conditioned) to reduce task complexity
  - Add action dimension masking for constant dimensions
  - Explore larger architectures (e.g., Transformer-based policy)
  - Implement DAgger or other interactive imitation learning methods