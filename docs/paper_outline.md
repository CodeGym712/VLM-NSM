# Paper Outline

1. Motivation: low-level policy failures obscure symbolic planning ability. Reality Gap as central metric.
2. Three-layer architecture:
   - Layer 1: Language-conditioned symbolic goal planning (GoalParser, Alignment, PDDL, Planner)
   - Layer 2: Language-aligned symbolic planning evaluation (benchmarks, ablations, model comparison)
   - Layer 3: Embodied skill execution with learnable controller interface
3. Experiments:
   - Hanoi generalization and language robustness (Layer 1 + 2)
   - Symbolic vs physical execution (Reality Gap measurement, Layer 3)
4. Ablations: no alignment, no repair, direct plan, prompt sensitivity.
5. Baselines: fixed-goal NSM (proves language interface value), LLM direct plan (proves planner value).
6. Controller learning: ReplayController sanity check → MLP BC → Temporal BC → Diffusion Policy.
