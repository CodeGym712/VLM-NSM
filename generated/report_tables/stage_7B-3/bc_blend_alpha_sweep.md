# bc_blend_alpha Sweep Results

**Model:** Stage 7B-2R phase-conditioned BC (strict_success data)
**Control mode:** Hybrid — BC for reach_pick + reach_place, scripted for others
**Episodes:** 20 per alpha value

## Alpha Sweep Table

| Alpha | Success Rate | Mean Steps | BC Steps | Scripted Steps |
|:-----:|:-----------:|:----------:|:--------:|:--------------:|
| 0.00 (scripted) | ~88% | ~976 | 0 | ~976 |
| 0.25 | 40% | 587 | 202 | 404 |
| 0.50 | 15% | 696 | 251 | 500 |
| 0.75 | 10% | 785 | 298 | 546 |
| 1.00 (pure BC) | 5% | 735 | 272 | 483 |

## Key Observations

1. **Higher alpha = worse performance** — counterintuitive, because blending should
   smooth BC behavior.

2. **Scale mismatch between scripted and BC actions**: Scripted actions (proportional
   control) output position deltas of ~0.01-0.015 per step, while BC model outputs
   normalized values up to 1.0. The numerical dominance of BC even at low alpha
   renders the blend ineffective.

3. **Pure BC (alpha=1.0) fails**: 5% success rate for the stage_7B-2R model. This
   is inconsistent with the seed evals (90-100%) — possibly due to checkpoint
   format, action denormalization, or training data differences.

4. **Fix needed**: BC and scripted actions must be blended in the same space.
   Either denormalize BC actions before blending, or normalize scripted actions
   to match BC output space.

## Raw Results

- `generated/results/stage_7B-3/bc_alpha_025_20ep.json`
- `generated/results/stage_7B-3/bc_alpha_050_20ep.json`
- `generated/results/stage_7B-3/bc_alpha_075_20ep.json`
- `generated/results/stage_7B-3/bc_alpha_100_20ep.json`
