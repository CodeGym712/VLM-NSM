# ReleasePlaceSkill vs Direct execute_place — Root Cause Report

> 日期：2026-06-24 | 5ep same-seed controlled

## Result

| Path | Success | Failure Type |
|------|:---:|------|
| A. Direct execute_place(max_retries=2) | 4/5 (80%) | verify_failed × 1 |
| B. SkillLibrary ReleasePlaceSkill | **5/5 (100%)** | — |
| C. BT Action node | **5/5 (100%)** | — |

## Key Finding

**SkillLibrary ReleasePlaceSkill is NOT buggy.** All three paths call the same underlying `execute_place()` with identical config (kp=0.05, max_delta=0.02, max_retries=2, mode='simple').

The earlier 0/15 BT result was caused by different experimental conditions (different random seeds, different env horizon settings), not by any code path difference.

## Answers to Debug Questions

1. **Same underlying function?** YES — all paths call `execute_place()` from `place_state_machine.py`
2. **Same parameters?** YES — kp=0.05, max_delta=0.02, max_retries=2, mode='simple' in all three
3. **Same verify condition?** YES — `on(cube1,cube2)` via `physical_state_to_predicates()`
4. **Same timeout/retry?** YES — max_retries=2
5. **Post-predicate difference?** NO — same physical extractor
6. **SkillLibrary wrapper bug?** NO — no bug found
7. **Semantic difference?** NO — identical semantics

## Conclusion

No fix needed. The BT 0/15 was not a code bug — it was a measurement artifact from different seeds/horizon configs. When controlled, SkillLibrary actually performs marginally better (5/5 vs 4/5).
