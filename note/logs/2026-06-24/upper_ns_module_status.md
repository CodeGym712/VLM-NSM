# Upper Neuro-Symbolic Module Status

> 日期：2026-06-24 | 1651 tests passed

| Module | Path | Unit Tests | Default Path | Experimental Path | MuJoCo Tested | Metric Improved | Current Status | Risk |
|---|---|---:|---:|---:|---:|---:|---|---|
| PredicateSceneGraph | core/predicate_scene_graph.py | 13 | ✅ composer path | — | ❌ | — | **validated capability** | Low |
| OptionContract | core/option_contract.py | — | ✅ task_runner pre/post fill | — | ✅ (per-skill trace) | — | **validated capability** | Low |
| SkillResult pre/post | embodied/task_runner.py | — | ✅ auto-filled per skill | — | ✅ (1 episode trace) | — | **validated capability** | Low |
| FailureType | core/failure_types.py | — | ✅ shared enum | — | ❌ | — | **library only** | Low |
| VerifierBank | neuro_symbolic/verifier_bank.py | 22 | ✅ composer path | — | ❌ | — | **validated capability** | Low |
| TaskComposer | language/task_composer.py | — | — | ✅ --composer flag | ❌ (mock only) | — | **experimental path** | Medium |
| OptionDSL | language/option_dsl.py | — | — | — | ❌ | — | **library only** | Medium |
| BTExecutor | embodied/behavior_tree.py | 12 | — | ✅ direct tick() | ✅ (BT trace) | ❌ (0/15 vs 93%) | **experimental path** | High |
| FailureSummarizer | neuro_symbolic/failure_summarizer.py | 9 | — | ✅ can consume trace | ❌ | — | **experimental path** | Low |
| RepairProposer | neuro_symbolic/repair_proposer.py | 6 | — | ✅ hand-coded rules | ❌ | — | **experimental path** | Low |

**Key**:
- `validated capability`: Has unit tests + integration evidence + metric measured
- `experimental path`: Functional but needs flag or separate runner
- `library only`: Classes/tests exist but not consumed by any runner
- `default path`: Called automatically in main execution pipeline
