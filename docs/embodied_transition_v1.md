# EmbodiedTransition v1

This is the P0 data boundary for failure-aware symbolic operator learning. The canonical machine contract is `schemas/embodied_transition_v1.schema.json`.

## Required evidence

Every row is one JSON object containing a complete pre-state, one grounded option, a complete post-state, expected and observed effects, observer/executor versions, controller diagnostics, raw failure, causal attribution, and evidence IDs. A boolean `success` without these fields is not a research transition.

`vlm_nsm.data.transition_recorder.EmbodiedTransitionRecorder` converts pre/post `PredicateSceneGraph` snapshots and a `SkillResult` into this representation. Successful options receive `SUCCESS`; failed options deliberately require an explicit causal attribution so a raw `nocontact`/`timeout` string cannot silently become a symbolic counterexample.

Each predicate observation records `source`, `confidence`, `observed_at`, `freshness_seconds`, and `evidence_id`. P0 uses `predicate_vocab_v1`; it does not infer new predicates.

## Attribution and operator-learning policy

| Attribution | Symbolic evidence policy |
|---|---|
| `SUCCESS` | positive nominal evidence |
| `SYMBOLIC_INVALID` | negative applicability/precondition evidence |
| `MODEL_MISMATCH` | counterexample to the current operator |
| `OBSERVATION_UNCERTAIN` | defer or down-weight |
| `LANGUAGE` | exclude from operator update |
| `CAPABILITY_OOD` | exclude from operator update |
| `CONTROLLER_FAILURE` | exclude from operator update |
| `EVIDENCE_INVALID` | exclude from operator update |

The raw controller failure string is preserved separately. Attribution is a causal research label, not a renamed failure code.

## Audit command

```powershell
.\.venv\Scripts\python.exe -m vlm_nsm.experiments.audit_embodied_transitions `
  --input generated/datasets/embodied_transitions_v1.jsonl `
  --report generated/datasets/embodied_transitions_v1.audit.json `
  --split-dir generated/datasets/splits_v1 `
  --seed 20260702
```

The command returns non-zero when a row violates the schema, transition IDs repeat, or fewer than three run groups make a leakage-safe three-way split impossible. Splits are deterministic and keep every `run_id` in exactly one partition.

## Version discipline

- Predicate semantics change → new predicate vocabulary version.
- PID gain/state-machine behavior change → new executor manifest version.
- Schema field semantics change → new transition schema version.
- Active domain learning must retain the transition IDs and all three versions.
