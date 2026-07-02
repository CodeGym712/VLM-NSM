# P0 Transition Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first P0 slice: a versioned embodied-transition contract, predicate observation provenance, and deterministic dataset audit/deduplication/group-split tooling.

**Architecture:** Keep the existing dataclass style and add no runtime framework. `Predicate` carries observation provenance; `EmbodiedTransition` is the typed in-memory and JSONL boundary; JSON Schema is the language-independent contract; dataset helpers validate records, reject duplicate IDs, and split whole runs rather than adjacent transitions.

**Tech Stack:** Python 3.10+, dataclasses, jsonschema, pytest, JSON/JSONL.

---

### Task 1: Specify the transition and predicate evidence contract

**Files:**
- Create: `schemas/embodied_transition_v1.schema.json`
- Modify: `src/vlm_nsm/symbolic/predicates.py`
- Test: `tests/test_embodied_transition_schema.py`
- Test: `tests/test_predicate_scene_graph.py`

- [x] Write tests that require predicate `observed_at`, `freshness_seconds`, and `evidence_id`, plus a valid transition fixture accepted by JSON Schema.
- [x] Run `pytest tests/test_embodied_transition_schema.py tests/test_predicate_scene_graph.py -q` and confirm failure because the schema and fields are missing.
- [x] Add the three backward-compatible predicate fields after `source`, serialize them, and add the draft-2020-12 transition schema.
- [x] Re-run the focused tests and confirm they pass.

### Task 2: Add the typed transition model

**Files:**
- Create: `src/vlm_nsm/data/embodied_transition.py`
- Test: `tests/test_embodied_transition.py`

- [x] Write tests for enum-backed attribution, round-trip serialization, stable content hash, and evidence eligibility.
- [x] Run `pytest tests/test_embodied_transition.py -q` and confirm import failure.
- [x] Implement `FailureAttribution`, `PredicateObservation`, and `EmbodiedTransition` dataclasses with `to_dict`, `from_dict`, `content_hash`, and `symbolic_evidence_policy`.
- [x] Re-run the tests and confirm they pass.

### Task 3: Add dataset validation, deduplication, and grouped splitting

**Files:**
- Create: `src/vlm_nsm/data/transition_dataset.py`
- Test: `tests/test_transition_dataset.py`

- [x] Write tests that require line-numbered schema errors, duplicate transition detection, and run-exclusive deterministic splits.
- [x] Run `pytest tests/test_transition_dataset.py -q` and confirm import failure.
- [x] Implement `load_jsonl`, `audit_records`, `deduplicate_records`, and `group_split` using `jsonschema.Draft202012Validator` and SHA-256 ordering of group IDs.
- [x] Re-run the tests and confirm they pass.

### Task 4: Expose an audit CLI and freeze P0 identifiers

**Files:**
- Create: `src/vlm_nsm/experiments/audit_embodied_transitions.py`
- Create: `configs/symbolic_learning/predicate_vocab_v1.yaml`
- Create: `configs/symbolic_learning/executor_v1.yaml`
- Create: `docs/embodied_transition_v1.md`
- Test: `tests/test_audit_embodied_transitions.py`

- [x] Write a CLI test that audits a temporary JSONL file and writes a machine-readable report.
- [x] Run the focused test and confirm failure because the CLI is absent.
- [x] Implement CLI arguments `--input`, `--schema`, `--report`, and optional `--split-dir`; return non-zero on invalid or duplicate records.
- [x] Add frozen vocabulary/executor manifests and document field semantics and attribution policy.
- [x] Re-run the focused test and confirm pass.

### Task 5: Regression verification

**Files:**
- Modify only if a regression is exposed by the tests above.

- [x] Run the four new test files plus `tests/test_predicate_scene_graph.py`.
- [ ] Run the complete non-deprecated test suite with `pytest tests --ignore=tests/deprecated -q`. Blocked at collection by 13 Torch tests and one HDF5 test; these optional dependencies are not declared in `pyproject.toml`.
- [x] Run the audit CLI through temporary-file integration tests and inspect its JSON output.
- [x] Review `git diff --check`, `git diff`, and `git status --short`; preserve unrelated `outputs/` and `tmp/` files.

Verified fallback: after adding the recorder, the repository's dependency-light local profile completed with `642 passed, 1 skipped`.

## P0 environment audit

- [x] Inspect `artifacts_manifest.yaml` and local generated-artifact inventory.
- [x] Record the recovery decision in `docs/p0_artifact_recovery_audit_2026-07-02.md`.
- [ ] Verify remote paths and sample 20 episodes. Blocked because the manifest host is the literal placeholder `current server`; no SSH alias/hostname is recorded.
- [ ] Collect 20 simulator smoke transitions. Server-only runtime is not available in the Windows local profile.

### Task 6: Add an option-boundary transition recorder

**Files:**
- Create: `src/vlm_nsm/data/transition_recorder.py`
- Test: `tests/test_transition_recorder.py`

- [x] Write failing tests for a schema-valid successful record, explicit failed-option attribution, and raw failure preservation.
- [x] Confirm test collection fails because the recorder module is absent.
- [x] Implement graph-to-predicate evidence conversion, observed-effect differencing, explicit failure attribution, and append-only JSONL output.
- [x] Run the focused recorder tests and confirm all pass.
- [ ] Wire the recorder into a server-side AtomicPick/AtomicPlace experiment runner and collect 20 transitions. This requires the simulator runtime and a selected failure-attribution callback.

## Scope intentionally deferred

- Remote artifact recovery and MuJoCo collection require external storage/simulator availability and are not fabricated by this slice.
- LOFT-style operator learning starts only after G0 data fields are stable.
- Predicate invention, ASP, RL, RGB perception, and PID changes remain out of scope.
