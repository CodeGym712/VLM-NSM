# P0 Artifact Recovery Audit — 2026-07-02

## Decision

Historical embodied data is **not yet recoverable from this workspace**. `artifacts_manifest.yaml` contains a server path but no resolvable host, account, transport, checksum, schema, or license/consent record. The manifest is therefore an inventory claim, not evidence that the data is currently available for the thesis.

## Evidence checked

| Check | Result |
|---|---|
| Manifest server host | Literal placeholder `current server` |
| Manifest base path | `/rhome/zhaowm/my_project/VLM-NSM/generated` |
| Local `generated/datasets` | Missing |
| Local `generated/demos` | Missing |
| Local `generated/models` | Missing |
| Local `generated/videos` | Missing |
| Local `generated/eval_inputs` | Missing |
| Local `generated/results` | 2 files, 647 bytes total; only `.gitkeep` plus one mock result |
| Local simulator/runtime | Explicitly excluded by `docs/local-development.md` |
| Remote checksum/index | Missing |
| EmbodiedTransition v1 compatibility | Unknown; historical files predate the schema |
| Data-use/license record | Not present in the artifact manifest |

The advertised 208 GB `generated/datasets/vla_demos/` is marked as abandoned VLA data. Even if recovered, it cannot be counted toward G0 until sampled files prove they contain or can reconstruct complete pre/post predicates, option identity, observer/executor versions, diagnostics, and failure attribution.

## Recovery request required

The project owner/server operator must provide one of:

1. an SSH alias or hostname and read-only project path; or
2. a small export containing the manifest plus 20 representative episodes.

The minimum remote audit output is:

```text
host alias (not a password or key)
absolute dataset roots
file counts and byte totals
20 sampled relative paths
SHA-256 for the sample
per-format field inventory
collection/runtime version if known
data-use or consent status
```

Credentials must remain in SSH/secret-manager configuration and must not be written to this repository.

## Current P0 status

- Completed: EmbodiedTransition v1 schema and typed model.
- Completed: predicate source/confidence/time/freshness/evidence fields.
- Completed: validation, duplicate-ID detection, and run-group-safe split.
- Completed: frozen `predicate_vocab_v1` and `executor_v1` manifests.
- Blocked externally: remote artifact existence and sample audit.
- Not started: simulator collection adapter and 20-transition smoke collection.
- G0 status: **not passed**; zero locally auditable embodied transitions.

## Next engineering action

Add a recorder adapter at the AtomicPick/AtomicPlace boundary that captures a `PredicateSceneGraph` before and after each grounded option and writes EmbodiedTransition v1 JSONL. This can be implemented locally with mock execution tests; actual 20-transition collection must run on the configured simulator server.
