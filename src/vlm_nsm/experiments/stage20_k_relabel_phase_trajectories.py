"""Relabel composite place trajectories to expose release/retreat phases.

This is an offline dataset utility. It copies an existing trajectory dataset
to a new output directory and rewrites ``phase_id`` for composite phase-3
place-state-machine segments. It does not run robosuite or train models.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.data.stage20_phase_labeling import phase_names, relabel_composite_place_phases


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _resolve_path(root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _phase_counts(phase_ids: np.ndarray) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in zip(*np.unique(phase_ids.astype(np.int64), return_counts=True))}


def relabel_dataset(input_dataset: Path, output_dataset: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dataset} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dataset)
    output_dataset.mkdir(parents=True, exist_ok=True)
    (output_dataset / "trajectories").mkdir(parents=True, exist_ok=True)

    manifest_path = input_dataset / "manifest.jsonl"
    records = _load_jsonl(manifest_path)
    output_records: list[dict[str, Any]] = []
    before_counts: Counter[str] = Counter()
    after_counts: Counter[str] = Counter()
    changed_trajectories = 0
    total_changed_steps = 0
    total_release_steps = 0
    total_retreat_steps = 0
    missing: Counter[str] = Counter()

    for record in records:
        in_path = _resolve_path(input_dataset, record.get("trajectory_path"))
        out_record = dict(record)
        if in_path is None or not in_path.exists():
            missing["trajectory_path"] += 1
            output_records.append(out_record)
            continue
        rel_path = Path(record["trajectory_path"])
        out_path = output_dataset / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with np.load(in_path, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}

        if "phase_id" not in arrays or "action" not in arrays:
            missing["phase_id_or_action"] += 1
            np.savez_compressed(out_path, **arrays)
            output_records.append(out_record)
            continue

        phase_raw = np.asarray(arrays["phase_id"], dtype=np.int32)
        before_counts.update(_phase_counts(phase_raw))
        phase_new, stats = relabel_composite_place_phases(phase_raw, arrays["action"])
        phase_new = phase_new.astype(np.int32)
        after_counts.update(_phase_counts(phase_new))

        arrays.setdefault("phase_id_raw", phase_raw.copy())
        arrays.setdefault("phase_name_raw", phase_names(phase_raw))
        arrays["phase_id"] = phase_new
        arrays["phase_name"] = phase_names(phase_new)
        arrays["phase_label_patch_applied"] = np.full((len(phase_new),), stats.applied, dtype=bool)
        arrays["phase_label_changed"] = phase_new != phase_raw
        arrays["phase_label_patch_version"] = np.array(["stage20_k_v1"] * len(phase_new))
        np.savez_compressed(out_path, **arrays)

        if stats.applied:
            changed_trajectories += 1
        total_changed_steps += stats.changed_steps
        total_release_steps += stats.release_steps
        total_retreat_steps += stats.retreat_steps
        out_record["phase_label_patch"] = stats.as_dict()
        out_record["phase_label_patch_version"] = "stage20_k_v1"
        output_records.append(out_record)

    _write_jsonl(output_dataset / "manifest.jsonl", output_records)

    for name in ("README.md",):
        src = input_dataset / name
        if src.exists():
            shutil.copy2(src, output_dataset / name)

    input_summary = {}
    if (input_dataset / "summary.json").exists():
        input_summary = json.loads((input_dataset / "summary.json").read_text(encoding="utf-8"))

    summary = {
        "source_dataset": str(input_dataset),
        "output_dataset": str(output_dataset),
        "phase_label_patch_version": "stage20_k_v1",
        "num_records": len(records),
        "changed_trajectories": changed_trajectories,
        "changed_steps": total_changed_steps,
        "release_steps_added": total_release_steps,
        "retreat_steps_added": total_retreat_steps,
        "phase_counts_before": dict(sorted(before_counts.items())),
        "phase_counts_after": dict(sorted(after_counts.items())),
        "missing_counts": dict(missing),
        "source_summary": input_summary,
    }
    (output_dataset / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (output_dataset / "README.md").write_text(
        "# Stage 20-K Phase-Labeled Teacher Dataset\n\n"
        "This dataset is copied from the Stage 20-F teacher trajectories with composite "
        "`reach_place` segments relabeled into `release` and `retreat` using recorded actions. "
        "No robosuite rollout was run for this relabeling step.\n",
        encoding="utf-8",
    )
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    return f"""# Stage 20-K Phase Labeling Report

## Summary

```text
source_dataset = {summary.get('source_dataset')}
output_dataset = {summary.get('output_dataset')}
num_records = {summary.get('num_records')}
changed_trajectories = {summary.get('changed_trajectories')}
changed_steps = {summary.get('changed_steps')}
release_steps_added = {summary.get('release_steps_added')}
retreat_steps_added = {summary.get('retreat_steps_added')}
```

## Phase Counts

Before:

```text
{summary.get('phase_counts_before')}
```

After:

```text
{summary.get('phase_counts_after')}
```

## Interpretation

- Composite phase 3 place-state-machine timesteps are now split into phase 4 release and phase 5 retreat when action evidence is available.
- The relabeling is offline and does not change controller behavior.
- Existing Stage 20-F source trajectories are not overwritten.

## Recommendation

Rebuild a role-conditioned dataset from the phase-labeled copy and use it before training the next learned controller baseline.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel Stage 20 composite place phases")
    parser.add_argument("--input_dataset", required=True)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--markdown_output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = relabel_dataset(Path(args.input_dataset), Path(args.output_dataset), overwrite=args.overwrite)
    output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
