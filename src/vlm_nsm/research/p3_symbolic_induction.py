"""P3 offline symbolic induction from LIBERO PredicateTrace.

This module intentionally does not depend on MuJoCo / robosuite.  It consumes
the P2v4.2 PredicateTrace package and produces the first P3 artifacts:

- input contract and trusted predicate mask
- normalized per-frame traces
- stable effect events
- operator candidates
- plan-skeleton replay audit
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

Atom = tuple[str, ...]
DemoKey = tuple[str, str]


DEFAULT_MASK: dict[str, Any] = {
    "trusted": ["in", "open", "activated", "handempty", "closed", "deactivated"],
    "trusted_with_temporal_filter": {
        "holding": {"min_consecutive_frames": 3},
    },
    "conditional": {
        "on": {
            "exclude_table_when_specific_support_exists": True,
            "note": "on(obj, table) is masked when obj has a specific non-table support in the same frame.",
        },
    },
    "excluded": ["at_region"],
}


def canonical_entity(name: str) -> str:
    """Normalize simulator entity names without deleting object instance ids."""
    if name.endswith("_joint0"):
        return name[: -len("_joint0")]
    return name


def atom_to_list(atom: Atom) -> list[str]:
    return list(atom)


def atom_to_str(atom: Atom) -> str:
    if len(atom) == 1:
        return f"{atom[0]}()"
    return f"{atom[0]}({', '.join(atom[1:])})"


def sort_atoms(atoms: Iterable[Atom]) -> list[Atom]:
    return sorted(set(atoms), key=lambda item: (item[0], item[1:]))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def raw_atoms_from_predicates(predicates: Mapping[str, Any]) -> dict[str, set[Atom]]:
    """Convert heterogeneous P2 predicate payloads into canonical atom sets."""
    atoms: dict[str, set[Atom]] = defaultdict(set)

    holding = predicates.get("holding")
    if holding:
        atoms["holding"].add(("holding", canonical_entity(str(holding))))

    if predicates.get("handempty") is True:
        atoms["handempty"].add(("handempty",))

    for item in predicates.get("on") or []:
        top = canonical_entity(str(item["top"]))
        bot = canonical_entity(str(item["bot"]))
        atoms["on"].add(("on", top, bot))

    for item in predicates.get("in") or []:
        obj = canonical_entity(str(item["obj"]))
        container = canonical_entity(str(item["container"]))
        atoms["in"].add(("in", obj, container))

    for pred_name in ("open", "closed", "activated", "deactivated"):
        for entity in predicates.get(pred_name) or []:
            atoms[pred_name].add((pred_name, canonical_entity(str(entity))))

    for item in predicates.get("at_region") or []:
        obj = canonical_entity(str(item["obj"]))
        region = canonical_entity(str(item["region"]))
        atoms["at_region"].add(("at_region", obj, region))

    return atoms


@dataclass(frozen=True)
class NormalizedFrame:
    file: str
    ds: str
    demo: str
    task: str
    t: int
    T: int
    trusted: tuple[Atom, ...]
    conditional: tuple[Atom, ...]
    excluded: tuple[Atom, ...]
    risk_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "ds": self.ds,
            "demo": self.demo,
            "task": self.task,
            "t": self.t,
            "T": self.T,
            "trusted_predicates": [atom_to_list(atom) for atom in self.trusted],
            "conditional_predicates": [atom_to_list(atom) for atom in self.conditional],
            "excluded_predicates": [atom_to_list(atom) for atom in self.excluded],
            "risk_notes": list(self.risk_notes),
        }


def task_lookup(package_dir: Path) -> dict[str, str]:
    tasks = read_json(package_dir / "metadata" / "task_list.json")
    return {row["file"]: row["task"] for row in tasks}


def group_trace_rows(rows: Iterable[dict[str, Any]]) -> dict[DemoKey, list[dict[str, Any]]]:
    grouped: dict[DemoKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["file"], row["demo"])].append(row)
    for demo_rows in grouped.values():
        demo_rows.sort(key=lambda row: int(row["t"]))
    return dict(grouped)


def stable_holding_atoms(raw_by_frame: list[dict[str, set[Atom]]], min_frames: int = 3) -> list[set[Atom]]:
    """Return holding atoms that belong to a run of at least ``min_frames``."""
    stable = [set() for _ in raw_by_frame]
    active_atom: Atom | None = None
    run_start = 0

    def flush(end_exclusive: int) -> None:
        if active_atom is None:
            return
        if end_exclusive - run_start >= min_frames:
            for idx in range(run_start, end_exclusive):
                stable[idx].add(active_atom)

    for idx, atoms in enumerate(raw_by_frame):
        current = next(iter(atoms.get("holding", set())), None)
        if current != active_atom:
            flush(idx)
            active_atom = current
            run_start = idx
    flush(len(raw_by_frame))
    return stable


def normalize_demo_rows(
    demo_rows: Sequence[dict[str, Any]],
    task_by_file: Mapping[str, str],
    holding_min_frames: int = 3,
) -> list[NormalizedFrame]:
    raw_by_frame = [raw_atoms_from_predicates(row["predicates"]) for row in demo_rows]
    stable_holding = stable_holding_atoms(raw_by_frame, min_frames=holding_min_frames)
    normalized: list[NormalizedFrame] = []

    for idx, (row, raw_atoms) in enumerate(zip(demo_rows, raw_by_frame)):
        trusted: set[Atom] = set()
        conditional: set[Atom] = set()
        excluded: set[Atom] = set(raw_atoms.get("at_region", set()))
        risk_notes: list[str] = []

        for pred_name in ("in", "open", "closed", "activated", "deactivated", "handempty"):
            trusted.update(raw_atoms.get(pred_name, set()))

        raw_holding = raw_atoms.get("holding", set())
        trusted.update(stable_holding[idx])
        for atom in raw_holding - stable_holding[idx]:
            conditional.add(atom)
            risk_notes.append("holding_masked_by_min_3_consecutive_frames")

        on_atoms = raw_atoms.get("on", set())
        specific_support_by_obj: dict[str, set[Atom]] = defaultdict(set)
        for atom in on_atoms:
            _, obj, support = atom
            if support != "table":
                specific_support_by_obj[obj].add(atom)
        for atom in on_atoms:
            _, obj, support = atom
            if support == "table" and specific_support_by_obj.get(obj):
                conditional.add(atom)
                risk_notes.append(f"on_table_masked_due_to_specific_support:{obj}")
            else:
                trusted.add(atom)

        file_name = str(row["file"])
        normalized.append(
            NormalizedFrame(
                file=file_name,
                ds=str(row.get("ds", "")),
                demo=str(row["demo"]),
                task=task_by_file.get(file_name, file_name.removesuffix("_demo.hdf5")),
                t=int(row["t"]),
                T=int(row["T"]),
                trusted=tuple(sort_atoms(trusted)),
                conditional=tuple(sort_atoms(conditional)),
                excluded=tuple(sort_atoms(excluded)),
                risk_notes=tuple(sorted(set(risk_notes))),
            )
        )
    return normalized


def normalize_traces(package_dir: Path, holding_min_frames: int = 3) -> tuple[list[NormalizedFrame], dict[str, Any]]:
    rows = read_jsonl(package_dir / "p2_predicate_frames_v4.jsonl")
    grouped = group_trace_rows(rows)
    by_file = task_lookup(package_dir)
    normalized: list[NormalizedFrame] = []
    report: dict[str, Any] = {
        "input_frames": len(rows),
        "demos": len(grouped),
        "holding_min_consecutive_frames": holding_min_frames,
        "predicate_counts": Counter(),
        "conditional_counts": Counter(),
        "excluded_counts": Counter(),
        "demo_lengths": {},
        "issues": [],
    }

    for key, demo_rows in grouped.items():
        frames = normalize_demo_rows(demo_rows, by_file, holding_min_frames=holding_min_frames)
        normalized.extend(frames)
        report["demo_lengths"][f"{key[0]}::{key[1]}"] = len(frames)
        expected_ts = list(range(len(frames)))
        actual_ts = [frame.t for frame in frames]
        if actual_ts != expected_ts:
            report["issues"].append({"demo": f"{key[0]}::{key[1]}", "issue": "non_contiguous_t"})
        for frame in frames:
            for atom in frame.trusted:
                report["predicate_counts"][atom[0]] += 1
            for atom in frame.conditional:
                report["conditional_counts"][atom[0]] += 1
            for atom in frame.excluded:
                report["excluded_counts"][atom[0]] += 1

    report["predicate_counts"] = dict(report["predicate_counts"])
    report["conditional_counts"] = dict(report["conditional_counts"])
    report["excluded_counts"] = dict(report["excluded_counts"])
    return normalized, report


def merge_change_points(
    frames: Sequence[NormalizedFrame],
    merge_gap: int = 3,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if len(frames) < 2:
        return events

    open_event: dict[str, Any] | None = None
    before_set: set[Atom] | None = None
    after_set: set[Atom] | None = None

    def close_event() -> None:
        nonlocal open_event, before_set, after_set
        if open_event is None or before_set is None or after_set is None:
            return
        added = sort_atoms(after_set - before_set)
        deleted = sort_atoms(before_set - after_set)
        if not added and not deleted:
            open_event = None
            before_set = None
            after_set = None
            return
        risk_notes = set(open_event["risk_notes"])
        if any(atom[0] == "on" and len(atom) > 2 and atom[2] == "table" for atom in added + deleted):
            risk_notes.add("event_depends_on_on_table")
        events.append(
            {
                **{k: open_event[k] for k in ("file", "ds", "demo", "task", "t_start", "t_end")},
                "before": [atom_to_list(atom) for atom in sort_atoms(before_set)],
                "after": [atom_to_list(atom) for atom in sort_atoms(after_set)],
                "added": [atom_to_list(atom) for atom in added],
                "deleted": [atom_to_list(atom) for atom in deleted],
                "confidence": "risky" if risk_notes else "trusted",
                "risk_notes": sorted(risk_notes),
            }
        )
        open_event = None
        before_set = None
        after_set = None

    previous = set(frames[0].trusted)
    for idx in range(1, len(frames)):
        current = set(frames[idx].trusted)
        if current == previous:
            previous = current
            continue
        if open_event is None:
            open_event = {
                "file": frames[idx].file,
                "ds": frames[idx].ds,
                "demo": frames[idx].demo,
                "task": frames[idx].task,
                "t_start": frames[idx].t,
                "t_end": frames[idx].t,
                "risk_notes": list(frames[idx].risk_notes),
            }
            before_set = set(previous)
            after_set = set(current)
        elif frames[idx].t <= int(open_event["t_end"]) + merge_gap:
            open_event["t_end"] = frames[idx].t
            open_event["risk_notes"].extend(frames[idx].risk_notes)
            after_set = set(current)
        else:
            close_event()
            open_event = {
                "file": frames[idx].file,
                "ds": frames[idx].ds,
                "demo": frames[idx].demo,
                "task": frames[idx].task,
                "t_start": frames[idx].t,
                "t_end": frames[idx].t,
                "risk_notes": list(frames[idx].risk_notes),
            }
            before_set = set(previous)
            after_set = set(current)
        previous = current
    close_event()
    return events


def extract_effect_events(frames: Sequence[NormalizedFrame], merge_gap: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[DemoKey, list[NormalizedFrame]] = defaultdict(list)
    for frame in frames:
        grouped[(frame.file, frame.demo)].append(frame)
    events: list[dict[str, Any]] = []
    for demo_frames in grouped.values():
        demo_frames.sort(key=lambda frame: frame.t)
        events.extend(merge_change_points(demo_frames, merge_gap=merge_gap))

    summary = {
        "event_count": len(events),
        "demo_count": len(grouped),
        "events_per_demo": Counter(f"{event['file']}::{event['demo']}" for event in events),
        "event_confidence_counts": Counter(event["confidence"] for event in events),
        "changed_predicate_counts": Counter(
            atom[0] for event in events for atom in event["added"] + event["deleted"]
        ),
    }
    summary["events_per_demo"] = dict(summary["events_per_demo"])
    summary["event_confidence_counts"] = dict(summary["event_confidence_counts"])
    summary["changed_predicate_counts"] = dict(summary["changed_predicate_counts"])
    return events, summary


def _atoms_from_event(event: Mapping[str, Any], field: str) -> set[Atom]:
    return {tuple(atom) for atom in event[field]}


def classify_event(event: Mapping[str, Any]) -> tuple[str, dict[str, str], list[str]]:
    added = _atoms_from_event(event, "added")
    deleted = _atoms_from_event(event, "deleted")
    risk_notes = list(event.get("risk_notes") or [])

    added_holding = [atom for atom in added if atom[0] == "holding"]
    deleted_holding = [atom for atom in deleted if atom[0] == "holding"]
    added_on = [atom for atom in added if atom[0] == "on"]
    added_in = [atom for atom in added if atom[0] == "in"]
    added_open = [atom for atom in added if atom[0] == "open"]
    added_activated = [atom for atom in added if atom[0] == "activated"]

    if added_holding and ("handempty",) in deleted:
        return "pick", {"obj": added_holding[0][1]}, risk_notes
    if deleted_holding and ("handempty",) in added:
        obj = deleted_holding[0][1]
        target = ""
        relation = ""
        if added_on:
            target = added_on[0][2]
            relation = "on"
        elif added_in:
            target = added_in[0][2]
            relation = "in"
        if target:
            return "place", {"obj": obj, "target": target, "relation": relation}, risk_notes
        return "release", {"obj": obj}, risk_notes + ["release_without_target_effect"]
    if added_open:
        return "open", {"drawer": added_open[0][1]}, risk_notes
    if added_activated:
        return "turn_on", {"device": added_activated[0][1]}, risk_notes
    if added_on or added_in:
        atom = (added_on or added_in)[0]
        return "relocate_effect", {"obj": atom[1], "target": atom[2], "relation": atom[0]}, risk_notes
    return "unclassified", {}, risk_notes


def induce_operator_candidates(events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    classified_events: list[dict[str, Any]] = []
    for event in events:
        operator, bindings, risk_notes = classify_event(event)
        enriched = dict(event)
        enriched["operator_type"] = operator
        enriched["bindings"] = bindings
        enriched["operator_risk_notes"] = risk_notes
        classified_events.append(enriched)
        grouped[operator].append(enriched)

    candidates: list[dict[str, Any]] = []
    for operator, items in sorted(grouped.items()):
        if operator == "unclassified":
            parameters: list[str] = []
            effects_add: list[list[str]] = []
            effects_del: list[list[str]] = []
        elif operator == "pick":
            parameters = ["?obj"]
            effects_add = [["holding", "?obj"]]
            effects_del = [["handempty"]]
        elif operator in {"place", "release"}:
            parameters = ["?obj", "?target"] if operator == "place" else ["?obj"]
            effects_add = [["handempty"], ["on-or-in", "?obj", "?target"]] if operator == "place" else [["handempty"]]
            effects_del = [["holding", "?obj"]]
        elif operator == "open":
            parameters = ["?drawer"]
            effects_add = [["open", "?drawer"]]
            effects_del = [["closed", "?drawer"]]
        elif operator == "turn_on":
            parameters = ["?device"]
            effects_add = [["activated", "?device"]]
            effects_del = [["deactivated", "?device"]]
        else:
            parameters = ["?obj", "?target"]
            effects_add = [["on-or-in", "?obj", "?target"]]
            effects_del = []
        risk_counter = Counter(note for item in items for note in item.get("operator_risk_notes", []))
        candidates.append(
            {
                "name": operator,
                "parameters": parameters,
                "effects_add_template": effects_add,
                "effects_del_template": effects_del,
                "support_count": len(items),
                "confidence": "risky" if risk_counter else "trusted",
                "risk_notes": dict(risk_counter),
                "example_events": [
                    {
                        "file": item["file"],
                        "demo": item["demo"],
                        "t_start": item["t_start"],
                        "t_end": item["t_end"],
                        "bindings": item.get("bindings", {}),
                    }
                    for item in items[:5]
                ],
            }
        )

    report = {
        "classified_event_count": len(classified_events),
        "operator_counts": dict(Counter(item["operator_type"] for item in classified_events)),
        "risky_operator_events": sum(1 for item in classified_events if item.get("operator_risk_notes")),
    }
    return candidates, {"classified_events": classified_events, "summary": report}


def collect_demo_index(frames: Sequence[NormalizedFrame]) -> dict[str, dict[str, str]]:
    demos: dict[str, dict[str, str]] = {}
    for frame in frames:
        demo_id = f"{frame.file}::{frame.demo}"
        demos.setdefault(
            demo_id,
            {
                "demo_id": demo_id,
                "file": frame.file,
                "demo": frame.demo,
                "task": frame.task,
                "ds": frame.ds,
            },
        )
    return demos


def build_plan_replay_eval(
    classified_events: Sequence[Mapping[str, Any]],
    demo_index: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    by_demo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in classified_events:
        by_demo[f"{event['file']}::{event['demo']}"].append(event)

    demos: list[dict[str, Any]] = []
    task_counter: dict[str, Counter] = defaultdict(Counter)
    all_demo_ids = sorted(set(demo_index or {}) | set(by_demo))
    for demo_id in all_demo_ids:
        items = by_demo.get(demo_id, [])
        items_sorted = sorted(items, key=lambda item: (int(item["t_start"]), int(item["t_end"])))
        sequence = [str(item["operator_type"]) for item in items_sorted if item["operator_type"] != "unclassified"]
        risky = [item for item in items_sorted if item.get("operator_risk_notes")]
        task = str(items_sorted[0]["task"]) if items_sorted else str((demo_index or {}).get(demo_id, {}).get("task", ""))
        task_counter[task]["demos"] += 1
        task_counter[task]["events"] += len(items_sorted)
        task_counter[task]["classified_events"] += len(sequence)
        demos.append(
            {
                "demo_id": demo_id,
                "task": task,
                "operator_sequence": sequence,
                "event_count": len(items_sorted),
                "classified_event_count": len(sequence),
                "risky_event_count": len(risky),
                "status": "covered" if sequence else "no_classified_events",
            }
        )

    return {
        "demo_count": len(demos),
        "covered_demos": sum(1 for row in demos if row["status"] == "covered"),
        "operator_counts": dict(Counter(op for row in demos for op in row["operator_sequence"])),
        "per_task": {
            task: {
                "demos": counts["demos"],
                "events": counts["events"],
                "classified_events": counts["classified_events"],
            }
            for task, counts in sorted(task_counter.items())
        },
        "demos": demos,
    }


def infer_plan_skeletons(
    classified_events: Sequence[Mapping[str, Any]],
    demo_index: Mapping[str, Mapping[str, str]],
    relation_lookback: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse low-level events into a demo-level operator skeleton.

    LIBERO predicate traces often show a target relation (e.g. on(bowl, plate))
    before the gripper release predicate change.  Treating each change point as
    a separate operator undercounts place events.  This pass keeps the raw event
    log intact and produces a separate, auditable skeleton view.
    """
    by_demo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in classified_events:
        by_demo[f"{event['file']}::{event['demo']}"].append(event)

    skeletons: list[dict[str, Any]] = []
    for demo_id in sorted(set(demo_index) | set(by_demo)):
        events = sorted(by_demo.get(demo_id, []), key=lambda item: (int(item["t_start"]), int(item["t_end"])))
        recent_relation: dict[str, tuple[str, str, int, str]] = {}
        steps: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []

        for event in events:
            op = str(event.get("operator_type"))
            bindings = dict(event.get("bindings") or {})
            t_start = int(event["t_start"])
            risk_notes = list(event.get("operator_risk_notes") or [])

            if op == "relocate_effect":
                obj = bindings.get("obj")
                target = bindings.get("target")
                relation = bindings.get("relation", "on-or-in")
                if obj and target:
                    recent_relation[obj] = (target, relation, t_start, event.get("confidence", "trusted"))
                continue

            if op == "release":
                obj = bindings.get("obj")
                relation = recent_relation.get(obj or "")
                if obj and relation and 0 <= t_start - relation[2] <= relation_lookback:
                    # Backward compatibility guard for old tuple shape is not
                    # needed for freshly generated artifacts, but keeping this
                    # unpack explicit makes the semantics visible.
                    target, relation_pred, relation_t, confidence = relation
                    steps.append(
                        {
                            "operator": "place",
                            "bindings": {"obj": obj, "target": target, "relation": relation_pred},
                            "t_start": t_start,
                            "t_end": int(event["t_end"]),
                            "source": "release_fused_with_recent_relation",
                            "relation_t": relation_t,
                            "confidence": "risky" if risk_notes or confidence == "risky" else "trusted",
                            "risk_notes": risk_notes + ["target_from_recent_relocate_effect"],
                        }
                    )
                else:
                    unresolved.append(
                        {
                            "operator": "release",
                            "bindings": bindings,
                            "t_start": t_start,
                            "reason": "no_recent_relation_target",
                        }
                    )
                continue

            if op in {"pick", "place", "open", "turn_on"}:
                steps.append(
                    {
                        "operator": op,
                        "bindings": bindings,
                        "t_start": t_start,
                        "t_end": int(event["t_end"]),
                        "source": "direct_event",
                        "confidence": "risky" if risk_notes else event.get("confidence", "trusted"),
                        "risk_notes": risk_notes,
                    }
                )

        meta = demo_index.get(demo_id, {})
        skeletons.append(
            {
                "demo_id": demo_id,
                "file": meta.get("file", demo_id.split("::")[0]),
                "demo": meta.get("demo", demo_id.split("::")[-1]),
                "task": meta.get("task", ""),
                "operator_sequence": [step["operator"] for step in steps],
                "steps": steps,
                "unresolved": unresolved,
                "status": "covered" if steps else "no_skeleton_steps",
            }
        )

    summary = {
        "demo_count": len(skeletons),
        "covered_demos": sum(1 for row in skeletons if row["status"] == "covered"),
        "operator_counts": dict(Counter(step["operator"] for row in skeletons for step in row["steps"])),
        "unresolved_count": sum(len(row["unresolved"]) for row in skeletons),
        "relation_lookback": relation_lookback,
    }
    return skeletons, summary


def build_object_centric_episodes(
    skeletons: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair object-centric pick/place steps and keep non-pick-place actions.

    The output is intentionally evidence preserving: every complete
    pick-place episode points back to its source pick step and place step.
    """
    episodes: list[dict[str, Any]] = []
    for skeleton in skeletons:
        active_pick: dict[str, dict[str, Any]] = {}
        for step in sorted(skeleton.get("steps", []), key=lambda item: (int(item["t_start"]), int(item["t_end"]))):
            op = str(step["operator"])
            bindings = dict(step.get("bindings") or {})

            if op == "pick":
                obj = bindings.get("obj")
                if obj:
                    active_pick[obj] = dict(step)
                continue

            if op == "place":
                obj = bindings.get("obj")
                target = bindings.get("target")
                relation = bindings.get("relation", "on-or-in")
                pick_step = active_pick.pop(obj, None) if obj else None
                episodes.append(
                    {
                        "episode_type": "pick_place" if pick_step else "place_without_pick",
                        "demo_id": skeleton["demo_id"],
                        "file": skeleton["file"],
                        "demo": skeleton["demo"],
                        "task": skeleton["task"],
                        "obj": obj,
                        "target": target,
                        "relation": relation,
                        "pick_t": pick_step.get("t_start") if pick_step else None,
                        "place_t": step.get("t_start"),
                        "duration": (int(step["t_start"]) - int(pick_step["t_start"])) if pick_step else None,
                        "confidence": "risky"
                        if (step.get("confidence") == "risky" or (pick_step and pick_step.get("confidence") == "risky"))
                        else "trusted",
                        "risk_notes": sorted(
                            set((step.get("risk_notes") or []) + ((pick_step or {}).get("risk_notes") or []))
                        ),
                        "source_pick": pick_step,
                        "source_place": step,
                    }
                )
                continue

            if op in {"open", "turn_on"}:
                episodes.append(
                    {
                        "episode_type": op,
                        "demo_id": skeleton["demo_id"],
                        "file": skeleton["file"],
                        "demo": skeleton["demo"],
                        "task": skeleton["task"],
                        "bindings": bindings,
                        "t": step.get("t_start"),
                        "confidence": step.get("confidence", "trusted"),
                        "risk_notes": list(step.get("risk_notes") or []),
                        "source_step": step,
                    }
                )

        for obj, pick_step in sorted(active_pick.items()):
            episodes.append(
                {
                    "episode_type": "pick_without_place",
                    "demo_id": skeleton["demo_id"],
                    "file": skeleton["file"],
                    "demo": skeleton["demo"],
                    "task": skeleton["task"],
                    "obj": obj,
                    "pick_t": pick_step.get("t_start"),
                    "confidence": pick_step.get("confidence", "trusted"),
                    "risk_notes": list(pick_step.get("risk_notes") or []),
                    "source_pick": pick_step,
                }
            )

    summary = {
        "episode_count": len(episodes),
        "episode_type_counts": dict(Counter(item["episode_type"] for item in episodes)),
        "trusted_count": sum(1 for item in episodes if item.get("confidence") == "trusted"),
        "risky_count": sum(1 for item in episodes if item.get("confidence") == "risky"),
        "per_task": {
            task: dict(counter)
            for task, counter in sorted(
                _counter_by_task_and_type(episodes).items(),
                key=lambda item: item[0],
            )
        },
    }
    return episodes, summary


def _counter_by_task_and_type(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Counter]:
    counters: dict[str, Counter] = defaultdict(Counter)
    for item in episodes:
        counters[str(item.get("task", ""))][str(item.get("episode_type", ""))] += 1
    return counters


def infer_operator_clauses_v1(episodes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Build evidence-backed operator-clause templates from object episodes."""
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_type[str(episode["episode_type"])].append(episode)

    clauses: list[dict[str, Any]] = []
    if by_type.get("pick_place") or by_type.get("pick_without_place"):
        pick_support = by_type.get("pick_place", []) + by_type.get("pick_without_place", [])
        clauses.append(
            {
                "name": "pick",
                "parameters": ["?obj"],
                "preconditions_pos": [["handempty"]],
                "preconditions_neg": [["holding", "?obj"]],
                "effects_add": [["holding", "?obj"]],
                "effects_del": [["handempty"]],
                "support_count": len(pick_support),
                "evidence_episode_types": dict(Counter(item["episode_type"] for item in pick_support)),
                "confidence": "conditional",
                "risk_notes": [
                    "Derived from holding transition under >=3-frame risk mask.",
                    "Object clear/support preconditions are not yet induced in v1.",
                ],
                "examples": [_episode_ref(item) for item in pick_support[:5]],
            }
        )

    if by_type.get("pick_place") or by_type.get("place_without_pick"):
        place_support = by_type.get("pick_place", []) + by_type.get("place_without_pick", [])
        target_counts = Counter(str(item.get("target")) for item in place_support)
        relation_counts = Counter(str(item.get("relation", "on-or-in")) for item in place_support)
        clauses.append(
            {
                "name": "place",
                "parameters": ["?obj", "?target"],
                "preconditions_pos": [["holding", "?obj"]],
                "preconditions_neg": [],
                "effects_add": [["handempty"], ["on-or-in", "?obj", "?target"]],
                "effects_del": [["holding", "?obj"]],
                "support_count": len(place_support),
                "target_counts": dict(target_counts),
                "relation_counts": dict(relation_counts),
                "confidence": "conditional",
                "risk_notes": [
                    "Target relation may appear before release; v1 uses fused skeleton evidence.",
                    "on(table) remains conditional under P2v4.2 risk controls.",
                ],
                "examples": [_episode_ref(item) for item in place_support[:5]],
            }
        )

    if by_type.get("open"):
        clauses.append(
            {
                "name": "open",
                "parameters": ["?drawer"],
                "preconditions_pos": [["closed", "?drawer"]],
                "preconditions_neg": [],
                "effects_add": [["open", "?drawer"]],
                "effects_del": [["closed", "?drawer"]],
                "support_count": len(by_type["open"]),
                "confidence": "trusted",
                "risk_notes": [],
                "examples": [_episode_ref(item) for item in by_type["open"][:5]],
            }
        )

    if by_type.get("turn_on"):
        clauses.append(
            {
                "name": "turn_on",
                "parameters": ["?device"],
                "preconditions_pos": [["deactivated", "?device"]],
                "preconditions_neg": [],
                "effects_add": [["activated", "?device"]],
                "effects_del": [["deactivated", "?device"]],
                "support_count": len(by_type["turn_on"]),
                "confidence": "trusted",
                "risk_notes": [],
                "examples": [_episode_ref(item) for item in by_type["turn_on"][:5]],
            }
        )

    report_lines = ["# P3 Operator Clauses v1", ""]
    for clause in clauses:
        report_lines.extend(
            [
                f"## {clause['name']}",
                "",
                f"- Support: {clause['support_count']}",
                f"- Confidence: {clause['confidence']}",
                f"- Preconditions +: `{clause['preconditions_pos']}`",
                f"- Preconditions -: `{clause['preconditions_neg']}`",
                f"- Effects +: `{clause['effects_add']}`",
                f"- Effects -: `{clause['effects_del']}`",
                f"- Risk notes: `{clause['risk_notes']}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Scope",
            "",
            "These clauses are v1 templates induced from P3 object-centric episodes.",
            "They are not final PDDL operators yet; P3-T7 should run counterexample",
            "checks against task goals and revise preconditions/effects.",
            "",
        ]
    )
    return clauses, "\n".join(report_lines)


def _episode_ref(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "demo_id": item.get("demo_id"),
        "task": item.get("task"),
        "obj": item.get("obj"),
        "target": item.get("target"),
        "relation": item.get("relation"),
        "pick_t": item.get("pick_t"),
        "place_t": item.get("place_t"),
        "t": item.get("t"),
    }


def frame_atom_sets_by_demo(frames: Sequence[NormalizedFrame]) -> dict[str, dict[str, set[Atom]]]:
    grouped: dict[str, list[NormalizedFrame]] = defaultdict(list)
    for frame in frames:
        grouped[f"{frame.file}::{frame.demo}"].append(frame)
    result: dict[str, dict[str, set[Atom]]] = {}
    for demo_id, demo_frames in grouped.items():
        ordered = sorted(demo_frames, key=lambda frame: frame.t)
        if not ordered:
            continue
        result[demo_id] = {
            "initial": set(ordered[0].trusted),
            "final": set(ordered[-1].trusted),
        }
    return result


def canonical_goal_arg(name: str, predicate: str) -> str:
    """Normalize BDDL goal arguments into P3 predicate atom names."""
    value = canonical_entity(name)
    if predicate in {"open", "in"} and value.startswith("wooden_cabinet_1_") and value.endswith("_region"):
        return value[: -len("_region")] + "_level"
    return value


def _goal_arg_aliases(goal_arg: str, grounded: Mapping[str, Any] | None, predicate: str) -> list[str]:
    aliases = {canonical_goal_arg(goal_arg, predicate)}
    if grounded:
        for key in ("bddl_name", "sim_joint", "sim_site", "sim_body"):
            value = grounded.get(key)
            if value:
                aliases.add(canonical_goal_arg(canonical_entity(str(value)), predicate))
        sim_body = grounded.get("sim_body")
        if sim_body and str(sim_body).endswith("_main"):
            aliases.add(str(sim_body)[: -len("_main")])
    if goal_arg.endswith("_top_region"):
        aliases.add(goal_arg[: -len("_top_region")] + "_main")
    if goal_arg.endswith("_cook_region"):
        aliases.add(goal_arg)
        aliases.add(goal_arg[: -len("_cook_region")] + "_main")
        aliases.add(goal_arg[: -len("_cook_region")] + "_burner_plate")
    if goal_arg.endswith("_top_side"):
        aliases.add(goal_arg[: -len("_top_side")] + "_top_level")
    return sorted(aliases)


def goal_specs_from_mapping(package_dir: Path) -> dict[str, list[dict[str, Any]]]:
    mapping = read_json(package_dir / "metadata" / "bddl_task_mapping.json")["mappings"]
    goals_by_task: dict[str, list[dict[str, Any]]] = {}
    for task, info in mapping.items():
        goals: list[dict[str, Any]] = []
        for goal in info.get("goals", []):
            pred = str(goal.get("internal_predicate"))
            bddl_args = [str(arg) for arg in goal.get("bddl_args", [])]
            grounded_args = list(goal.get("grounded_args", []))
            arg_aliases = [
                _goal_arg_aliases(arg, grounded_args[idx] if idx < len(grounded_args) else None, pred)
                for idx, arg in enumerate(bddl_args)
            ]
            if pred == "on":
                atom = ("on", canonical_goal_arg(bddl_args[0], pred), canonical_goal_arg(bddl_args[1], pred))
            elif pred == "in":
                atom = ("in", canonical_goal_arg(bddl_args[0], pred), canonical_goal_arg(bddl_args[1], pred))
            elif pred == "open":
                atom = ("open", canonical_goal_arg(bddl_args[0], pred))
            elif pred == "activated":
                atom = ("activated", canonical_goal_arg(bddl_args[0], pred))
            elif pred == "at_region":
                atom = ("at_region", canonical_goal_arg(bddl_args[0], pred), canonical_goal_arg(bddl_args[1], pred))
            else:
                atom = tuple([pred, *[canonical_goal_arg(arg, pred) for arg in bddl_args]])
            goals.append(
                {
                    "atom": atom,
                    "arg_aliases": arg_aliases,
                    "bddl_predicate": goal.get("bddl_predicate"),
                    "bddl_args": bddl_args,
                    "internal_predicate": pred,
                }
            )
        goals_by_task[task] = goals
    return goals_by_task


def repair_skeletons_with_goal_effects(
    skeletons: Sequence[Mapping[str, Any]],
    goals_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    atom_sets_by_demo: Mapping[str, Mapping[str, set[Atom]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add a risky v2 step when final state proves a missed goal effect.

    This is not allowed to erase the v1 evidence.  It creates a separate v2
    skeleton that says: "the final PredicateTrace contains the goal effect, but
    the event binding failed to turn it into an operator step."  That is exactly
    the counterexample-driven revision signal P3 needs.
    """
    repaired: list[dict[str, Any]] = []
    added_repairs: list[dict[str, Any]] = []

    for skeleton in skeletons:
        task = str(skeleton.get("task", ""))
        demo_id = str(skeleton.get("demo_id"))
        goals = list(goals_by_task.get(task, []))
        frame_sets = atom_sets_by_demo.get(demo_id, {})
        initial_atoms = set(frame_sets.get("initial", set()))
        final_atoms = set(frame_sets.get("final", set()))
        steps = [dict(step) for step in skeleton.get("steps", [])]

        for spec in goals:
            atom = tuple(spec["atom"])
            if atom[0] == "at_region":
                continue
            if any(step_satisfies_goal(step, spec) for step in steps):
                continue
            if not _goal_spec_matches_atom_set(spec, final_atoms):
                continue
            if _goal_spec_matches_atom_set(spec, initial_atoms):
                # Goal already true at t=0 is not an operator effect.
                continue

            repair_step = _repair_step_from_goal_spec(spec, skeleton, len(steps))
            if repair_step is None:
                continue
            steps.append(repair_step)
            added_repairs.append(
                {
                    "demo_id": demo_id,
                    "task": task,
                    "goal": atom_to_list(atom),
                    "operator": repair_step["operator"],
                    "bindings": repair_step["bindings"],
                    "reason": "final_state_goal_effect_not_bound_to_event",
                }
            )

        repaired_skeleton = dict(skeleton)
        steps.sort(key=lambda item: (int(item.get("t_start", 10**9)), int(item.get("t_end", 10**9))))
        repaired_skeleton["steps"] = steps
        repaired_skeleton["operator_sequence"] = [str(step["operator"]) for step in steps]
        repaired_skeleton["status"] = "covered" if steps else "no_skeleton_steps"
        repaired.append(repaired_skeleton)

    summary = {
        "input_skeletons": len(skeletons),
        "repaired_skeletons": len(repaired),
        "repair_count": len(added_repairs),
        "repair_operator_counts": dict(Counter(item["operator"] for item in added_repairs)),
        "repair_examples": added_repairs[:20],
    }
    return repaired, summary


def _goal_spec_matches_atom_set(spec: Mapping[str, Any], atoms: set[Atom]) -> bool:
    return any(_atom_satisfies_goal_spec(atom, spec) for atom in atoms)


def _atom_satisfies_goal_spec(atom: Atom, spec: Mapping[str, Any]) -> bool:
    goal = tuple(spec["atom"])
    aliases = spec.get("arg_aliases") or []
    if atom[0] != goal[0]:
        return False
    if len(goal) == 2 and len(atom) == 2:
        return _value_matches_aliases(atom[1], aliases[0] if aliases else [goal[1]])
    if len(goal) == 3 and len(atom) == 3:
        return _value_matches_aliases(atom[1], aliases[0] if aliases else [goal[1]]) and _value_matches_aliases(
            atom[2], aliases[1] if len(aliases) > 1 else [goal[2]]
        )
    return atom == goal


def _repair_step_from_goal_spec(
    spec: Mapping[str, Any],
    skeleton: Mapping[str, Any],
    step_index: int,
) -> dict[str, Any] | None:
    atom = tuple(spec["atom"])
    t_estimate = _skeleton_end_time(skeleton) + 1 + step_index
    if atom[0] in {"on", "in"}:
        return {
            "operator": "place",
            "bindings": {"obj": atom[1], "target": atom[2], "relation": atom[0]},
            "t_start": t_estimate,
            "t_end": t_estimate,
            "source": "goal_implied_final_state_repair",
            "confidence": "risky",
            "risk_notes": [
                "goal_implied_effect_from_final_state",
                "event_binding_missed_goal_relation",
            ],
        }
    if atom[0] == "open":
        return {
            "operator": "open",
            "bindings": {"drawer": atom[1]},
            "t_start": t_estimate,
            "t_end": t_estimate,
            "source": "goal_implied_final_state_repair",
            "confidence": "risky",
            "risk_notes": ["goal_implied_effect_from_final_state"],
        }
    if atom[0] == "activated":
        return {
            "operator": "turn_on",
            "bindings": {"device": atom[1]},
            "t_start": t_estimate,
            "t_end": t_estimate,
            "source": "goal_implied_final_state_repair",
            "confidence": "risky",
            "risk_notes": ["goal_implied_effect_from_final_state"],
        }
    return None


def _skeleton_end_time(skeleton: Mapping[str, Any]) -> int:
    times = [int(step.get("t_end", step.get("t_start", 0))) for step in skeleton.get("steps", [])]
    return max(times) if times else 0


def goal_atoms_from_mapping(package_dir: Path) -> dict[str, list[Atom]]:
    return {
        task: [tuple(spec["atom"]) for spec in specs]
        for task, specs in goal_specs_from_mapping(package_dir).items()
    }


def step_satisfies_goal(step: Mapping[str, Any], goal: Atom | Mapping[str, Any]) -> bool:
    op = str(step.get("operator"))
    bindings = dict(step.get("bindings") or {})
    if isinstance(goal, Mapping):
        aliases = goal.get("arg_aliases") or []
        goal_atom = tuple(goal["atom"])
    else:
        aliases = []
        goal_atom = goal
    goal = goal_atom
    pred = goal[0]

    if pred == "open" and op == "open":
        return _value_matches_aliases(bindings.get("drawer"), aliases[0] if aliases else [goal[1]])
    if pred == "activated" and op == "turn_on":
        device = str(bindings.get("device", ""))
        candidates = aliases[0] if aliases else [goal[1]]
        return any(device == item or device.startswith(item) or item.startswith(device) for item in candidates)
    if pred in {"on", "in"} and op == "place":
        relation = bindings.get("relation", "on-or-in")
        obj_aliases = aliases[0] if aliases else [goal[1]]
        target_aliases = aliases[1] if len(aliases) > 1 else [goal[2]]
        return (
            _value_matches_aliases(bindings.get("obj"), obj_aliases)
            and _value_matches_aliases(bindings.get("target"), target_aliases)
            and (relation == pred or relation == "on-or-in")
        )
    return False


def _value_matches_aliases(value: Any, aliases: Sequence[str]) -> bool:
    if value is None:
        return False
    value_s = str(value)
    return any(value_s == alias or value_s.startswith(alias) or alias.startswith(value_s) for alias in aliases)


def replay_goals_against_skeletons(
    skeletons: Sequence[Mapping[str, Any]],
    goals_by_task: Mapping[str, Sequence[Atom | Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Audit whether fused P3 skeletons can explain each task goal."""
    rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []

    for skeleton in skeletons:
        task = str(skeleton.get("task", ""))
        goals = list(goals_by_task.get(task, []))
        steps = list(skeleton.get("steps", []))
        matched: list[Atom] = []
        missing: list[Atom] = []

        goal_atoms = [tuple(goal["atom"]) if isinstance(goal, Mapping) else tuple(goal) for goal in goals]

        if any(goal[0] == "at_region" for goal in goal_atoms):
            status = "blocked_excluded_predicate"
            reason = "goal_depends_on_at_region_excluded_from_P3"
            missing = goal_atoms
        else:
            for goal in goals:
                atom = tuple(goal["atom"]) if isinstance(goal, Mapping) else tuple(goal)
                if any(step_satisfies_goal(step, goal) for step in steps):
                    matched.append(atom)
                else:
                    missing.append(atom)
            if not goals:
                status = "no_goal_mapping"
                reason = "missing_bddl_goal_mapping"
            elif not steps:
                status = "failed"
                reason = "no_skeleton_steps"
            elif not missing:
                status = "covered"
                reason = "all_goals_matched_by_skeleton"
            else:
                status = "failed"
                reason = infer_goal_failure_reason(steps, missing)

        row = {
            "demo_id": skeleton.get("demo_id"),
            "file": skeleton.get("file"),
            "demo": skeleton.get("demo"),
            "task": task,
            "goals": [atom_to_list(goal) for goal in goal_atoms],
            "goal_specs": [
                {
                    "atom": atom_to_list(tuple(goal["atom"])),
                    "arg_aliases": goal.get("arg_aliases", []),
                }
                for goal in goals
                if isinstance(goal, Mapping)
            ],
            "matched_goals": [atom_to_list(goal) for goal in matched],
            "missing_goals": [atom_to_list(goal) for goal in missing],
            "operator_sequence": list(skeleton.get("operator_sequence", [])),
            "status": status,
            "reason": reason,
        }
        rows.append(row)
        if status != "covered":
            counterexamples.append(row)

    status_counts = Counter(row["status"] for row in rows)
    reason_counts = Counter(row["reason"] for row in rows if row["status"] != "covered")
    per_task: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_task[str(row["task"])][str(row["status"])] += 1
    report = {
        "demo_count": len(rows),
        "covered": status_counts.get("covered", 0),
        "failed": status_counts.get("failed", 0),
        "blocked_excluded_predicate": status_counts.get("blocked_excluded_predicate", 0),
        "no_goal_mapping": status_counts.get("no_goal_mapping", 0),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "per_task": {task: dict(counts) for task, counts in sorted(per_task.items())},
    }
    markdown = render_goal_replay_report(report, counterexamples)
    return {"summary": report, "demos": rows}, counterexamples, markdown


def infer_goal_failure_reason(steps: Sequence[Mapping[str, Any]], missing: Sequence[Atom]) -> str:
    operators = [str(step.get("operator")) for step in steps]
    if any(goal[0] in {"on", "in"} for goal in missing):
        if "place" not in operators:
            return "missing_place_step"
        return "place_target_or_relation_mismatch"
    if any(goal[0] == "open" for goal in missing):
        return "missing_open_step"
    if any(goal[0] == "activated" for goal in missing):
        return "missing_turn_on_step"
    return "missing_goal_effect"


def render_goal_replay_report(report: Mapping[str, Any], counterexamples: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# P3 Goal Replay and Counterexample Report",
        "",
        f"- Demo count: {report['demo_count']}",
        f"- Covered: {report['covered']}",
        f"- Failed: {report['failed']}",
        f"- Blocked by excluded predicate: {report['blocked_excluded_predicate']}",
        f"- Status counts: `{report['status_counts']}`",
        f"- Reason counts: `{report['reason_counts']}`",
        "",
        "## Per-task status",
        "",
        "| Task | covered | failed | blocked |",
        "|---|---:|---:|---:|",
    ]
    for task, counts in report["per_task"].items():
        lines.append(
            f"| {task} | {counts.get('covered', 0)} | {counts.get('failed', 0)} | {counts.get('blocked_excluded_predicate', 0)} |"
        )
    lines.extend(["", "## Counterexamples", ""])
    for row in counterexamples[:40]:
        lines.extend(
            [
                f"### {row['demo_id']}",
                "",
                f"- Task: `{row['task']}`",
                f"- Status: `{row['status']}`",
                f"- Reason: `{row['reason']}`",
                f"- Goals: `{row['goals']}`",
                f"- Missing: `{row['missing_goals']}`",
                f"- Sequence: `{row['operator_sequence']}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_counterexample_revision_plan(counterexamples: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in counterexamples:
        grouped[str(item["reason"])].append(item)

    recommendations = {
        "goal_depends_on_at_region_excluded_from_P3": {
            "priority": "server_required",
            "action": "Validate replacement at_region queries and re-enable push/at_region event extraction.",
        },
        "no_skeleton_steps": {
            "priority": "high",
            "action": "Inspect predicate trace windows; likely holding/on changes are masked or object-specific goal entities differ from bowl-only assumptions.",
        },
        "missing_place_step": {
            "priority": "high",
            "action": "Improve release-target fusion and allow final-state goal relation to supply target when release is missing.",
        },
        "place_target_or_relation_mismatch": {
            "priority": "medium",
            "action": "Use goal-aware target adjudication: when multiple relocate_effect targets exist before release, prefer the BDDL goal-compatible target.",
        },
        "missing_turn_on_step": {
            "priority": "medium",
            "action": "Inspect activated/deactivated transition window.",
        },
        "missing_open_step": {
            "priority": "medium",
            "action": "Inspect open/closed transition window.",
        },
    }

    plan = {
        "counterexample_count": len(counterexamples),
        "reason_counts": {reason: len(items) for reason, items in sorted(grouped.items())},
        "groups": [],
    }
    for reason, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        rec = recommendations.get(
            reason,
            {"priority": "medium", "action": "Inspect examples and add a reason-specific revision rule."},
        )
        plan["groups"].append(
            {
                "reason": reason,
                "count": len(items),
                "priority": rec["priority"],
                "recommended_action": rec["action"],
                "example_demo_ids": [str(item["demo_id"]) for item in items[:8]],
                "tasks": sorted(set(str(item["task"]) for item in items)),
            }
        )

    md = ["# P3 Counterexample-driven Revision Plan", ""]
    md.extend(
        [
            f"- Counterexamples: {plan['counterexample_count']}",
            f"- Reason counts: `{plan['reason_counts']}`",
            "",
            "## Revision groups",
            "",
        ]
    )
    for group in plan["groups"]:
        md.extend(
            [
                f"### {group['reason']}",
                "",
                f"- Count: {group['count']}",
                f"- Priority: `{group['priority']}`",
                f"- Action: {group['recommended_action']}",
                f"- Tasks: `{group['tasks']}`",
                f"- Examples: `{group['example_demo_ids']}`",
                "",
            ]
        )
    return plan, "\n".join(md)


def write_markdown_reports(
    out_dir: Path,
    event_summary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    skeleton_summary: Mapping[str, Any],
) -> None:
    event_report = [
        "# P3 Effect Event Report",
        "",
        f"- Event count: {event_summary['event_count']}",
        f"- Demo count: {event_summary['demo_count']}",
        f"- Confidence counts: `{event_summary['event_confidence_counts']}`",
        f"- Changed predicate counts: `{event_summary['changed_predicate_counts']}`",
        "",
        "## Notes",
        "",
        "- Events are extracted from trusted predicates after P2v4.2 risk masks.",
        "- `at_region` is excluded.",
        "- `holding` requires >=3 consecutive frames.",
        "- `on(table)` is masked when a specific support exists in the same frame.",
        "",
    ]
    (out_dir / "p3_effect_event_report.md").write_text("\n".join(event_report), encoding="utf-8")

    candidate_lines = ["# P3 Operator Candidate Report", ""]
    for candidate in candidates:
        candidate_lines.extend(
            [
                f"## {candidate['name']}",
                "",
                f"- Support: {candidate['support_count']}",
                f"- Confidence: {candidate['confidence']}",
                f"- Parameters: `{candidate['parameters']}`",
                f"- Effects add: `{candidate['effects_add_template']}`",
                f"- Effects del: `{candidate['effects_del_template']}`",
                f"- Risk notes: `{candidate['risk_notes']}`",
                "",
            ]
        )
    (out_dir / "p3_operator_candidate_report.md").write_text("\n".join(candidate_lines), encoding="utf-8")

    replay_lines = [
        "# P3 Plan Skeleton Replay Report",
        "",
        f"- Demo count: {replay['demo_count']}",
        f"- Covered demos: {replay['covered_demos']}",
        f"- Operator counts: `{replay['operator_counts']}`",
        f"- Fused skeleton covered demos: {skeleton_summary['covered_demos']}/{skeleton_summary['demo_count']}",
        f"- Fused skeleton operator counts: `{skeleton_summary['operator_counts']}`",
        f"- Unresolved release count: {skeleton_summary['unresolved_count']}",
        "",
        "## Interpretation",
        "",
        "This is a skeleton replay audit: it checks whether predicate changes can be mapped to",
        "operator-like events, not whether a planner can yet solve every task from scratch.",
        "The fused skeleton view links release events to recent on/in effects because physical",
        "contact relations can appear before gripper release in the predicate trace.",
        "",
    ]
    (out_dir / "p3_plan_replay_report.md").write_text("\n".join(replay_lines), encoding="utf-8")


def build_input_contract(package_dir: Path) -> tuple[dict[str, Any], str]:
    files = sorted(path for path in package_dir.rglob("*") if path.is_file())
    hashes = {str(path.relative_to(package_dir)).replace("\\", "/"): sha256_file(path) for path in files}
    metrics = read_json(package_dir / "p2v4_2_human_eval_metrics.json")
    contract = {
        "package_dir": str(package_dir),
        "decision": "G2_CONDITIONAL_PASS",
        "metrics": metrics.get("overall", {}),
        "risk_controls": DEFAULT_MASK,
        "input_hashes": hashes,
    }
    md = [
        "# P3 Input Contract",
        "",
        "## Decision",
        "",
        "`P2v4.2 = G2_CONDITIONAL_PASS`，P3 在可信谓词掩码下启动。",
        "",
        "## Risk Controls",
        "",
        "- `in/open/activated/handempty` unrestricted.",
        "- `holding` requires >=3 consecutive frames.",
        "- `on(table)` is masked when specific support exists.",
        "- `at_region` is excluded until replacement queries are validated.",
        "",
        "## Metrics",
        "",
        f"```json\n{json.dumps(metrics.get('overall', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Input Hashes",
        "",
    ]
    md.extend(f"- `{name}`: `{digest}`" for name, digest in hashes.items())
    return contract, "\n".join(md) + "\n"


def run_p3_pipeline(package_dir: Path, out_dir: Path, holding_min_frames: int = 3, merge_gap: int = 3) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    contract, contract_md = build_input_contract(package_dir)
    write_json(out_dir / "p3_trusted_predicate_mask.json", DEFAULT_MASK)
    write_json(out_dir / "p3_input_contract.json", contract)
    (out_dir / "p3_input_contract.md").write_text(contract_md, encoding="utf-8")

    normalized, trace_report = normalize_traces(package_dir, holding_min_frames=holding_min_frames)
    write_jsonl(out_dir / "p3_normalized_traces.jsonl", [frame.to_dict() for frame in normalized])
    write_json(out_dir / "p3_trace_schema_report.json", trace_report)

    events, event_summary = extract_effect_events(normalized, merge_gap=merge_gap)
    write_jsonl(out_dir / "p3_effect_events.jsonl", events)
    write_json(out_dir / "p3_effect_event_summary.json", event_summary)

    candidates, classified = induce_operator_candidates(events)
    write_json(out_dir / "p3_operator_candidates.json", candidates)
    write_jsonl(out_dir / "p3_classified_events.jsonl", classified["classified_events"])
    write_json(out_dir / "p3_operator_induction_summary.json", classified["summary"])

    demo_index = collect_demo_index(normalized)
    replay = build_plan_replay_eval(classified["classified_events"], demo_index=demo_index)
    write_json(out_dir / "p3_plan_replay_eval.json", replay)
    skeletons, skeleton_summary = infer_plan_skeletons(classified["classified_events"], demo_index=demo_index)
    write_jsonl(out_dir / "p3_plan_skeletons.jsonl", skeletons)
    write_json(out_dir / "p3_plan_skeleton_summary.json", skeleton_summary)
    write_markdown_reports(out_dir, event_summary, candidates, replay, skeleton_summary)

    object_episodes, object_episode_summary = build_object_centric_episodes(skeletons)
    write_jsonl(out_dir / "p3_object_centric_episodes.jsonl", object_episodes)
    write_json(out_dir / "p3_object_centric_summary.json", object_episode_summary)
    clauses, clause_report = infer_operator_clauses_v1(object_episodes)
    write_json(out_dir / "p3_operator_clauses_v1.json", clauses)
    (out_dir / "p3_operator_clause_report.md").write_text(clause_report, encoding="utf-8")
    goal_replay, counterexamples, goal_report_md = replay_goals_against_skeletons(
        skeletons,
        goals_by_task=goal_specs_from_mapping(package_dir),
    )
    write_json(out_dir / "p3_goal_replay_eval.json", goal_replay)
    write_jsonl(out_dir / "p3_goal_counterexamples.jsonl", counterexamples)
    (out_dir / "p3_goal_replay_report.md").write_text(goal_report_md, encoding="utf-8")
    revision_plan, revision_plan_md = build_counterexample_revision_plan(counterexamples)
    write_json(out_dir / "p3_counterexample_revision_plan.json", revision_plan)
    (out_dir / "p3_counterexample_revision_plan.md").write_text(revision_plan_md, encoding="utf-8")

    goal_specs = goal_specs_from_mapping(package_dir)
    skeletons_v2, skeleton_repair_summary = repair_skeletons_with_goal_effects(
        skeletons,
        goals_by_task=goal_specs,
        atom_sets_by_demo=frame_atom_sets_by_demo(normalized),
    )
    write_jsonl(out_dir / "p3_plan_skeletons_v2.jsonl", skeletons_v2)
    write_json(out_dir / "p3_plan_skeleton_repair_summary.json", skeleton_repair_summary)
    object_episodes_v2, object_episode_summary_v2 = build_object_centric_episodes(skeletons_v2)
    write_jsonl(out_dir / "p3_object_centric_episodes_v2.jsonl", object_episodes_v2)
    write_json(out_dir / "p3_object_centric_summary_v2.json", object_episode_summary_v2)
    clauses_v2, clause_report_v2 = infer_operator_clauses_v1(object_episodes_v2)
    write_json(out_dir / "p3_operator_clauses_v2.json", clauses_v2)
    (out_dir / "p3_operator_clause_report_v2.md").write_text(clause_report_v2, encoding="utf-8")
    goal_replay_v2, counterexamples_v2, goal_report_md_v2 = replay_goals_against_skeletons(
        skeletons_v2,
        goals_by_task=goal_specs,
    )
    write_json(out_dir / "p3_goal_replay_eval_v2.json", goal_replay_v2)
    write_jsonl(out_dir / "p3_goal_counterexamples_v2.jsonl", counterexamples_v2)
    (out_dir / "p3_goal_replay_report_v2.md").write_text(goal_report_md_v2, encoding="utf-8")
    revision_plan_v2, revision_plan_md_v2 = build_counterexample_revision_plan(counterexamples_v2)
    write_json(out_dir / "p3_counterexample_revision_plan_v2.json", revision_plan_v2)
    (out_dir / "p3_counterexample_revision_plan_v2.md").write_text(revision_plan_md_v2, encoding="utf-8")

    final_summary = {
        "ok": True,
        "package_dir": str(package_dir),
        "out_dir": str(out_dir),
        "normalized_frames": len(normalized),
        "demos": trace_report["demos"],
        "events": event_summary["event_count"],
        "operator_counts": classified["summary"]["operator_counts"],
        "covered_demos": replay["covered_demos"],
        "demo_count": replay["demo_count"],
        "skeleton_covered_demos": skeleton_summary["covered_demos"],
        "skeleton_operator_counts": skeleton_summary["operator_counts"],
        "object_episode_counts": object_episode_summary["episode_type_counts"],
        "operator_clause_count": len(clauses),
        "goal_replay": goal_replay["summary"],
        "skeleton_repair": skeleton_repair_summary,
        "object_episode_counts_v2": object_episode_summary_v2["episode_type_counts"],
        "operator_clause_count_v2": len(clauses_v2),
        "goal_replay_v2": goal_replay_v2["summary"],
        "risk_controls": {
            "holding_min_frames": holding_min_frames,
            "merge_gap": merge_gap,
            "at_region_excluded": True,
        },
    }
    write_json(out_dir / "p3_pipeline_summary.json", final_summary)
    return final_summary
