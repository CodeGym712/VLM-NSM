from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _safe_load(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return None
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def collect_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for fpath in sorted(results_dir.iterdir()):
        if not fpath.name.startswith("language_variation_") or not fpath.name.endswith(".json"):
            continue
        if fpath.name.endswith("_summary.json"):
            continue
        data = _safe_load(fpath)
        if not data:
            continue
        for sample in data:
            rows.append(sample)
    return rows


def extract_repair_stats(rows: list[dict]) -> list[dict]:
    raw: list[dict[str, Any]] = []

    for r in rows:
        if not r.get("repaired"):
            continue
        repair_types = r.get("repair_types", [])
        if isinstance(repair_types, list):
            for rt in repair_types:
                raw.append({
                    "model_name": r.get("model_name", ""),
                    "task": r.get("task", ""),
                    "difficulty": r.get("difficulty", ""),
                    "prompt_variant": r.get("prompt_variant", ""),
                    "alignment_mode": r.get("alignment_mode", ""),
                    "repair_type": rt,
                    "sample_id": r.get("sample_id", ""),
                    "instruction": r.get("instruction", ""),
                    "raw_output_path": r.get("raw_output_path", ""),
                })

    groups: dict[tuple, dict] = defaultdict(lambda: {
        "count": 0, "sample_ids": [], "instructions": [], "raw_output_paths": [],
    })
    for item in raw:
        key = (
            item["model_name"],
            item["task"],
            item["difficulty"],
            item["prompt_variant"],
            item["alignment_mode"],
            item["repair_type"],
        )
        groups[key]["count"] += 1
        groups[key]["sample_ids"].append(item["sample_id"])
        groups[key]["instructions"].append(item["instruction"])
        groups[key]["raw_output_paths"].append(item["raw_output_path"])

    stats: list[dict] = []
    for key, g in sorted(groups.items()):
        (model_name, task, difficulty, prompt_variant, alignment_mode, repair_type) = key
        stats.append({
            "model_name": model_name,
            "task": task,
            "difficulty": difficulty,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "repair_type": repair_type,
            "count": g["count"],
            "example_sample_id": g["sample_ids"][0],
            "example_instruction": g["instructions"][0],
            "raw_output_path": g["raw_output_paths"][0] if g["raw_output_paths"] else "",
        })

    return stats


def main():
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    rows = collect_results(results_dir)
    if not rows:
        print("No language variation results found.")
        return

    stats = extract_repair_stats(rows)

    json_path = results_dir / "qwen_alignment_repair_stats.json"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Repair stats written to {json_path} ({len(stats)} rows)")

    csv_path = results_dir / "qwen_alignment_repair_stats.csv"
    if stats:
        fieldnames = list(stats[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)
    else:
        # Write header-only file
        fieldnames = [
            "model_name", "task", "difficulty", "prompt_variant",
            "alignment_mode", "repair_type", "count",
            "example_sample_id", "example_instruction", "raw_output_path",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print("No repairs observed under any setting.")
    print(f"Repair stats CSV written to {csv_path}")

    total = len(rows)
    repaired_count = sum(1 for r in rows if r.get("repaired"))
    print(f"\nTotal samples: {total}, Repaired: {repaired_count}")
    if stats:
        print(f"Unique repair type groups: {len(stats)}")
        for s in stats:
            print(f"  {s['model_name']} / {s['task']} / {s['repair_type']}: {s['count']} times "
                  f"(e.g. {s['example_sample_id']})")


if __name__ == "__main__":
    main()
