from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def load_json(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return None
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def load_raw_results(results_dir: Path) -> list[dict]:
    """Load per-sample results from language_variation_*.json files."""
    rows: list[dict] = []
    for fpath in sorted(results_dir.iterdir()):
        if not fpath.name.startswith("language_variation_") or not fpath.name.endswith(".json"):
            continue
        if fpath.name.endswith("_summary.json"):
            continue
        data = load_json(fpath)
        if data:
            rows.extend(data)
    return rows


def _fraction_str(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{round(100 * numerator / denominator, 1)}"


def _alignment_pct(align_ok: int, align_total: int) -> str:
    if align_total == 0:
        return "N/A"
    return f"{round(100 * align_ok / align_total, 1)}"


def export_model_comparison(results: list[dict], output_path: Path) -> None:
    """Table 1: model comparison.
    Filters: difficulty=hard, prompt_variant in (full,medium), alignment_mode=full.
    Groups by: model_name + task + difficulty.
    """
    filtered = [
        r for r in results
        if r.get("difficulty") == "hard"
        and r.get("prompt_variant") in ("full", "medium")
        and r.get("alignment_mode") == "full"
    ]
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "samples": 0, "json_ok": 0, "schema_ok": 0,
        "align_ok": 0, "align_applicable": 0,
        "repaired": 0, "planner_ok": 0, "goal_ok": 0,
        "total_time": 0.0, "total_out": 0,
    })

    for r in filtered:
        key = (r.get("model_name", ""), r.get("task", ""), r.get("difficulty", ""))
        g = groups[key]
        g["samples"] += 1
        if r.get("json_parse_success"):
            g["json_ok"] += 1
        if r.get("schema_validity"):
            g["schema_ok"] += 1
        if r.get("alignment_applicable", True):
            g["align_applicable"] += 1
            if r.get("alignment_success") is True:
                g["align_ok"] += 1
        if r.get("repaired"):
            g["repaired"] += 1
        if r.get("planner_success"):
            g["planner_ok"] += 1
        if r.get("goal_success"):
            g["goal_ok"] += 1
        g["total_time"] += float(r.get("inference_time") or 0)
        g["total_out"] += int(r.get("output_tokens") or 0)

    fieldnames = [
        "model_name", "task", "difficulty", "samples",
        "json_parse_rate", "schema_validity_rate", "alignment_success_rate",
        "repair_rate", "goal_success_rate", "avg_inference_time", "avg_output_tokens",
    ]
    rows: list[dict] = []
    for (model_name, task, difficulty), g in sorted(groups.items()):
        t = g["samples"]
        rows.append({
            "model_name": model_name,
            "task": task,
            "difficulty": difficulty,
            "samples": t,
            "json_parse_rate": _fraction_str(g["json_ok"], t),
            "schema_validity_rate": _fraction_str(g["schema_ok"], t),
            "alignment_success_rate": _alignment_pct(g["align_ok"], g["align_applicable"]),
            "repair_rate": _fraction_str(g["repaired"], t),
            "goal_success_rate": _fraction_str(g["goal_ok"], t),
            "avg_inference_time": round(g["total_time"] / max(t, 1), 3),
            "avg_output_tokens": round(g["total_out"] / max(t, 1), 1),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Model comparison table -> {output_path} ({len(rows)} rows)")


def export_prompt_comparison(results: list[dict], output_path: Path) -> None:
    """Table 2: prompt comparison.
    Filters: model_name=qwen25_32b, difficulty=hard, alignment_mode=full.
    Groups by: model_name + task + difficulty + prompt_variant.
    """
    filtered = [
        r for r in results
        if r.get("model_name") == "qwen25_32b"
        and r.get("difficulty") == "hard"
        and r.get("alignment_mode") == "full"
    ]
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "samples": 0, "json_ok": 0, "schema_ok": 0,
        "align_ok": 0, "align_applicable": 0,
        "repaired": 0, "planner_ok": 0, "goal_ok": 0,
    })

    for r in filtered:
        key = (
            r.get("model_name", ""), r.get("task", ""),
            r.get("difficulty", ""), r.get("prompt_variant", ""),
        )
        g = groups[key]
        g["samples"] += 1
        if r.get("json_parse_success"):
            g["json_ok"] += 1
        if r.get("schema_validity"):
            g["schema_ok"] += 1
        if r.get("alignment_applicable", True):
            g["align_applicable"] += 1
            if r.get("alignment_success") is True:
                g["align_ok"] += 1
        if r.get("repaired"):
            g["repaired"] += 1
        if r.get("planner_success"):
            g["planner_ok"] += 1
        if r.get("goal_success"):
            g["goal_ok"] += 1

    fieldnames = [
        "model_name", "task", "difficulty", "prompt_variant", "samples",
        "json_parse_rate", "schema_validity_rate", "alignment_success_rate",
        "repair_rate", "goal_success_rate",
    ]
    rows: list[dict] = []
    for key, g in sorted(groups.items()):
        (model_name, task, difficulty, prompt_variant) = key
        t = g["samples"]
        rows.append({
            "model_name": model_name,
            "task": task,
            "difficulty": difficulty,
            "prompt_variant": prompt_variant,
            "samples": t,
            "json_parse_rate": _fraction_str(g["json_ok"], t),
            "schema_validity_rate": _fraction_str(g["schema_ok"], t),
            "alignment_success_rate": _alignment_pct(g["align_ok"], g["align_applicable"]),
            "repair_rate": _fraction_str(g["repaired"], t),
            "goal_success_rate": _fraction_str(g["goal_ok"], t),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prompt comparison table -> {output_path} ({len(rows)} rows)")


def export_alignment_benefit(results: list[dict], output_path: Path) -> None:
    """Table 3: alignment benefit.
    Filters: model_name=qwen25_32b, difficulty=hard, prompt_variant=medium.
    Groups by: model_name + task + difficulty + prompt_variant + alignment_mode.
    """
    filtered = [
        r for r in results
        if r.get("model_name") == "qwen25_32b"
        and r.get("difficulty") == "hard"
        and r.get("prompt_variant") == "medium"
    ]
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "samples": 0, "json_ok": 0, "schema_ok": 0,
        "align_ok": 0, "align_applicable": 0,
        "repaired": 0, "planner_ok": 0, "goal_ok": 0,
    })

    for r in filtered:
        key = (
            r.get("model_name", ""), r.get("task", ""),
            r.get("difficulty", ""), r.get("prompt_variant", ""),
            r.get("alignment_mode", ""),
        )
        g = groups[key]
        g["samples"] += 1
        if r.get("json_parse_success"):
            g["json_ok"] += 1
        if r.get("schema_validity"):
            g["schema_ok"] += 1
        if r.get("alignment_applicable", True):
            g["align_applicable"] += 1
            if r.get("alignment_success") is True:
                g["align_ok"] += 1
        if r.get("repaired"):
            g["repaired"] += 1
        if r.get("planner_success"):
            g["planner_ok"] += 1
        if r.get("goal_success"):
            g["goal_ok"] += 1

    fieldnames = [
        "model_name", "task", "difficulty", "prompt_variant",
        "alignment_mode", "samples",
        "json_parse_rate", "schema_validity_rate", "alignment_success_rate",
        "repair_rate", "planner_success_rate", "goal_success_rate",
    ]
    rows: list[dict] = []
    for key, g in sorted(groups.items()):
        (model_name, task, difficulty, prompt_variant, alignment_mode) = key
        t = g["samples"]
        rows.append({
            "model_name": model_name,
            "task": task,
            "difficulty": difficulty,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "samples": t,
            "json_parse_rate": _fraction_str(g["json_ok"], t),
            "schema_validity_rate": _fraction_str(g["schema_ok"], t),
            "alignment_success_rate": _alignment_pct(g["align_ok"], g["align_applicable"]),
            "repair_rate": _fraction_str(g["repaired"], t),
            "planner_success_rate": _fraction_str(g["planner_ok"], t),
            "goal_success_rate": _fraction_str(g["goal_ok"], t),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Alignment benefit table -> {output_path} ({len(rows)} rows)")


def export_by_category(results: list[dict], output_path: Path) -> None:
    """Table 4: by-category breakdown."""
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "samples": 0, "goal_ok": 0, "repaired": 0,
        "json_ok": 0, "schema_ok": 0,
        "align_ok": 0, "align_applicable": 0,
    })

    for r in results:
        key = (
            r.get("model_name", ""), r.get("task", ""),
            r.get("difficulty", ""), r.get("category", ""),
            r.get("prompt_variant", ""), r.get("alignment_mode", ""),
        )
        g = groups[key]
        g["samples"] += 1
        if r.get("goal_success"):
            g["goal_ok"] += 1
        if r.get("repaired"):
            g["repaired"] += 1
        if r.get("json_parse_success"):
            g["json_ok"] += 1
        if r.get("schema_validity"):
            g["schema_ok"] += 1
        if r.get("alignment_applicable", True):
            g["align_applicable"] += 1
            if r.get("alignment_success") is True:
                g["align_ok"] += 1

    fieldnames = [
        "model_name", "task", "difficulty", "category",
        "prompt_variant", "alignment_mode", "samples",
        "json_parse_rate", "schema_validity_rate",
        "alignment_success_rate", "repair_rate", "goal_success_rate",
    ]
    rows: list[dict] = []
    for key, g in sorted(groups.items()):
        (model_name, task, difficulty, category, prompt_variant, alignment_mode) = key
        t = g["samples"]
        rows.append({
            "model_name": model_name,
            "task": task,
            "difficulty": difficulty,
            "category": category,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "samples": t,
            "json_parse_rate": _fraction_str(g["json_ok"], t),
            "schema_validity_rate": _fraction_str(g["schema_ok"], t),
            "alignment_success_rate": _alignment_pct(g["align_ok"], g["align_applicable"]),
            "repair_rate": _fraction_str(g["repaired"], t),
            "goal_success_rate": _fraction_str(g["goal_ok"], t),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"By-category table -> {output_path} ({len(rows)} rows)")


def main():
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    results = load_raw_results(results_dir)
    if not results:
        print("No raw language variation results found.")
        return

    print(f"Loaded {len(results)} raw samples from {results_dir}")

    export_model_comparison(results, results_dir / "report_table_model_comparison.csv")
    export_prompt_comparison(results, results_dir / "report_table_prompt_comparison.csv")
    export_alignment_benefit(results, results_dir / "report_table_alignment_benefit.csv")
    export_by_category(results, results_dir / "report_table_by_category.csv")

    print("\nAll report tables generated.")


if __name__ == "__main__":
    main()
