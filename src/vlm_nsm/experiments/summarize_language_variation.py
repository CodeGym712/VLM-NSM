from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _safe_load(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return None
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def _float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


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


def compute_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "total": 0, "json_ok": 0, "schema_ok": 0,
        "align_applicable": 0, "align_ok": 0, "repaired": 0,
        "planner_ok": 0, "exec_ok": 0, "goal_ok": 0,
        "total_inference_time": 0.0, "total_input_tokens": 0, "total_output_tokens": 0,
    })

    for r in rows:
        key = (
            r.get("model_name", ""),
            r.get("parser", ""),
            r.get("task", ""),
            r.get("n_blocks", 0),
            r.get("difficulty", ""),
            r.get("category", ""),
            r.get("prompt_variant", ""),
            r.get("alignment_mode", ""),
        )
        g = groups[key]
        g["total"] += 1
        if r.get("json_parse_success"):
            g["json_ok"] += 1
        if r.get("schema_validity"):
            g["schema_ok"] += 1
        # alignment_applicable defaults to True for old results
        if r.get("alignment_applicable", True):
            g["align_applicable"] += 1
            if r.get("alignment_success") is True:
                g["align_ok"] += 1
        if r.get("repaired"):
            g["repaired"] += 1
        if r.get("planner_success"):
            g["planner_ok"] += 1
        if r.get("execution_success"):
            g["exec_ok"] += 1
        if r.get("goal_success"):
            g["goal_ok"] += 1
        g["total_inference_time"] += _float(r.get("inference_time"))
        g["total_input_tokens"] += int(r.get("input_tokens") or 0)
        g["total_output_tokens"] += int(r.get("output_tokens") or 0)

    summary: list[dict] = []
    for key, g in sorted(groups.items()):
        (model_name, parser, task, n_blocks, difficulty, category,
         prompt_variant, alignment_mode) = key
        t = g["total"]
        aa = g["align_applicable"]
        row: dict = {
            "model_name": model_name,
            "parser": parser,
            "task": task,
            "n_blocks": n_blocks,
            "difficulty": difficulty,
            "category": category,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "sample_count": t,
            "json_parse_rate": f"{g['json_ok']}/{t}",
            "json_parse_pct": round(100 * g["json_ok"] / max(t, 1), 1),
            "schema_validity_rate": f"{g['schema_ok']}/{t}",
            "schema_validity_pct": round(100 * g["schema_ok"] / max(t, 1), 1),
            "repair_rate": f"{g['repaired']}/{t}",
            "repair_pct": round(100 * g["repaired"] / max(t, 1), 1),
            "planner_success_rate": f"{g['planner_ok']}/{t}",
            "planner_success_pct": round(100 * g["planner_ok"] / max(t, 1), 1),
            "execution_success_rate": f"{g['exec_ok']}/{t}",
            "execution_success_pct": round(100 * g["exec_ok"] / max(t, 1), 1),
            "goal_success_rate": f"{g['goal_ok']}/{t}",
            "goal_success_pct": round(100 * g["goal_ok"] / max(t, 1), 1),
            "avg_inference_time": round(g["total_inference_time"] / max(t, 1), 3),
            "avg_input_tokens": round(g["total_input_tokens"] / max(t, 1), 1),
            "avg_output_tokens": round(g["total_output_tokens"] / max(t, 1), 1),
        }
        if aa > 0:
            row["alignment_success_rate"] = f"{g['align_ok']}/{aa}"
            row["alignment_success_pct"] = round(100 * g["align_ok"] / aa, 1)
        else:
            row["alignment_success_rate"] = "N/A"
            row["alignment_success_pct"] = None
        summary.append(row)
    return summary


def main():
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    rows = collect_results(results_dir)
    if not rows:
        print("No language variation results found.")
        return

    summary = compute_summary(rows)
    summary_path = results_dir / "language_variation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary written to {summary_path}")
    print(f"Total samples: {len(rows)}, groups: {len(summary)}")

    csv_path = results_dir / "language_variation_summary.csv"
    if summary:
        fieldnames = list(summary[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
    print(f"CSV written to {csv_path}")

    # Print summary table
    print(f"\n{'model':<18} {'task':<10} {'diff':<6} {'category':<22} {'prompt':<6} {'align':<18} {'json%':<8} {'schema%':<8} {'align%':<8} {'repair%':<8} {'goal%':<8}")
    print("-" * 140)
    for s in summary:
        diff = s.get("difficulty", "")
        prompt = s.get("prompt_variant", "")
        align = s.get("alignment_mode", "")
        align_pct = s.get("alignment_success_pct")
        align_str = f"{align_pct}" if align_pct is not None else "N/A"
        print(f"{s['model_name']:<18} {s['task']:<10} {diff:<6} {s['category']:<22} {prompt:<6} {align:<18} {s['json_parse_pct']:<8} {s['schema_validity_pct']:<8} {align_str:<8} {s['repair_pct']:<8} {s['goal_success_pct']:<8}")


if __name__ == "__main__":
    main()
