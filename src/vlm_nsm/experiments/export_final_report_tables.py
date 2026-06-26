"""Generate final Stage 8 report tables from all experiment results.

Usage:
    python3 -m vlm_nsm.experiments.export_final_report_tables

Output: generated/report_tables/final/*.md
"""

from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "generated" / "results"
TABLES_DIR = PROJECT_ROOT / "generated" / "report_tables" / "final"
DEMOS_DIR = PROJECT_ROOT / "generated" / "demos"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def _pct_str(numer: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{100 * numer / denom:.1f}%"


# ─── Table 1: Language Model Comparison ──────────────────────────────

def table_language_model_comparison() -> str:
    """Final language model comparison (hard set, full alignment)."""
    src = RESULTS_DIR / "language_variation_summary.json"
    data = _load_json(src)
    if not data:
        return "# Language Model Comparison\n\n(Data not available)\n"

    rows = []
    for d in data:
        model = d.get("model_name", "")
        task = d.get("task", "")
        diff = d.get("difficulty", "")
        prompt = d.get("prompt_variant", "")
        align = d.get("alignment_mode", "")
        if diff != "hard" or align != "full" or prompt not in ("full", "medium"):
            continue

        total = d.get("n_samples", 0)
        goal_ok = d.get("goal_success_count", 0) or 0
        json_ok = d.get("json_parse_count", 0) or 0
        schema_ok = d.get("schema_validity_count", 0) or 0
        align_ok = d.get("alignment_success_count", 0) or 0
        repair_ok = d.get("repair_count", 0) or 0
        infer_time = d.get("avg_inference_time", 0) or 0
        out_tokens = d.get("avg_output_tokens", 0) or 0

        align_str = _pct_str(align_ok, total) if align_ok is not None else "N/A"
        rows.append((
            model, task, prompt, total,
            _pct_str(json_ok, total),
            _pct_str(schema_ok, total),
            align_str,
            _pct_str(goal_ok, total),
            f"{infer_time:.1f}s",
            f"{out_tokens:.0f}",
        ))

    lines = [
        "# Final Language Model Comparison\n",
        "| Model | Task | Prompt | Samples | JSON Parse | Schema Valid | Alignment | Goal Success | Avg Time | Avg Tokens |",
        "|-------|:---:|:------:|:-------:|:----------:|:------------:|:---------:|:------------:|:--------:|:----------:|",
    ]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(*r))
    return "\n".join(lines) + "\n"


# ─── Table 2: Language Failure Taxonomy ──────────────────────────────

def table_language_failure_taxonomy() -> str:
    """Categorize all 556 failures by type, model, task, difficulty."""
    src = RESULTS_DIR / "qwen_failure_cases.json"
    data = _load_json(src)
    if not data:
        return "# Language Failure Taxonomy\n\n(Data not available)\n"

    # Count failures by type
    fail_counter: Counter = Counter()
    fail_by_model: dict[str, Counter] = defaultdict(Counter)
    fail_by_task: dict[str, Counter] = defaultdict(Counter)
    fail_by_diff: dict[str, Counter] = defaultdict(Counter)

    for d in data:
        ft = d.get("failure_type", "unknown")
        model = d.get("model_name", "unknown")
        task = d.get("task", "unknown")
        diff = d.get("difficulty", "unknown")
        fail_counter[ft] += 1
        fail_by_model[model][ft] += 1
        fail_by_task[task][ft] += 1
        fail_by_diff[diff][ft] += 1

    total_fails = sum(fail_counter.values())

    fail_order = [ft for ft, _ in fail_counter.most_common()]

    lines = [
        "# Language Failure Taxonomy\n",
        f"Total failures: {total_fails} across {len(data)} samples ({984 - total_fails} successes).\n",
        "### By Failure Type\n",
        "| Failure Type | Count | % of Total | Description |",
        "|:------------|:-----:|:----------:|-------------|",
    ]
    descriptions = {
        "schema_invalid": "Output does not match required JSON schema",
        "invalid_object": "References object not present in the environment",
        "goal_not_satisfied": "Parsed goal does not satisfy task constraints",
        "json_parse_error": "Model output is not valid JSON",
        "incomplete_goal": "Goal is missing required fields",
    }
    for ft in fail_order:
        cnt = fail_counter[ft]
        lines.append(f"| {ft} | {cnt} | {cnt / total_fails * 100:.1f}% | {descriptions.get(ft, '')} |")

    lines += ["\n### By Model\n", "| Model | " + " | ".join(ft for ft in fail_order) + " | Total |"]
    lines.append("|-------|" + "|".join(":---:" for _ in fail_order) + "|:----:|")
    for model in sorted(fail_by_model):
        vals = [str(fail_by_model[model].get(ft, 0)) for ft in fail_order]
        total_m = sum(fail_by_model[model].values())
        lines.append(f"| {model} | {' | '.join(vals)} | {total_m} |")

    lines += ["\n### By Task\n", "| Task | " + " | ".join(ft for ft in fail_order) + " | Total |"]
    lines.append("|------|" + "|".join(":---:" for _ in fail_order) + "|:----:|")
    for task in sorted(fail_by_task):
        vals = [str(fail_by_task[task].get(ft, 0)) for ft in fail_order]
        total_t = sum(fail_by_task[task].values())
        lines.append(f"| {task} | {' | '.join(vals)} | {total_t} |")

    return "\n".join(lines) + "\n"


# ─── Table 3: Language Prompt Comparison ─────────────────────────────

def table_language_prompt_comparison() -> str:
    """Compare full vs medium vs weak prompts."""
    src = RESULTS_DIR / "language_variation_summary.json"
    data = _load_json(src)
    if not data:
        return "# Language Prompt Comparison\n\n(Data not available)\n"

    rows = []
    for d in data:
        model = d.get("model_name", "")
        task = d.get("task", "")
        diff = d.get("difficulty", "")
        prompt = d.get("prompt_variant", "")
        align = d.get("alignment_mode", "")
        if diff != "hard" or align != "full":
            continue

        total = d.get("n_samples", 0) or 0
        goal_ok = d.get("goal_success_count", 0) or 0
        rows.append((model, task, prompt, total, goal_ok))

    lines = [
        "# Language Prompt Comparison\n",
        "| Model | Task | Prompt Variant | Samples | Goal Success |",
        "|-------|:---:|:--------------:|:-------:|:------------:|",
    ]
    for r in sorted(rows):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {_pct_str(r[4], r[3])} |")
    return "\n".join(lines) + "\n"


# ─── Table 4: Embodied Policy Comparison ─────────────────────────────

def table_embodied_policy_comparison() -> str:
    """Multi-seed policy comparison aggregating Stage 7B-5 data."""
    s8_dir = RESULTS_DIR / "stage_7B-5"

    policies = {
        "Scripted (scripted-only)": "scripted",
        "bc_main (expert-only BC)": "bc_main",
        "DAgger Reach BC (ours)": "dagger_reach",
    }

    header = [
        "| Policy | seed=0 | seed=1 | seed=2 | Avg | Min | Mean Steps | Mean BC Steps |",
        "|--------|:-----:|:-----:|:-----:|:---:|:---:|:----------:|:-------------:|",
    ]
    rows = []
    for label, prefix in policies.items():
        rates, steps, bc_steps = [], [], []
        for seed in range(3):
            fp = s8_dir / f"{prefix}_seed{seed}_20ep.json"
            if fp.exists():
                d = json.loads(fp.read_text())
                rates.append(d["success_rate"])
                steps.append(d["mean_steps"])
                bc_steps.append(d["mean_bc_steps"])
        if rates:
            avg_r = np.mean(rates) * 100
            min_r = min(rates) * 100
            avg_st = np.mean(steps)
            avg_bc = np.mean(bc_steps)
            rows.append(
                f"| {label} | {rates[0]*100:.0f}% | {rates[1]*100:.0f}% | {rates[2]*100:.0f}% "
                f"| {avg_r:.1f}% | {min_r:.0f}% | {avg_st:.0f} | {avg_bc:.0f} |"
            )

    # Add 50ep row
    fp50 = s8_dir / "dagger_reach_50ep.json"
    if fp50.exists():
        d50 = json.loads(fp50.read_text())
        rows.append(
            f"| DAgger Reach BC (50ep) | — | — | — | {d50['success_rate']*100:.1f}% | — | "
            f"{d50['mean_steps']:.0f} | {d50['mean_bc_steps']:.0f} |"
        )

    lines = ["# Embodied Policy Comparison\n"] + header + rows + [""]
    return "\n".join(lines)


# ─── Table 5: DAgger Ablation ────────────────────────────────────────

def table_dagger_ablation() -> str:
    """Data ablation: expert-only vs DAgger-only vs expert+DAgger."""
    s8_dir = RESULTS_DIR / "stage_7B-5"

    # Expert-only (multi-seed avg)
    bc_rates = []
    for s in range(3):
        fp = s8_dir / f"bc_main_seed{s}_20ep.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            bc_rates.append(d["success_rate"])
    bc_avg = np.mean(bc_rates) * 100 if bc_rates else 0

    # DAgger-only (seed=0)
    do_eval = _load_json(s8_dir / "dagger_only_bc_eval_20ep.json")

    # Expert+DAgger (multi-seed avg)
    dag_rates = []
    dag_steps = []
    for s in range(3):
        fp = s8_dir / f"dagger_reach_seed{s}_20ep.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            dag_rates.append(d["success_rate"])
            dag_steps.append(d["mean_steps"])
    dag_avg = np.mean(dag_rates) * 100 if dag_rates else 0
    dag_mean_st = np.mean(dag_steps) if dag_steps else 0

    do_rate = do_eval["success_rate"] * 100 if do_eval else 0
    do_steps = do_eval["mean_steps"] if do_eval else 0

    lines = [
        "# DAgger Data Ablation\n",
        "| Training Data | Episodes | Samples | Success Rate (avg) | Mean Steps | Notes |",
        "|:-------------|:--------:|:-------:|:-----------------:|:----------:|-------|",
        f"| Expert-only (bc_main) | 100 | 61,618 | {bc_avg:.1f}% | 744 | Baseline expert BC |",
        f"| DAgger-only | 30 | 18,820 | {do_rate:.0f}% | {do_steps:.0f} | Collapses — offline val loss 0.000627 but closed-loop 0% |",
        f"| Expert + DAgger (ours) | 130 | 80,438 | {dag_avg:.1f}% | {dag_mean_st:.0f} | +6.6pp over expert-only — DAgger improves state coverage |",
    ]
    return "\n".join(lines) + "\n"


# ─── Table 6: Phase Ablation ─────────────────────────────────────────

def table_phase_ablation() -> str:
    """Phase ablation results from Stage 7B-5."""
    s8_dir = RESULTS_DIR / "stage_7B-5"
    phase_data = {
        "reach_pick only": "phase_reach_pick_only_20ep.json",
        "reach_place only": "phase_reach_place_only_20ep.json",
        "Both reach (hybrid)": None,  # use dagger_reach_seed0
        "reach + grasp": "phase_reach_and_grasp_20ep.json",
        "No BC (scripted-only)": "scripted_seed0_20ep.json",
    }

    lines = [
        "# Phase Ablation\n",
        "| BC-controlled Phases | Success Rate | Mean Steps | BC Steps | Scripted Steps | Main Failure |",
        "|--------------------|:-----------:|:---------:|:--------:|:--------------:|:------------:|",
    ]

    for label, fname in phase_data.items():
        if fname is None:
            d = _load_json(s8_dir / "dagger_reach_seed0_20ep.json")
        else:
            d = _load_json(s8_dir / fname)
        if not d:
            continue
        fails = Counter(
            ep.get("failure_type") for ep in d["episode_results"] if not ep["success"]
        )
        main_fail = fails.most_common(1)[0][0] if fails else "—"
        lines.append(
            f"| {label} | {d['success_rate']*100:.0f}% | {d['mean_steps']:.0f} | "
            f"{d['mean_bc_steps']:.0f} | {d['mean_scripted_steps']:.0f} | {main_fail} |"
        )
    return "\n".join(lines) + "\n"


# ─── Table 7: Dataset Summary ────────────────────────────────────────

def table_dataset_summary() -> str:
    """Summarize all datasets created across stages."""
    # Expert v2 dataset
    coll = _load_json(
        DEMOS_DIR / "stage_7A-14" / "robosuite_expert_v2_100strict" / "collection_summary.json"
    )
    expert_attempts = coll["total_attempts"] if coll else "N/A"
    expert_strict = coll.get("strict_success_count", "N/A") if coll else "N/A"
    expert_rate = f"{coll['strict_success_rate']*100:.1f}%" if coll else "N/A"

    # Count npz files
    strict_npz = DEMOS_DIR / "stage_7A-14" / "robosuite_expert_v2_100strict" / "trajectories_strict_only.npz"
    strict_steps = None
    if strict_npz.exists():
        import numpy as np
        try:
            arr = np.load(strict_npz)
            strict_steps = len(arr["action"])
        except Exception:
            pass

    dagger_npz = (
        PROJECT_ROOT / "generated" / "demos" / "stage_7B-4"
        / "dagger_reach_aggregated" / "aggregated_reach_dataset.npz"
    )
    dagger_steps = None
    if dagger_npz.exists():
        try:
            arr = np.load(dagger_npz)
            dagger_steps = len(arr["action"])
        except Exception:
            pass

    lines = [
        "# Dataset Summary\n",
        "| Dataset | Episodes | Steps | Source | Used In |",
        "|---------|:--------:|:----:|:------|:--------|",
        f"| Expert v2 strict_only | 100 (in {expert_attempts} attempts) | {strict_steps or 'N/A'} | Scripted policy, true phase_id | bc_main, DAgger aggregation |",
        f"| DAgger round 1 (reach) | 30 | 18,820 | On-policy BC rollout + oracle actions | DAgger Reach BC training |",
        f"| Aggregated reach dataset | 130 | {dagger_steps or 'N/A'} | Expert v2 + DAgger merged | DAgger Reach BC (best model) |",
        "",
        f"Expert v2 collection efficiency: {expert_rate} strict success rate",
        "(93 strict + 7 semantic-only in 107 attempts = 100 valid strict demos).",
    ]
    return "\n".join(lines) + "\n"


# ─── Table 8: Artifact Manifest ──────────────────────────────────────

def table_artifact_manifest() -> str:
    """List all important artifacts with paths and descriptions."""
    artifacts = [
        ("Configs", "", ""),
        ("Execution config", "generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json",
         "Best execution parameters (kp, max_delta, state machines)"),
        ("", "", ""),
        ("Expert Demos", "", ""),
        ("Expert v2 strict_only (npz)", "generated/demos/stage_7A-14/robosuite_expert_v2_100strict/trajectories_strict_only.npz",
         "100 strict expert demos with true phase_id"),
        ("Expert v2 strict_only (dir)", "generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only/",
         "Per-episode npz files for BC training"),
        ("", "", ""),
        ("DAgger Datasets", "", ""),
        ("DAgger round 1", "generated/demos/stage_7B-4/dagger_round1_reach/trajectories_dagger_round1_reach.npz",
         "18,820 on-policy DAgger samples"),
        ("Aggregated reach dataset", "generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz",
         "Expert + DAgger merged (80,438 steps)"),
        ("", "", ""),
        ("Checkpoints", "", ""),
        ("DAgger Reach BC", "generated/results/stage_7B-4/dagger_reach_bc/best.pt",
         "Best learned controller: DAgger-trained, phase-conditioned, 72K params"),
        ("bc_main (expert-only)", "generated/results/stage_7B-3/bc_main/best.pt",
         "Expert-only BC baseline"),
        ("DAgger-only BC", "generated/results/stage_7B-5/dagger_only_bc/best.pt",
         "DAgger-only ablative (0% closed-loop)"),
        ("", "", ""),
        ("Embodied Eval Results", "", ""),
        ("Multi-seed comparison", "generated/results/stage_7B-5/",
         "Scripted/bc_main/DAgger over 3 seeds × 20ep"),
        ("DAgger 50ep validation", "generated/results/stage_7B-5/dagger_reach_50ep.json",
         "92.0% over 50 episodes"),
        ("Phase ablation", "generated/results/stage_7B-5/phase_*.json",
         "Phase ablation eval results"),
        ("Data ablation", "generated/results/stage_7B-5/dagger_only_bc_eval_20ep.json",
         "DAgger-only BC eval (0%)"),
        ("", "", ""),
        ("Language Eval Results", "", ""),
        ("Summary", "generated/results/language_variation_summary.json",
         "984 samples, 282 groups"),
        ("Failure cases", "generated/results/qwen_failure_cases.json",
         "556 failure cases with metadata"),
        ("Alignment repair stats", "generated/results/qwen_alignment_repair_stats.json",
         "9 repairs tracked"),
        ("", "", ""),
        ("Report Tables", "", ""),
        ("Stage 7B-5 tables", "generated/report_tables/stage_7B-5/",
         "Multi-seed, data ablation, phase ablation, failure audit"),
        ("Final tables", "generated/report_tables/final/",
         "Integrated final report tables"),
        ("", "", ""),
        ("Documentation", "", ""),
        ("Stage 7B-5 report", "docs/bc_stage7b5_report.md",
         "DAgger robust validation"),
        ("Final empirical report", "docs/final_empirical_report.md",
         "Integrated language + embodied report"),
        ("Evaluation protocol", "docs/evaluation_protocol.md",
         "Canonical and deprecated eval paths"),
        ("Artifact manifest", "docs/artifact_manifest.md",
         "This file"),
        ("", "", ""),
        ("Tests", "", ""),
        ("Stage 7B-5 tests", "tests/test_stage7b5_report_export.py",
         "Verify report tables and JSON results"),
        ("Policy comparison tests", "tests/test_policy_comparison_table.py",
         "Multi-seed policy validation"),
        ("Phase ablation tests", "tests/test_phase_ablation_table.py",
         "Phase ablation validation"),
        ("Failure audit tests", "tests/test_dagger_failure_audit.py",
         "Failure analysis tests"),
        ("Final report tests", "tests/test_final_report_tables.py",
         "Verify final Stage 8 tables"),
    ]

    lines = [
        "# Artifact Manifest\n",
        "| Category | Path | Description |",
        "|----------|:----:|-------------|",
    ]
    for cat, path, desc in artifacts:
        if path:
            lines.append(f"| {cat} | `{path}` | {desc} |")
        elif cat:
            lines.append(f"| **{cat}** | | |")
    return "\n".join(lines) + "\n"


# ─── Main ────────────────────────────────────────────────────────────

def main():
    out = _ensure_dir(TABLES_DIR)

    tables = {
        "final_language_model_comparison": table_language_model_comparison(),
        "final_language_failure_taxonomy": table_language_failure_taxonomy(),
        "final_language_prompt_comparison": table_language_prompt_comparison(),
        "final_embodied_policy_comparison": table_embodied_policy_comparison(),
        "final_dagger_ablation": table_dagger_ablation(),
        "final_phase_ablation": table_phase_ablation(),
        "final_dataset_summary": table_dataset_summary(),
        "final_artifact_manifest": table_artifact_manifest(),
    }

    for name, content in tables.items():
        fp = out / f"{name}.md"
        fp.write_text(content)
        print(f"  {fp.name} -> {fp}")

    print(f"\nAll tables written to {out}/")


if __name__ == "__main__":
    main()
