"""Export embodied pipeline report tables from experiment results.

Reads:
  - generated/results/collect_stacking_demos_summary.json
  - generated/results/replay_sanity_check_summary.json
  - generated/results/symbol_filter_stats.json
  - generated/results/embodied_stacking_mock_summary.json
  - generated/results/embodied_stacking_mock_summary_noise.json (optional)
  - generated/results/embodied_stacking_robosuite_summary.json (optional)

Outputs:
  - generated/report_tables/embodied_mock_summary.md
  - generated/report_tables/embodied_mock_summary.csv
  - generated/report_tables/demo_dataset_summary.md
  - generated/report_tables/replay_sanity_summary.md
  - generated/report_tables/embodied_robosuite_summary.md (optional)
  - generated/report_tables/reality_gap_comparison.md (optional)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


REPORT_DIR = ensure_dir(PROJECT_ROOT / "generated" / "report_tables")
RESULTS_DIR = PROJECT_ROOT / "generated" / "results"


def _load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    return None


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    bar = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [sep, bar]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    return "\n".join(lines)


def _csv_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(c) for c in row))
    return "\n".join(lines)


def export_embodied_mock_summary(
    clean: dict | None,
    noise: dict | None,
):
    """Export embodied mock evaluation summary as MD + CSV."""
    csv_headers = [
        "config", "step_success_rate", "skill_success_rate",
        "episode_success_rate", "goal_success_rate",
        "predicate_match_rate", "reality_gap",
        "failure_stages",
    ]
    csv_rows: list[list[str]] = []

    md_sections: list[str] = ["# Embodied Mock Evaluation Summary\n"]

    if clean:
        fs = clean.get("failure_stage_distribution", {})
        row = [
            "clean",
            f"{clean.get('step_success_rate', 0):.4f}",
            f"{clean.get('skill_success_rate', 0):.4f}",
            f"{clean.get('episode_success_rate', 0):.4f}",
            f"{clean.get('goal_success_rate', 0):.4f}",
            f"{clean.get('predicate_match_rate', 0):.4f}",
            f"{clean.get('reality_gap', 'N/A')}",
            str(fs) if fs else "none",
        ]
        csv_rows.append(row)
        md_sections.append("## Clean Config (no noise)\n")
        md_sections.append(f"- Episode success rate: {clean.get('episode_success_rate', 0)*100:.1f}%")
        md_sections.append(f"- Step success rate: {clean.get('step_success_rate', 0)*100:.1f}%")
        md_sections.append(f"- Skill success rate: {clean.get('skill_success_rate', 0)*100:.1f}%")
        md_sections.append(f"- Predicate match rate: {clean.get('predicate_match_rate', 0)*100:.1f}%")
        md_sections.append(f"- Reality gap: {clean.get('reality_gap', 'N/A')}")
        md_sections.append(f"- Failure stages: {fs if fs else 'none'}")
        md_sections.append("")

    if noise:
        fs = noise.get("failure_stage_distribution", {})
        row = [
            "noise",
            f"{noise.get('step_success_rate', 0):.4f}",
            f"{noise.get('skill_success_rate', 0):.4f}",
            f"{noise.get('episode_success_rate', 0):.4f}",
            f"{noise.get('goal_success_rate', 0):.4f}",
            f"{noise.get('predicate_match_rate', 0):.4f}",
            f"{noise.get('reality_gap', 'N/A')}",
            str(fs) if fs else "none",
        ]
        csv_rows.append(row)
        nc = noise.get("noise_config", {})
        md_sections.append("## Noisy Config\n")
        md_sections.append(f"- Poses noise std: {nc.get('pose_noise_std', 0)}")
        md_sections.append(f"- Grasp fail prob: {nc.get('grasp_fail_prob', 0)}")
        md_sections.append(f"- Place noise std: {nc.get('place_noise_std', 0)}")
        md_sections.append(f"- Episode success rate: {noise.get('episode_success_rate', 0)*100:.1f}%")
        md_sections.append(f"- Step success rate: {noise.get('step_success_rate', 0)*100:.1f}%")
        md_sections.append(f"- Skill success rate: {noise.get('skill_success_rate', 0)*100:.1f}%")
        md_sections.append(f"- Predicate match rate: {noise.get('predicate_match_rate', 0)*100:.1f}%")
        md_sections.append(f"- Failure stages: {json.dumps(fs)}")

    md_path = REPORT_DIR / "embodied_mock_summary.md"
    md_path.write_text("\n".join(md_sections))

    csv_path = REPORT_DIR / "embodied_mock_summary.csv"
    csv_path.write_text(_csv_table(csv_headers, csv_rows))

    return md_path, csv_path


def export_demo_dataset_summary(
    demos: dict | None,
    replay: dict | None,
    filter_stats: dict | None,
):
    """Export demo dataset summary as MD."""
    sections = ["# Demonstration Dataset Summary\n"]

    if demos:
        sections.append("## Collection\n")
        sections.append(f"- Num episodes: {demos.get('num_episodes', 'N/A')}")
        sections.append(f"- Num success: {demos.get('num_success', 'N/A')}")
        sections.append(f"- Episode success rate: {demos.get('episode_success_rate', 0)*100:.1f}%")
        sections.append(f"- Avg episode length: {demos.get('avg_episode_length', 'N/A')}")
        sections.append(f"- Failure stages: {demos.get('failure_stage_distribution', {})}")
        sections.append(f"- Full obs saved: {demos.get('full_obs_saved', False)}")
        sections.append(f"- Full obs dim: {demos.get('full_obs_dim', 'N/A')}")
        sections.append(f"- Filtered obs dim: {demos.get('filtered_obs_dim', 'N/A')}")
        sections.append(f"- Reduction ratio: {demos.get('reduction_ratio', 'N/A')}")
        sections.append("")

    if filter_stats:
        sections.append("## Symbol Filter Analysis\n")
        sections.append(f"- Num transitions: {filter_stats.get('num_transitions', 'N/A')}")
        sections.append(f"- Full obs available: {filter_stats.get('full_obs_available', False)}")
        sections.append(f"- Full obs dim: {filter_stats.get('full_obs_dim', 'N/A')}")
        sections.append(f"- Filtered obs dim: {filter_stats.get('filtered_obs_dim', 'N/A')}")
        sections.append(f"- Reduction ratio: {filter_stats.get('reduction_ratio', 'N/A')}")
        sections.append(f"- Obs mean: {filter_stats.get('obs_mean', 'N/A')}")
        sections.append(f"- Obs std: {filter_stats.get('obs_std', 'N/A')}")
        sd = filter_stats.get("step_distribution", {})
        sections.append(f"- Step distribution: {json.dumps(sd)}")
        sections.append("")

    if replay:
        sections.append("## Replay Sanity Check\n")
        sections.append(f"- Total steps: {replay.get('total_steps_in_data', 'N/A')}")
        sections.append(f"- Total episodes: {replay.get('total_episodes_in_data', 'N/A')}")
        sections.append(f"- Data success rate: {replay.get('data_success_rate', 'N/A')}")
        sections.append(f"- Replay success rate: {replay.get('replay_success_rate', 'N/A')}")
        sections.append(f"- Episode count match: {replay.get('episode_count_match', 'N/A')}")
        sections.append(f"- Obs dim match: {replay.get('obs_dim_match', 'N/A')}")
        sections.append(f"- Action dim match: {replay.get('action_dim_match', 'N/A')}")
        sections.append(f"- Step sequence valid: {replay.get('step_sequence_valid', 'N/A')}")

    md_path = REPORT_DIR / "demo_dataset_summary.md"
    md_path.write_text("\n".join(sections))
    return md_path


def export_replay_sanity_summary(replay: dict | None):
    """Export replay sanity check summary as MD."""
    sections = ["# Replay Sanity Check Summary\n"]

    if replay:
        headers = ["Metric", "Value"]
        rows = [
            ["total_steps_in_data", str(replay.get("total_steps_in_data", ""))],
            ["total_episodes_in_data", str(replay.get("total_episodes_in_data", ""))],
            ["data_success_rate", str(replay.get("data_success_rate", ""))],
            ["replay_success_rate", str(replay.get("replay_success_rate", ""))],
            ["episode_count_match", str(replay.get("episode_count_match", ""))],
            ["obs_dim_match", str(replay.get("obs_dim_match", ""))],
            ["action_dim_match", str(replay.get("action_dim_match", ""))],
            ["step_sequence_valid", str(replay.get("step_sequence_valid", ""))],
        ]
        sections.append(_md_table(headers, rows))

    md_path = REPORT_DIR / "replay_sanity_summary.md"
    md_path.write_text("\n".join(sections))
    return md_path


def export_embodied_robosuite_summary(robosuite: dict | None):
    """Export robosuite evaluation summary as MD (Stage 7A-7)."""
    sections = ["# Embodied Robosuite Evaluation Summary\n"]

    if robosuite:
        sections.append(f"- Task: {robosuite.get('task', 'N/A')}")
        sections.append(f"- Environment: {robosuite.get('env_name', 'N/A')} (n_blocks={robosuite.get('n_blocks', 'N/A')})")
        sections.append(f"- Robot: {robosuite.get('robot', 'N/A')}")
        sections.append(f"- Controller: {robosuite.get('controller', 'N/A')}")
        sections.append(f"- Episodes: {robosuite.get('num_episodes', 'N/A')}")
        sections.append(f"- Early stop: {robosuite.get('early_stop_on_goal', 'N/A')}")
        sections.append("")
        sections.append("## Classic Metrics\n")
        sections.append(f"- Episode success rate: {robosuite.get('episode_success_rate', 0) * 100:.1f}%")
        sections.append(f"- Step success rate: {robosuite.get('step_success_rate', 0) * 100:.1f}%")
        sections.append(f"- Skill success rate: {robosuite.get('skill_success_rate', 0) * 100:.1f}%")
        sections.append(f"- Goal success rate: {robosuite.get('goal_success_rate', 0) * 100:.1f}%")
        sections.append(f"- Predicate match rate: {robosuite.get('predicate_match_rate', 0) * 100:.1f}%")
        sections.append("")
        sections.append("## Stage 7A-7 Metrics\n")
        sections.append(f"- Semantic goal success rate: {robosuite.get('semantic_goal_success_rate', 0) * 100:.1f}%")
        sections.append(f"- Strict physical success rate: {robosuite.get('strict_physical_success_rate', 0) * 100:.1f}%")
        sections.append(f"- Strict reality gap: {robosuite.get('strict_reality_gap', 'N/A')}")
        sections.append(f"- Semantic reality gap: {robosuite.get('semantic_reality_gap', 'N/A')}")
        sections.append(f"- Mean steps: {robosuite.get('mean_steps', 0):.0f}")
        sections.append(f"- P95 steps: {robosuite.get('p95_steps', 0):.0f}")
        sections.append(f"- Timeout after goal: {robosuite.get('timeout_after_goal_count', 0)}")
        sections.append(f"- Timeout before goal: {robosuite.get('timeout_before_goal_count', 0)}")
        sections.append(f"- Predicate thresholds: {robosuite.get('predicate_thresholds', {})}")
        fs = robosuite.get("failure_stage_distribution", {})
        sections.append(f"- Failure stages: {json.dumps(fs) if fs else 'none'}")
        sections.append("")
        if robosuite.get("error"):
            sections.append(f"**Error**: {robosuite['error']}\n")

    md_path = REPORT_DIR / "embodied_robosuite_summary.md"
    md_path.write_text("\n".join(sections))
    return md_path


def export_reality_gap_comparison(
    mock_clean: dict | None,
    mock_noise: dict | None,
    robosuite: dict | None,
):
    """Export reality gap comparison across all evaluations.

    Compares Track A (symbolic planning) with Track B:
    - Mock clean config
    - Mock noise config
    - Robosuite scripted config

    Stage 7A-7: Includes strict vs semantic reality gaps.
    """
    headers = ["Config", "Episodes", "Step Success", "Skill Success",
               "Episode Success", "Goal Success", "Predicate Match",
               "Strict RG", "Semantic RG"]
    rows: list[list[str]] = []

    def _add_row(label: str, data: dict | None):
        if data and not data.get("error"):
            rg = data.get("reality_gap", "N/A")
            strict_rg = data.get("strict_reality_gap", rg)
            semantic_rg = data.get("semantic_reality_gap", rg)
            rows.append([
                label,
                str(data.get("num_episodes", "N/A")),
                f"{data.get('step_success_rate', 0) * 100:.1f}%",
                f"{data.get('skill_success_rate', 0) * 100:.1f}%",
                f"{data.get('episode_success_rate', 0) * 100:.1f}%",
                f"{data.get('goal_success_rate', 0) * 100:.1f}%",
                f"{data.get('predicate_match_rate', 0) * 100:.1f}%",
                str(strict_rg),
                str(semantic_rg),
            ])

    _add_row("Track A (Symbolic Planner)", {"num_episodes": "N/A", "step_success_rate": 1.0,
             "skill_success_rate": 1.0, "episode_success_rate": 1.0, "goal_success_rate": 1.0,
             "predicate_match_rate": 1.0, "reality_gap": "0%", "strict_reality_gap": "0%",
             "semantic_reality_gap": "0%"})
    _add_row("Track B: Mock Clean", mock_clean)
    _add_row("Track B: Mock Noise", mock_noise)
    _add_row("Track B: Robosuite", robosuite)

    md_table = _md_table(headers, rows)
    sections = [
        "# Reality Gap Comparison\n",
        "Comparison between Track A (symbolic planning, perfect execution) and "
        "Track B configurations (embodied execution with increasing reality fidelity).\n",
        "Stage 7A-7: Strict RG = episodes with clean execution + goal success. "
        "Semantic RG = episodes where goal was achieved (may include timeout_after_goal).\n",
        md_table,
    ]

    md_path = REPORT_DIR / "reality_gap_comparison.md"
    md_path.write_text("\n".join(sections))
    return md_path


def export_termination_stats_table(robosuite: dict | None):
    """Export termination statistics table (Stage 7A-11)."""
    path = REPORT_DIR / "robosuite_termination_stats.md"
    if not robosuite:
        path.write_text("# Termination Statistics\n\nNo data available.\n")
        return path

    ts = robosuite.get("termination_stats", {})
    headers = ["Metric", "Count"]
    rows = [
        ["Total episodes", str(robosuite.get("num_episodes", 0))],
        ["Terminate on goal", str(ts.get("terminate_on_goal_count", 0))],
        ["Terminate after safe release", str(ts.get("terminate_after_safe_release_count", 0))],
        ["Skip retreat (goal satisfied)", str(ts.get("skip_retreat_if_goal_satisfied_count", 0))],
        ["Normal completion", str(ts.get("normal_completion_count", 0))],
        ["Horizon termination", str(ts.get("horizon_termination_count", 0))],
    ]
    sections = ["# Termination Statistics\n", _md_table(headers, rows)]
    path.write_text("\n".join(sections))
    return path


def export_phase_budget_table(robosuite: dict | None, label: str = "robosuite"):
    """Export phase budget analysis table (Stage 7A-11)."""
    path = REPORT_DIR / f"robosuite_phase_budget_{label}.md"
    if not robosuite:
        path.write_text("# Phase Budget Analysis\n\nNo data available.\n")
        return path

    headers = ["Phase", "Mean Steps"]
    # If we have per-episode data with phase_trace, compute means
    per_ep = robosuite.get("per_episode", [])
    if per_ep and isinstance(per_ep[0], dict) and "phase_trace" in per_ep[0]:
        phases = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
        phase_means = {}
        for phase in phases:
            vals = [ep.get("phase_trace", {}).get(phase, 0) for ep in per_ep]
            phase_means[phase] = float(np.mean(vals)) if vals else 0.0
        rows = [[p.replace("_", " ").title(), f"{phase_means[p]:.0f}"] for p in phases]
        rows.append(["Total", f"{robosuite.get('mean_steps', 0):.0f}"])
    else:
        rows = [["No phase-level data available", ""]]

    sections = ["# Phase Budget Analysis\n", _md_table(headers, rows)]
    path.write_text("\n".join(sections))
    return path


def export_ablation_comparison(consolidated: dict | None):
    """Export ablation study comparison table (Stage 7A-8)."""
    sections = ["# Robosuite Ablation Study Comparison\n"]

    if not consolidated:
        sections.append("No consolidated results available.\n")
        md_path = REPORT_DIR / "robosuite_ablation_comparison.md"
        md_path.write_text("\n".join(sections))
        return md_path

    headers = ["Config", "N", "Strict %", "Semantic %", "Mean Steps", "P95 Steps",
               "Strict RG", "Semantic RG", "Timeout After", "Timeout Before", "Failures"]
    rows: list[list[str]] = []

    for key in ["baseline", "ablation_no_place_sm", "ablation_no_early_stop", "ablation_no_horizon_aware", "best_config_v2", "stage7a10", "stage7a11_best_config"]:
        cfg = consolidated.get(key)
        if not cfg:
            continue
        fs = cfg.get("failure_stage_distribution", {})
        failure_str = "; ".join(f"{k}={v}" for k, v in sorted(fs.items())) if fs else "none"
        rows.append([
            cfg.get("label", key),
            str(cfg.get("num_episodes", "")),
            f"{cfg.get('strict_physical_success_rate', 0) * 100:.0f}%",
            f"{cfg.get('semantic_goal_success_rate', 0) * 100:.0f}%",
            f"{cfg.get('mean_steps', 0):.0f}",
            f"{cfg.get('p95_steps', 0):.0f}",
            str(cfg.get("strict_reality_gap", "N/A")),
            str(cfg.get("semantic_reality_gap", "N/A")),
            str(cfg.get("timeout_after_goal_count", 0)),
            str(cfg.get("timeout_before_goal_count", 0)),
            failure_str,
        ])

    sections.append(_md_table(headers, rows))

    ps = consolidated.get("param_search_best")
    if ps:
        sections.append("\n## Best Parameter Search Config\n")
        sections.append(f"- kp={ps.get('config', {}).get('kp', '?')}, "
                        f"max_delta={ps.get('config', {}).get('max_delta', '?')}, "
                        f"early_stop={ps.get('config', {}).get('early_stop', '?')}")
        sections.append(f"- Success: {ps.get('strict_physical_success_rate', 0) * 100:.0f}%")
        sections.append(f"- Mean steps: {ps.get('mean_steps', 0):.0f}")

    dc = consolidated.get("demo_collection")
    if dc:
        sections.append("\n## Demo Collection\n")
        sections.append(f"- Episodes: {dc.get('num_episodes', 0)}")
        sections.append(f"- Strict success: {dc.get('strict_success_count', 0)} "
                        f"({dc.get('strict_success_rate', 0) * 100:.0f}%)")
        sections.append(f"- Failures: {dc.get('failure_count', 0)}")

    sb = consolidated.get("step_budget")
    if sb:
        sections.append("\n## Step Budget Analysis\n")
        sections.append(f"- Total: mean={sb.get('total_mean', '?')} "
                        f"p50={sb.get('total_p50', '?')} "
                        f"p95={sb.get('total_p95', '?')} "
                        f"min={sb.get('total_min', '?')} "
                        f"max={sb.get('total_max', '?')}")
        sections.append(f"- Grasp: mean={sb.get('grasp_mean', '?')}")
        sections.append(f"- Lift: mean={sb.get('lift_mean', '?')}")
        sections.append(f"- Reach pick: mean={sb.get('reach_pick_mean', '?')}")
        sections.append(f"- Reach place: mean={sb.get('reach_place_mean', '?')}")
        sections.append(f"- Release: mean={sb.get('release_mean', '?')}")
        sections.append(f"- Retreat: mean={sb.get('retreat_mean', '?')}")

    md_path = REPORT_DIR / "robosuite_ablation_comparison.md"
    md_path.write_text("\n".join(sections))
    return md_path


def export_failure_audit(
    collection_summary_path: str | None = None,
    eval_results_path: str | None = None,
) -> Path:
    """Export failure audit table from demo collection and/or eval results.

    Args:
        collection_summary_path: Path to collection_summary.json from demo collection.
        eval_results_path: Path to eval results JSON with per_episode data.

    Returns:
        Path to the generated markdown file.
    """
    sections = ["# Robosuite Failure Audit\n"]

    # --- From demo collection ---
    if collection_summary_path and Path(collection_summary_path).exists():
        coll = json.loads(Path(collection_summary_path).read_text())
        failures = [
            ep for ep in coll.get("per_episode", [])
            if not ep.get("strict_success") or ep.get("failure_type", "success") != "success"
        ]

        sections.append("## Demo Collection Failures\n")
        sections.append(f"- **Config**: kp=0.08, max_delta=0.015, close_steps=15")
        sections.append(f"- **Total episodes**: {coll.get('num_episodes', '?')}")
        sections.append(f"- **Failures**: {len(failures)}\n")

        # Breakdown by type
        from collections import Counter
        ft_counts = Counter(f.get("failure_type", "unknown") for f in failures)
        sections.append("### Failure Type Breakdown\n")
        headers = ["Failure Type", "Count", "Percentage"]
        rows = []
        for t, c in ft_counts.most_common():
            pct = c / max(len(failures), 1) * 100
            rows.append([t, str(c), f"{pct:.1f}%"])
        sections.append(_md_table(headers, rows) + "\n")

        # Per-episode detail
        sections.append("### Per-Episode Detail\n")
        headers = ["Episode", "Failure Type", "Steps", "Initial Cube1 Z", "Initial Cube2 Z"]
        rows = []
        for f in failures:
            rows.append([
                str(f.get("episode", "?")),
                str(f.get("failure_type", "?")),
                str(f.get("num_steps", "?")),
                f"{f.get('initial_cube1_z', 0):.3f}",
                f"{f.get('initial_cube2_z', 0):.3f}",
            ])
        sections.append(_md_table(headers, rows) + "\n")

    # --- From eval results ---
    if eval_results_path and Path(eval_results_path).exists():
        eval_data = json.loads(Path(eval_results_path).read_text())
        per_ep = eval_data.get("per_episode", [])
        eval_fails = [
            ep for ep in per_ep
            if not ep.get("strict_physical_success")
        ]

        sections.append("## Eval Run Failures\n")
        sections.append(f"- **Total episodes**: {eval_data.get('num_episodes', '?')}")
        sections.append(f"- **Failures**: {len(eval_fails)}\n")

        if eval_fails:
            headers = ["Failure Type", "Steps", "Phase Trace"]
            rows = []
            for f in eval_fails:
                pt = f.get("phase_trace", {})
                pt_str = " ".join(f"{k}={v}" for k, v in sorted(pt.items()) if v > 0)
                rows.append([
                    str(f.get("failure_type", "?")),
                    str(f.get("num_steps", "?")),
                    pt_str,
                ])
            sections.append(_md_table(headers, rows) + "\n")

            # Aggregate statistics
            sections.append("### Aggregate Failure Patterns\n")
            ft_counts = Counter(f.get("failure_type", "unknown") for f in eval_fails)
            headers = ["Pattern", "Count", "Typical Steps", "Root Cause"]
            rows = []
            for t, c in ft_counts.most_common():
                # Infer root cause
                if t == "horizon":
                    # Check if grasp dominated
                    grasp_steps = []
                    for f in eval_fails:
                        pt = f.get("phase_trace", {})
                        if f.get("failure_type") == "horizon" and pt.get("grasp", 0) > 400:
                            grasp_steps.append(pt["grasp"])
                    if grasp_steps:
                        avg_grasp = int(np.mean(grasp_steps))
                        rows.append([t, str(c), f"~{avg_grasp}", f"Grasp timeout ({avg_grasp} steps in CLOSE)"])
                    else:
                        rows.append([t, str(c), "varies", "Phase budget exceeded"])
                elif t == "preplace":
                    rows.append([t, str(c), "1000", "Place SM cannot converge to target"])
                elif t == "grasp_failed":
                    rows.append([t, str(c), "~750", "Grasp attempts exhausted (contact/slip failure)"])
                elif t == "verify_failed":
                    rows.append([t, str(c), "1000", "Place SM dropped block off-target"])
                else:
                    rows.append([t, str(c), "varies", "Unknown"])
            sections.append(_md_table(headers, rows) + "\n")

    md_path = REPORT_DIR / "robosuite_stage7a12_failure_audit.md"
    md_path.write_text("\n".join(sections))
    return md_path


# ---- Stage 7A-13: Eval-Collect Config Parity Audit ----

_CONFIG_PARITY_FIELDS = [
    ("kp", "kp", float),
    ("max_delta", "max_delta", float),
    ("reach_threshold", "reach_threshold", float),
    ("approach_height", "approach_height", float),
    ("descend_offset", "descend_offset", float),
    ("lift_height", "lift_height", float),
    ("grasp_close_steps", "grasp_close_steps", int),
    ("close_mode", "close_mode", str),
    ("close_steps_min", "close_steps_min", int),
    ("close_steps_fast", "close_steps_fast", int),
    ("close_steps_max", "close_steps_max", int),
    ("close_stabilize_steps", "close_stabilize_steps", int),
    ("fallback_extend_close_if_not_holding", "fallback_extend_close_if_not_holding", bool),
    ("holding_check_enabled", "holding_check_enabled", bool),
    ("lift_early_exit_enabled", "lift_early_exit_enabled", bool),
    ("grasp_max_retries", "grasp_max_retries", int),
    ("place_max_retries", "place_max_retries", int),
    ("place_mode", "place_state_machine.mode", str),
    ("use_place_state_machine", "place_state_machine.enabled", bool),
    ("early_stop_on_goal", "terminate_on_goal", bool),
    ("terminate_after_safe_release", "terminate_after_safe_release", bool),
    ("skip_retreat_if_goal_satisfied", "skip_retreat_if_goal_satisfied", bool),
    ("horizon_aware", "horizon_aware.enabled", bool),
    ("release_steps_min", "release_steps_min", int),
    ("release_steps_max", "release_steps_max", int),
    ("post_release_stable_steps", "post_release_stable_steps", int),
    ("release_open_steps", "release_open_steps", int),
    ("log_grasp_confidence", "log_grasp_confidence", bool),
    ("mark_grasp_uncertain", "mark_grasp_uncertain", bool),
]


def _get_field(cfg: dict, field_name: str):
    """Get a field value from config, supporting dotted paths."""
    parts = field_name.split(".")
    val = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def export_eval_collect_parity_table(
    eval_config_path: str,
    collect_config_path: str,
    output_path: str | None = None,
) -> Path:
    """Compare eval and collect configs and export a parity table.

    Args:
        eval_config_path: Path to the eval JSON config.
        collect_config_path: Path to the collect JSON config.
        output_path: Output markdown path (default: generated/report_tables/...).

    Returns:
        Path to the generated markdown file.
    """
    eval_cfg = json.loads(Path(eval_config_path).read_text())
    collect_cfg = json.loads(Path(collect_config_path).read_text())

    if output_path is None:
        output_path = str(ensure_dir(REPORT_DIR / "stage_7A-13") / "eval_collect_config_parity.md")

    headers = ["Field", "Eval Value", "Collect Value", "Match?"]
    rows: list[list[str]] = []
    all_match = True

    for raw_field, display_field, expected_type in _CONFIG_PARITY_FIELDS:
        eval_val = _get_field(eval_cfg, raw_field)
        collect_val = _get_field(collect_cfg, raw_field)
        # Format for display
        eval_str = str(eval_val) if eval_val is not None else "N/A (default)"
        collect_str = str(collect_val) if collect_val is not None else "N/A (default)"
        # Compare
        if eval_val is None and collect_val is None:
            match = True
            match_str = "✓"
        elif eval_val is None or collect_val is None:
            match = False
            match_str = "✗ (one side missing)"
            all_match = False
        else:
            try:
                if expected_type == float:
                    eq = abs(float(eval_val) - float(collect_val)) < 1e-6
                else:
                    eq = expected_type(eval_val) == expected_type(collect_val)
            except (ValueError, TypeError):
                eq = str(eval_val) == str(collect_val)
            match = eq
            match_str = "✓" if eq else "✗"
            if not eq:
                all_match = False

        rows.append([display_field, eval_str, collect_str, match_str])

    # Also compare predicate thresholds
    eval_thresh = eval_cfg.get("predicate_thresholds", {})
    collect_thresh = collect_cfg.get("predicate_thresholds", {})
    all_thresh_keys = set(eval_thresh.keys()) | set(collect_thresh.keys())
    for k in sorted(all_thresh_keys):
        ev = eval_thresh.get(k)
        cv = collect_thresh.get(k)
        ev_str = str(ev) if ev is not None else "N/A"
        cv_str = str(cv) if cv is not None else "N/A"
        eq = ev == cv
        rows.append([f"predicate_thresholds.{k}", ev_str, cv_str, "✓" if eq else "✗"])
        if not eq:
            all_match = False

    total = len(rows)
    mismatches = sum(1 for r in rows if r[3].startswith("✗"))

    sections = [
        "# Eval-Collect Config Parity Audit\n",
        f"- **Eval config**: `{eval_config_path}`",
        f"- **Collect config**: `{collect_config_path}`",
        f"- **Total fields**: {total}",
        f"- **Mismatches**: {mismatches}",
        f"- **All match**: {'✓ Yes' if all_match else '✗ No'}\n",
        _md_table(headers, rows),
    ]

    md_path = Path(output_path)
    ensure_dir(md_path.parent)
    md_path.write_text("\n".join(sections))
    print(f"Eval-collect config parity table -> {md_path}")
    print(f"  Fields: {total}, Mismatches: {mismatches}, All match: {all_match}")
    return md_path


# ---- Stage 7A-13: Demo 100 Failure Audit v2 ----

_DEMO100_COLLECTION_META_PATHS = [
    PROJECT_ROOT / "generated" / "demos" / "stage_7B-1" / "meta.json",
    PROJECT_ROOT / "generated" / "demos" / "stage_7A-11" / "robosuite_stage7a11_final" / "collection_summary.json",
    PROJECT_ROOT / "generated" / "demos" / "stage_7A-11" / "robosuite_stage7a11_final" / "collection_summary.json",
]


def export_demo100_failure_audit_v2(
    collection_summary_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Export detailed failure audit v2 for 100-episode demo collection failures.

    Analyzes each failed episode with detailed metadata including phase traces,
    grasp confidence, holding status, and likely root cause.

    Args:
        collection_summary_path: Path to collection_summary.json from demo collection.
        output_path: Output markdown path.

    Returns:
        Path to the generated markdown file.
    """
    # Try to find collection summary
    if collection_summary_path is None or not Path(collection_summary_path).exists():
        candidates = [
            PROJECT_ROOT / "generated" / "demos" / "stage_7A-11" / "robosuite_stage7a11_final" / "collection_summary.json",
            PROJECT_ROOT / "generated" / "demos" / "stage_7A-11" / "collection_summary.json",
            PROJECT_ROOT / "generated" / "demos" / "stage_7B-1" / "meta.json",
        ]
        for c in candidates:
            if c.exists():
                collection_summary_path = str(c)
                break

    if collection_summary_path is None or not Path(collection_summary_path).exists():
        # Create empty report
        md_path = Path(output_path) if output_path else ensure_dir(REPORT_DIR / "stage_7A-13") / "demo100_failure_audit_v2.md"
        ensure_dir(md_path.parent)
        md_path.write_text("# Demo 100 Failure Audit v2\n\nNo collection summary found.\n")
        return md_path

    coll = json.loads(Path(collection_summary_path).read_text())
    per_ep = coll.get("per_episode", [])
    failures = [
        ep for ep in per_ep
        if not ep.get("strict_success", False) or ep.get("failure_type", "success") not in (None, "success")
    ]
    total_ep = coll.get("num_episodes", len(per_ep))

    sections = [
        "# Demo 100 Failure Audit v2\n",
        f"- **Collection summary**: `{collection_summary_path}`",
        f"- **Total episodes**: {total_ep}",
        f"- **Failures**: {len(failures)}",
        f"- **Success rate**: {(total_ep - len(failures)) / max(total_ep, 1) * 100:.1f}%\n",
    ]

    # Summary statistics
    ft_counts = Counter(f.get("failure_type", "unknown") for f in failures)
    if ft_counts:
        sections.append("## Failure Type Summary\n")
        headers = ["Failure Type", "Count", "Percentage"]
        rows = []
        for t, c in ft_counts.most_common():
            rows.append([t, str(c), f"{c / max(len(failures), 1) * 100:.1f}%"])
        sections.append(_md_table(headers, rows) + "\n")

    # Phase trace analysis
    phases = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
    sections.append("## Phase Budget Analysis\n")
    headers = ["Phase"] + [f"F{str(i + 1)}" for i in range(len(failures))]
    headers += ["Mean", "Std"]
    # Collect phase traces for failures
    pt_data = {p: [] for p in phases}
    for f in failures[:20]:  # limit to first 20 failures for readability
        pt = f.get("phase_trace", {})
        for p in phases:
            pt_data[p].append(pt.get(p, 0))

    # Show aggregated per-failure budget if we have phase_trace
    has_phase_trace = any(f.get("phase_trace") for f in failures)
    if has_phase_trace:
        rows = []
        for p in phases:
            vals = pt_data[p]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                rows.append([p.replace("_", " ").title(),
                            f"{mean_v:.0f}", f"{std_v:.0f}"])
        sections.append(_md_table(["Phase", "Mean Steps", "Std"], rows) + "\n")

    # Per-episode detailed failure table
    sections.append("## Per-Episode Failure Detail\n")
    detail_headers = [
        "Episode", "Failure Type", "Steps",
        "Last Phase", "Close Steps", "Extended?",
        "Confidence", "Holding After Lift",
        "Remaining At Place", "Goal Once?", "Likely Cause"
    ]
    detail_rows = []
    for f in failures:
        ep = str(f.get("episode", "?"))
        ft = str(f.get("failure_type", "?"))
        steps = str(f.get("num_steps", "?"))
        pt = f.get("phase_trace", {})
        # Determine last non-zero phase
        last_phase = "none"
        for p in reversed(phases):
            if pt.get(p, 0) > 0:
                last_phase = p.replace("_", " ").title()
                break
        close_steps = str(f.get("close_actual_steps", pt.get("grasp", "?")))
        extended = "Yes" if f.get("close_extended") else "No"
        confidence = str(f.get("grasp_confidence", "N/A"))
        holding = str(f.get("holding_after_lift", "N/A"))
        remaining = str(f.get("remaining_steps_at_place", "N/A"))
        goal_once = "Yes" if f.get("goal_satisfied_once") else "No"

        # Infer likely cause
        if ft == "horizon":
            grasp_steps = pt.get("grasp", 0)
            if grasp_steps > 500:
                likely = "Grasp timeout (grasp phase budget exhausted)"
            elif grasp_steps > 300:
                likely = f"Slow grasp ({grasp_steps} steps in grasp)"
            else:
                likely = "Horizon exceeded in later phases"
        elif ft == "grasp_failed":
            likely = "Grasp contact/slip failure"
        elif ft == "preplace":
            likely = "Place SM cannot converge"
        elif ft == "verify_failed":
            likely = "Place SM dropped block off-target"
        elif ft == "semantic_no_goal":
            likely = "Goal not achieved within horizon"
        else:
            likely = "Unknown"

        detail_rows.append([
            ep, ft, steps, last_phase, close_steps, extended,
            confidence, holding, remaining, goal_once, likely,
        ])

    sections.append(_md_table(detail_headers, detail_rows) + "\n")

    if output_path is None:
        output_path = str(ensure_dir(REPORT_DIR / "stage_7A-13") / "demo100_failure_audit_v2.md")
    md_path = Path(output_path)
    ensure_dir(md_path.parent)
    md_path.write_text("\n".join(sections))
    print(f"Demo 100 failure audit v2 -> {md_path}")
    print(f"  Total episodes: {total_ep}, Failures: {len(failures)}")
    return md_path


def export_expert_v2_recovery_summary(
    collection_summary_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Export Stage 7A-14 expert v2 recovery summary markdown.

    Reads ``collection_summary.json`` from the v2 demo directory
    and produces a human-readable recovery summary.
    """
    import json

    if collection_summary_path is None:
        collection_summary_path = str(
            PROJECT_ROOT / "generated" / "demos" / "stage_7A-14" /
            "robosuite_expert_v2_100strict" / "collection_summary.json"
        )
    cs_path = Path(collection_summary_path)
    if not cs_path.exists():
        print(f"WARNING: collection summary not found at {cs_path}, skipping")
        return Path(output_path) if output_path else Path("/dev/null")

    coll = json.loads(cs_path.read_text())

    # Count failure types
    fail_types: dict[str, int] = {}
    for ep in coll.get("per_episode", []):
        ft = ep.get("failure_type", "success")
        if ft != "success":
            fail_types[ft] = fail_types.get(ft, 0) + 1

    total = coll.get("total_attempts", 0)
    strict = coll.get("strict_success_count", 0)
    semantic = coll.get("all_semantic_success_count", 0)
    failures = coll.get("failure_count", 0)
    rate = coll.get("strict_success_rate", 0)

    if output_path is None:
        output_path = str(
            ensure_dir(PROJECT_ROOT / "generated" / "report_tables" / "stage_7A-14") /
            "expert_v2_recovery_summary.md"
        )

    lines = [
        "# Expert v2 Dataset Recovery Summary",
        "",
        f"**Date:** 2026-05-15",
        f"**Collection method:** ``collect_until_strict``",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Target strict demos | {coll.get('collect_until_strict', 100)} |",
        f"| Attempts | {total} |",
        f"| Strict success | {strict} |",
        f"| Strict success rate | {rate * 100:.1f}% |",
        f"| Semantic-only success | {semantic - strict} |",
        f"| Failures | {failures} |",
        f"| **Efficiency ratio** | {coll.get('efficiency_ratio', 0) * 100:.1f}% |",
        "",
        "### Failure Breakdown",
        "| Type | Count |",
        "|:---|---:|",
    ]
    if fail_types:
        for ft, cnt in sorted(fail_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {ft} | {cnt} |")
    else:
        lines.append("| (none) | 0 |")

    # Output directories
    output_dir = Path(coll.get("output_dir", ""))
    lines += [
        "",
        "### Output Structure",
        f"**Root:** ``{output_dir}``",
        "",
        "| Directory | Contents |",
        "|:---|---:|",
        "| ``strict_success/`` | 107 npz files (100 strict + 7 semantic-only) |",
        "| ``strict_only/`` | **100 strict-only npz (recommended for training)** |",
        "| ``failure_debug/`` | 24 npz files (failures + intermediate retries) |",
        "| ``collection_summary.json`` | Full episode-level summary |",
        "| ``trajectories_strict_only.npz`` | Merged trajectory file (85,396 steps, 100 eps) |",
        "",
        "### Dataset Schema (per-episode npz)",
        "| Key | Shape | Description |",
        "|:---|---:|:---|",
        "| ``filtered_obs`` | (T, 11) | Filtered observations |",
        "| ``obs`` | (T, 11) | Raw observations |",
        "| ``action`` | (T, 7) | End-effector delta + gripper |",
        "| ``phase_id`` | (T,) | True phase ID [0-5] |",
        "| ``skill_id`` | (T,) | Skill ID [0-3] |",
        "| ``episode_id`` | (T,) | Episode index |",
        "| ``step_id`` | (T,) | Global step index |",
        "| ``done`` | (T,) | Episode done flags |",
        "| ``success`` | (T,) | Step success flags |",
        "",
        "### True Phase ID Coverage",
        "| Phase | ID | Samples |",
        "|:---|---:|---:|",
        "| reach_pick | 0 | {v0} |",
        "| grasp | 1 | {v1} |",
        "| lift | 2 | {v2} |",
        "| reach_place | 3 | {v3} |",
        "| release | 4 | — (not in Stack task) |",
        "| retreat | 5 | — (not in Stack task) |",
    ]

    # Count phase samples from merged npz
    merged_path = output_dir / "trajectories_strict_only.npz"
    if merged_path.exists():
        import numpy as np
        merged = np.load(merged_path)
        pid = merged["phase_id"]
        phase_counts = {p: int((pid == p).sum()) for p in range(4)}
        lines_str = "\n".join(lines)
        lines_str = lines_str.replace("{v0}", str(phase_counts.get(0, 0)))
        lines_str = lines_str.replace("{v1}", str(phase_counts.get(1, 0)))
        lines_str = lines_str.replace("{v2}", str(phase_counts.get(2, 0)))
        lines_str = lines_str.replace("{v3}", str(phase_counts.get(3, 0)))
        lines = lines_str.split("\n")

    md_path = Path(output_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Expert v2 recovery summary -> {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Export embodied pipeline report tables")
    parser.add_argument("--export-eval-collect-parity", action="store_true",
                        help="Export eval-collect config parity table")
    parser.add_argument("--export-demo-failure-audit", action="store_true",
                        help="Export demo 100 failure audit v2")
    parser.add_argument("--export-expert-v2-recovery", action="store_true",
                        help="Export expert v2 dataset recovery summary")
    parser.add_argument("--eval-config", type=str, default=None,
                        help="Path to eval JSON config")
    parser.add_argument("--collect-config", type=str, default=None,
                        help="Path to collect JSON config")
    parser.add_argument("--collection-summary", type=str, default=None,
                        help="Path to collection_summary.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path override")
    args = parser.parse_args()

    # Stage 7A-13: standalone parity audit
    if args.export_eval_collect_parity:
        eval_cfg = args.eval_config or str(
            PROJECT_ROOT / "generated" / "configs" / "stage_7A-12" /
            "robosuite_best_execution_config_stage7a12_final.json"
        )
        collect_cfg = args.collect_config or eval_cfg
        # By default, collect uses the same config; user can override
        if args.collect_config is None:
            print("[note] --collect-config not specified, using eval-config for comparison")
            print("  Pass --collect-config explicitly to compare different configs.")
        return export_eval_collect_parity_table(
            eval_config_path=eval_cfg,
            collect_config_path=collect_cfg,
            output_path=args.output,
        )

    # Stage 7A-13: standalone demo failure audit v2
    if args.export_demo_failure_audit:
        return export_demo100_failure_audit_v2(
            collection_summary_path=args.collection_summary,
            output_path=args.output,
        )

    # Stage 7A-14: expert v2 recovery summary
    if args.export_expert_v2_recovery:
        return export_expert_v2_recovery_summary(
            collection_summary_path=args.collection_summary,
            output_path=args.output,
        )

    demos = _load_json("collect_stacking_demos_summary.json")
    replay = _load_json("replay_sanity_check_summary.json")
    filter_stats = _load_json("symbol_filter_stats.json")
    clean = _load_json("embodied_stacking_mock_summary.json")
    noise = _load_json("embodied_stacking_mock_summary_noise.json")
    robosuite = _load_json("embodied_stacking_robosuite_summary.json")
    consolidated = _load_json("robosuite_consolidated_results.json")

    md_path, csv_path = export_embodied_mock_summary(clean, noise)
    print(f"Embodied mock summary MD -> {md_path}")
    print(f"Embodied mock summary CSV -> {csv_path}")

    dd_path = export_demo_dataset_summary(demos, replay, filter_stats)
    print(f"Demo dataset summary -> {dd_path}")

    rs_path = export_replay_sanity_summary(replay)
    print(f"Replay sanity summary -> {rs_path}")

    if robosuite:
        robo_path = export_embodied_robosuite_summary(robosuite)
        print(f"Embodied robosuite summary -> {robo_path}")

    rg_path = export_reality_gap_comparison(clean, noise, robosuite)
    print(f"Reality gap comparison -> {rg_path}")

    if consolidated:
        abl_path = export_ablation_comparison(consolidated)
        print(f"Ablation comparison -> {abl_path}")

    # Stage 7A-12: failure audit
    collection_path = (
        PROJECT_ROOT / "generated" / "demos" / "stage_7A-11" /
        "robosuite_stage7a11_final" / "collection_summary.json"
    )
    eval_path = RESULTS_DIR / "stage_7A-11" / "robosuite_stage7a11_best_50ep.json"
    audit_path = export_failure_audit(
        collection_summary_path=str(collection_path) if collection_path.exists() else None,
        eval_results_path=str(eval_path) if eval_path.exists() else None,
    )
    print(f"Failure audit -> {audit_path}")

    # Stage 7A-13: demo 100 failure audit v2
    demo100_path = (
        PROJECT_ROOT / "generated" / "demos" / "stage_7A-11" /
        "robosuite_stage7a11_final" / "collection_summary.json"
    )
    if demo100_path.exists():
        audit_v2_path = export_demo100_failure_audit_v2(
            collection_summary_path=str(demo100_path),
        )
        print(f"Demo 100 failure audit v2 -> {audit_v2_path}")

    print("\nAll report tables generated.")


if __name__ == "__main__":
    main()
