"""Stage 9 canonical simulation report export.

Generates:
  - dagger_100ep_failure_audit_v3.md  (Stage 8 100ep DAgger deep-dive)
  - canonical_sim_comparison.md       (scripted vs DAgger comparison)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "generated/report_tables/stage_9"
RESULTS_DIR = PROJECT_ROOT / "generated/results/stage_9"
STAGE8_DIR = PROJECT_ROOT / "generated/results/stage_8"


def load_result(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Failure Audit v3 ──────────────────────────────────────────────────────


def _audit_failure_detail(data: dict) -> list[dict]:
    """Annotate each failed episode with likely cause."""
    out = []
    for i, ep in enumerate(data["episode_results"]):
        if ep["success"]:
            continue
        ft = ep.get("failure_type", "?")
        steps = ep["steps"]
        bc = ep["bc_steps"]

        # Heuristic: early-horizon + very low bc_steps → policy collapse
        if ft == "horizon" and bc < 200:
            cause = "Policy collapse in reach_pick (early-HZ, low BC steps)"
        elif ft == "horizon":
            cause = "Place SM timeout or reach_place oscillation"
        elif ft == "verify_failed":
            cause = "Place SM dropped block off-target"
        elif ft == "preplace":
            cause = "Preplace verification never satisfied during place SM"
        else:
            cause = "Unknown"

        out.append({
            "episode": i,
            "failure_type": ft,
            "steps": steps,
            "bc_steps": bc,
            "scripted_steps": steps - bc,
            "likely_cause": cause,
        })
    return out


def _failure_audit_v3_table(data: dict) -> str:
    """Generate failure audit v3 markdown table."""
    details = _audit_failure_detail(data)
    if not details:
        return "No failures found.\n"

    lines = [
        "| Ep | Failure Type | Steps | BC Steps | Scripted | Likely Cause |",
        "|:--:|:-----------:|:-----:|:--------:|:--------:|:-------------|",
    ]
    for d in details:
        lines.append(
            f"| {d['episode']} | {d['failure_type']} | {d['steps']} | {d['bc_steps']} | "
            f"{d['scripted_steps']} | {d['likely_cause']} |"
        )
    return "\n".join(lines) + "\n"


def _count_truly_clean(data: dict) -> int:
    """Count episodes with success=True and failure_type=None."""
    return sum(
        1 for ep in data["episode_results"]
        if ep["success"] and ep.get("failure_type") is None
    )


def export_failure_audit_v3():
    """Generate dagger_100ep_failure_audit_v3.md."""
    path = STAGE8_DIR / "dagger_reach_final_100ep.json"
    data = load_result(path)
    if data is None:
        msg = f"# Failure Audit v3\n\nStage 8 result not found at {path}\n"
        (REPORT_DIR / "dagger_100ep_failure_audit_v3.md").write_text(msg)
        return

    n_total = data["num_episodes"]
    n_success = data["success_count"]
    n_fail = n_total - n_success
    sr = data["success_rate"]
    ftypes = data["failure_type_counts"]
    clean = _count_truly_clean(data)

    mean_steps = data["mean_steps"]
    mean_bc = data["mean_bc_steps"]
    mean_scripted = data["mean_scripted_steps"]
    p95 = data["p95_steps"]
    elapsed = data["elapsed_seconds"]

    # Aggregate timing by failure type
    ft_timing = {}
    for ep in data["episode_results"]:
        ft = ep.get("failure_type", "none")
        if ft not in ft_timing:
            ft_timing[ft] = {"count": 0, "steps": [], "bc_steps": []}
        ft_timing[ft]["count"] += 1
        ft_timing[ft]["steps"].append(ep["steps"])
        ft_timing[ft]["bc_steps"].append(ep.get("bc_steps", 0))

    lines = [
        "# DAgger 100ep Failure Audit v3",
        "",
        f"**Stage 8 result**: `{path.name}`",
        f"- **Total episodes**: {n_total}",
        f"- **Success**: {n_success} ({sr*100:.1f}%)",
        f"- **Failures**: {n_fail} ({100-n_success:.1f}%)",
        f"- **Truly clean** (success + no failure_type): {clean}",
        f"- **Elapsed**: {elapsed:.1f}s",
        "",
        "---",
        "## Failure Type Summary",
        "",
        f"| Type | Count | % of Total | % of Failures | Mean Steps | Mean BC Steps |",
        "|:---:|:----:|:--------:|:------------:|:--------:|:------------:|",
    ]
    for ft in ["horizon", "preplace", "verify_failed"]:
        cnt = ftypes.get(ft, 0)
        t = ft_timing.get(ft, {})
        mean_s = f"{np.mean(t['steps']):.0f}" if t.get("steps") else "N/A"
        mean_b = f"{np.mean(t['bc_steps']):.0f}" if t.get("bc_steps") else "N/A"
        lines.append(
            f"| {ft} | {cnt} | {cnt/n_total*100:.1f}% | "
            f"{cnt/n_fail*100:.1f}% | {mean_s} | {mean_b} |"
        )

    lines += [
        "",
        "---",
        "## Per-Episode Failure Detail",
        "",
        _failure_audit_v3_table(data),
        "",
        "---",
        "## Root Cause Analysis",
        "",
        "### Hypothesis 1: Policy collapse in reach_pick (early-horizon, bc_steps < 200)",
        "",
        "Failures with `horizon` type and very low BC steps indicate the policy "
        "outputs near-zero or invalid actions early, causing the episode to stall "
        "in reach_pick phase. If the block is never approached, the state machine "
        "cannot advance.",
        "",
        "### Hypothesis 2: reach_place divergence (horizon, bc_steps > 400)",
        "",
        "Failures where bc_steps are high (>400) but still hit horizon suggest "
        "the BC policy oscillates or diverges near the target during the place "
        "phase, preventing the place state machine from converging.",
        "",
        "### Hypothesis 3: Preplace verification failure (preplace type)",
        "",
        "Even in successful episodes, `failure_type: preplace` is common — it "
        "means the preplace heuristic triggered at least once before eventual "
        "success. In failed episodes, the place SM simply could not converge.",
        "",
        "### Hypothesis 4: verify_failed (block dropped off-target)",
        "",
        "The block was released but the final predicate check fails. This can "
        "happen when release happens while the gripper is still over the target "
        "but the block bounces off, or when release position is imprecise.",
        "",
        "---",
        "## Comparison: Stage 7B-5 (93.3%) vs Stage 8 (80.0%)",
        "",
        "| Metric | Stage 7B-5 (3×20ep) | Stage 8 (100ep) | Delta |",
        "|:------|:------------------:|:--------------:|:----:|",
        f"| Episodes | 60 | 100 | — |",
        f"| Success rate | 93.3% | {sr*100:.1f}% | -{93.3-sr*100:.1f}% |",
        f"| Mean steps | ~720 | {mean_steps:.1f} | — |",
        f"| Mean BC steps | ~520 | {mean_bc:.1f} | — |",
        "",
        "The 13.3pp drop suggests either:",
        "1. **Seed variance**: seed=0 underperforms the multi-seed average",
        "2. **Small-sample bias**: 20ep runs happen to land on easier env initializations",
        "3. **Policy degradation**: Large-N rollouts accumulate distributional shift",
        "",
        "To distinguish: DAgger canonical 3×50ep (S9-C) at different seeds will "
        "show whether seed=0 is unlucky or the policy degrades at scale.",
    ]

    (REPORT_DIR / "dagger_100ep_failure_audit_v3.md").write_text("\n".join(lines))
    print(f"  → {REPORT_DIR / 'dagger_100ep_failure_audit_v3.md'}")


# ── Canonical Sim Comparison ──────────────────────────────────────────────


def _summarize_result(label: str, data: dict) -> dict:
    ft = data.get("failure_type_counts", {})
    return {
        "label": label,
        "episodes": data["num_episodes"],
        "success_rate": data["success_rate"] * 100,
        "mean_steps": data["mean_steps"],
        "mean_bc": data.get("mean_bc_steps", 0),
        "mean_scripted": data.get("mean_scripted_steps", 0),
        "p95_steps": data["p95_steps"],
        "horizon": ft.get("horizon", 0),
        "preplace": ft.get("preplace", 0),
        "verify_failed": ft.get("verify_failed", 0),
    }


def export_canonical_comparison():
    """Generate canonical_sim_comparison.md."""
    rows = []

    # Stage 8 100ep DAgger (baseline)
    d8 = load_result(STAGE8_DIR / "dagger_reach_final_100ep.json")
    if d8:
        rows.append(_summarize_result("Stage 8 DAgger 100ep (seed=0)", d8))

    # Stage 9 results
    for fname in [
        "scripted_canonical_100ep.json",
        "dagger_reach_seed0_50ep.json",
        "dagger_reach_seed1_50ep.json",
        "dagger_reach_seed2_50ep.json",
    ]:
        d = load_result(RESULTS_DIR / fname)
        if d:
            label = fname.replace(".json", "").replace("_", " ").title()
            rows.append(_summarize_result(label, d))

    if not rows:
        (REPORT_DIR / "canonical_sim_comparison.md").write_text(
            "# Canonical Simulation Comparison\n\nNo results available yet.\n"
        )
        return

    lines = [
        "# Canonical Simulation Comparison",
        "",
        "| Config | N | Success | Mean Steps | Mean BC | Mean Scripted | P95 Steps | HZ | PP | VF |",
        "|:------|:-:|:------:|:---------:|:-------:|:------------:|:--------:|:--:|:--:|:--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['episodes']} | {r['success_rate']:.1f}% | "
            f"{r['mean_steps']:.0f} | {r['mean_bc']:.0f} | {r['mean_scripted']:.0f} | "
            f"{r['p95_steps']:.0f} | {r['horizon']} | {r['preplace']} | {r['verify_failed']} |"
        )

    lines += [
        "",
        "### Legend",
        "- HZ = horizon (timeout)",
        "- PP = preplace (place verification failed)",
        "- VF = verify_failed (goal predicate mismatch)",
        "",
        "### Key Questions",
        "1. **Scripted baseline**: What is the ceiling for scripted-only execution?",
        "2. **DAgger canonical 3x50ep**: Does multi-seed average match Stage 7B-5 (93.3%) "
        "or Stage 8 (80.0%)?",
        "3. **reach_pick-only diagnostic**: Does limiting BC to reach_pick improve "
        "reach_place (preplace) outcomes?",
    ]

    (REPORT_DIR / "canonical_sim_comparison.md").write_text("\n".join(lines))
    print(f"  → {REPORT_DIR / 'canonical_sim_comparison.md'}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting Stage 9 reports...")
    export_failure_audit_v3()
    export_canonical_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
