"""Stage 10 deployment policy comparison export.

Generates:
  - deployment_policy_comparison.md
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "generated/report_tables/stage_10"
RESULTS_S9 = PROJECT_ROOT / "generated/results/stage_9"
RESULTS_S10 = PROJECT_ROOT / "generated/results/stage_10"


def load_result(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _summarize(label: str, data: dict, prefix: str = "") -> dict:
    ft = data.get("failure_type_counts", {})
    return {
        "label": label,
        "n": data["num_episodes"],
        "sr": data["success_rate"] * 100,
        "steps": data["mean_steps"],
        "bc": data.get("mean_bc_steps", 0),
        "p95": data["p95_steps"],
        "preplace": ft.get("preplace", 0),
        "horizon": ft.get("horizon", 0),
        "verify": ft.get("verify_failed", 0),
        "grasp_failed": ft.get("grasp_failed", 0),
    }


_ROWS: list[dict] = []


def _add(label: str, fname: str, *,
         results_dir: Path = RESULTS_S9,
         prefix: str = "") -> None:
    d = load_result(results_dir / fname)
    if d:
        _ROWS.append(_summarize(label, d, prefix=prefix))


def export_deployment_comparison():
    _ROWS.clear()

    # Scripted baseline
    _add("Scripted-only", "scripted_canonical_100ep.json")

    # DAgger reach_pick+reach_place
    _add("DAgger reach+place (seed0)", "dagger_reach_seed0_50ep.json")
    _add("DAgger reach+place (seed1)", "dagger_reach_seed1_50ep.json")
    _add("DAgger reach+place (seed2)", "dagger_reach_seed2_50ep.json")

    # DAgger reach_pick-only 100ep
    _add("DAgger reach-only (100ep)", "dagger_reach_pick_only_100ep.json")

    # DAgger reach_pick-only 3x50ep
    for seed in range(3):
        _add(f"DAgger reach-only seed{seed}",
             f"reach_pick_only_seed{seed}_50ep.json",
             results_dir=RESULTS_S10)

    if not _ROWS:
        (REPORT_DIR / "deployment_policy_comparison.md").write_text(
            "# Deployment Policy Comparison\n\nNo results available yet.\n"
        )
        return

    # Compute averages
    dagger_rp_rows = [r for r in _ROWS if "reach+place (seed" in r["label"]]
    dagger_ro_50ep = [r for r in _ROWS if "reach-only seed" in r["label"]]

    lines = [
        "# Deployment Policy Comparison",
        "",
        "| Policy | N | Success | Steps | BC | P95 | PP | HZ | VF | GF |",
        "|:------|:-:|:------:|:----:|:--:|:---:|:--:|:--:|:--:|:--:|",
    ]
    for r in _ROWS:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['sr']:.1f}% | {r['steps']:.0f} | "
            f"{r['bc']:.0f} | {r['p95']:.0f} | {r['preplace']} | {r['horizon']} | "
            f"{r['verify']} | {r['grasp_failed']} |"
        )

    # Averages
    if dagger_rp_rows:
        avg_sr = sum(r["sr"] for r in dagger_rp_rows) / len(dagger_rp_rows)
        avg_n = sum(r["n"] for r in dagger_rp_rows)
        avg_pp = sum(r["preplace"] for r in dagger_rp_rows) / len(dagger_rp_rows)
        lines.append(
            f"| _DAgger reach+place avg_ | {avg_n} | {avg_sr:.1f}% | — | — | — | "
            f"{avg_pp:.1f} | — | — | — |"
        )
    if dagger_ro_50ep:
        avg_sr2 = sum(r["sr"] for r in dagger_ro_50ep) / len(dagger_ro_50ep)
        avg_n2 = sum(r["n"] for r in dagger_ro_50ep)
        lines.append(
            f"| _DAgger reach-only avg_ | {avg_n2} | {avg_sr2:.1f}% | — | — | — | "
            f"0 | — | — | — |"
        )

    lines += [
        "",
        "### Legend",
        "- PP = preplace (place verification failed)",
        "- HZ = horizon (timeout)",
        "- VF = verify_failed",
        "- GF = grasp_failed",
        "",
        "### Recommendation",
        "",
        "| Policy | Verdict | Rationale |",
        "|:------|:-------:|:----------|",
    ]

    # Determine recommendation from available data
    ro_100 = next((r for r in _ROWS if "reach-only (100ep)" in r["label"]), None)
    if ro_100 and ro_100["sr"] >= 90:
        lines.append(
            "| **DAgger reach_pick-only** | ✅ **Canonical default** | "
            f"{ro_100['sr']:.0f}% on 100ep, 0 preplace failures |"
        )
    lines += [
        "| DAgger reach_pick+reach_place | ⚠️ Research variant | "
        "Higher BC step count but preplace failures degrade reliability |",
        "| Scripted-only | 📊 Baseline | No learned control, used as ceiling reference |",
    ]

    (REPORT_DIR / "deployment_policy_comparison.md").write_text("\n".join(lines))
    print(f"  → {REPORT_DIR / 'deployment_policy_comparison.md'}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting Stage 10 deployment report...")
    export_deployment_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
