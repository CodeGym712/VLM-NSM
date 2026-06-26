"""Summarize gate attribution from --save_trace JSONL files.

Produces per-phase table: attempted/accepted/rejected/VLA-steps/scripted-steps.
Reads all trace dirs under generated/results/vla/ and control/.

Usage:
    nsm/bin/python -m vlm_nsm.experiments.summarize_gate_attribution
"""
import json, sys
from pathlib import Path
from collections import defaultdict

TRACE_ROOTS = [
    "generated/results/vla",
    "generated/results/control",
]

def summarize_traces():
    all_results = []
    for root in TRACE_ROOTS:
        for traces_dir in Path(root).glob("*/traces"):
            name = traces_dir.parent.name
            files = sorted(traces_dir.glob("ep_*.jsonl"))
            if not files:
                continue

            per_phase = defaultdict(lambda: {"vla": 0, "scripted": 0, "vla_steps": 0, "scripted_steps": 0})
            total_vla, total_scripted = 0, 0
            eps = 0

            for f in files:
                eps += 1
                for line in f.read_text().strip().split("\n"):
                    if not line:
                        continue
                    step = json.loads(line)
                    phase = step.get("phase", "unknown")
                    ctrl = step.get("controller", "scripted")
                    per_phase[phase][ctrl] += 1
                    if ctrl == "vla":
                        total_vla += 1
                    else:
                        total_scripted += 1

            total = total_vla + total_scripted
            vla_pct = total_vla / max(total, 1) * 100
            all_results.append({
                "name": name, "eps": eps, "total_steps": total,
                "vla_steps": total_vla, "scripted_steps": total_scripted,
                "vla_pct": vla_pct,
                "per_phase": {p: {"vla": d["vla"], "scripted": d["scripted"],
                                  "vla_pct": d["vla"]/max(d["vla"]+d["scripted"],1)*100}
                              for p, d in sorted(per_phase.items())},
            })

    # Print table
    print(f"{'Condition':<30} {'Ep':>4} {'Total':>7} {'VLA':>7} {'Script':>7} {'VLA%':>6}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: x["vla_pct"], reverse=True):
        print(f"{r['name']:<30} {r['eps']:>4} {r['total_steps']:>7} {r['vla_steps']:>7} "
              f"{r['scripted_steps']:>7} {r['vla_pct']:>5.1f}%")

    # Per-phase detail for key conditions
    print("\n=== Per-Phase Detail (top conditions) ===")
    for r in all_results[:5]:
        print(f"\n{r['name']} ({r['eps']}ep, {r['vla_pct']:.1f}% VLA):")
        for phase, d in r["per_phase"].items():
            print(f"  {phase:<20}: VLA={d['vla']:>5}  Script={d['scripted']:>5}  VLA%={d['vla_pct']:.1f}")

    return all_results


if __name__ == "__main__":
    summarize_traces()
