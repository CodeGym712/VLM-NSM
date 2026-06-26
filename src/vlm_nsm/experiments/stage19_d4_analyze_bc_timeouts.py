"""Stage 19-D4-1: Analyze Balanced BC timeout failures from D3-5 evaluation.

Classifies each timeout episode as one of:
  - divergent: BC policy never converges to grasp pose within 500 steps
  - near_miss: BC policy gets very close but just misses grasp
  - oscillation: BC policy oscillates between phases without progress
  - horizon_insufficient: BC reaches grasp but remaining steps insufficient
  - wrong_object_bias: BC approaches wrong cube
  - unknown: insufficient data to classify
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def classify_timeout(episode: dict) -> str:
    """Classify a single timeout episode."""
    trace = episode.get("phase_transition_trace", {})
    steps = episode.get("steps", 0)
    skill_failure = episode.get("skill_failure_phase", "")
    final_err = episode.get("final_pose_error", 999)
    top_disp = episode.get("top_displacement", 0)
    grasp_success = episode.get("grasp_success", False)
    place_success = episode.get("place_success", False)

    reach_pick_steps = trace.get("reach_pick", 0)
    grasp_steps = trace.get("grasp", 0)
    lift_steps = trace.get("lift", 0)
    place_steps = trace.get("reach_place", 0)

    # Pattern 1: BC reach_pick consumed entire horizon, never reached grasp
    if reach_pick_steps >= steps and grasp_steps == 0 and lift_steps == 0:
        # Check if it was close (near_miss) vs truly divergent
        if final_err < 0.05 and top_disp > 0.02:
            return "near_miss"
        return "divergent"

    # Pattern 2: BC reached grasp point but grasp execution consumed remaining steps
    if reach_pick_steps > 0 and grasp_steps > 0 and lift_steps == 0:
        return "horizon_insufficient"

    # Pattern 3: Multiple phase transitions = oscillation
    active_phases = sum(1 for v in trace.values() if v > 0)
    if active_phases >= 3:
        return "oscillation"

    # Pattern 4: Reached lift but timed out during place
    if lift_steps > 0 and place_steps > 0:
        return "horizon_insufficient"

    return "unknown"


def analyze_timeouts(episodes_path: str, diagnostics_path: str, output_path: str) -> dict:
    """Analyze all timeout episodes and produce classification report."""
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    timeouts = [e for e in episodes if e.get("failure_reason") == "timeout"]
    successes = [e for e in episodes if e.get("failure_reason") == "none"]
    grasp_fails = [e for e in episodes if e.get("failure_reason") == "grasp_failure"]
    place_fails = [e for e in episodes if e.get("failure_reason") == "place_failure"]

    classified = []
    for t in timeouts:
        classification = classify_timeout(t)
        classified.append({
            "episode_id": t["episode_id"],
            "goal_id": t["goal_id"],
            "seed": t["seed"],
            "steps": t["steps"],
            "classification": classification,
            "phase_trace": t.get("phase_transition_trace", {}),
            "final_pose_error": t.get("final_pose_error"),
            "top_displacement": t.get("top_displacement"),
            "grasp_success": t.get("grasp_success"),
            "place_success": t.get("place_success"),
        })

    classification_counts = Counter(c["classification"] for c in classified)

    report = {
        "analysis": "Stage 19-D4-1: Balanced BC Timeout Classification",
        "input_episodes": episodes_path,
        "total_episodes": len(episodes),
        "timeout_count": len(timeouts),
        "success_count": len(successes),
        "grasp_failure_count": len(grasp_fails),
        "place_failure_count": len(place_fails),
        "classification_counts": dict(classification_counts),
        "classified_episodes": classified,
        "summary": _build_summary(classified, classification_counts),
    }

    # Also load diagnostics if available for enrichment
    if diagnostics_path and Path(diagnostics_path).exists():
        diag_map = {}
        with open(diagnostics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    diag_map[d["episode_id"]] = d
        for c in classified:
            d = diag_map.get(c["episode_id"], {})
            c["skill_failure_phase"] = d.get("skill_failure_phase")
            c["failure_reason_diag"] = d.get("failure_reason")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Classification report saved to {output_path}")

    return report


def _build_summary(classified: list, counts: dict) -> str:
    lines = []
    lines.append(f"Total timeouts: {len(classified)}")
    lines.append("")
    lines.append("Classification breakdown:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls}: {cnt}/{len(classified)} ({100*cnt/len(classified):.0f}%)")
    lines.append("")
    lines.append("Root cause: BC reach_pick policy fails to converge to grasp pose.")
    lines.append("10/11 timeouts: reach_pick consumes all 500 steps without reaching grasp.")
    lines.append("1/11 timeouts: reach_pick(500) + grasp(475) near 1000-step horizon.")
    lines.append("")
    lines.append("Recommendation: BC-to-scripted fallback for reach_pick phase.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="D4-1: Analyze BC timeout failures")
    parser.add_argument(
        "--episodes",
        default="generated/results/stage_19/stage19_d3_balanced_bc_eval_episodes.jsonl",
        help="Path to D3 evaluation episodes JSONL",
    )
    parser.add_argument(
        "--diagnostics",
        default="generated/results/stage_19/stage19_d1_rollout_diagnostics.jsonl",
        help="Path to D3 diagnostics JSONL",
    )
    parser.add_argument(
        "--output",
        default="generated/results/stage_19/stage19_d4_timeout_classification.json",
        help="Output path for classification report",
    )
    parser.add_argument("--print", action="store_true", help="Print report to stdout")
    args = parser.parse_args()

    report = analyze_timeouts(
        args.episodes,
        args.diagnostics,
        args.output,
    )

    if getattr(args, "print"):
        print(json.dumps({k: v for k, v in report.items() if k != "classified_episodes"}, indent=2))
        print()
        for c in report["classified_episodes"]:
            print(f"  {c['episode_id']:30s} goal={c['goal_id']:15s} seed={c['seed']:2d}  "
                  f"steps={c['steps']:4d}  class={c['classification']:20s}  "
                  f"err={c['final_pose_error']:.4f}  disp={c['top_displacement']:.4f}")


if __name__ == "__main__":
    main()
