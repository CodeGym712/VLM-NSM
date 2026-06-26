"""Stage 20-T1: Q vs S dedicated grasp audit.

Compares Stage 20-Q and Stage 20-S checkpoints on direction1/phase1 metrics.
Reuses the existing audit functions from stage20_o_grasp_error_audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.experiments.stage20_o_grasp_error_audit import (
    _collect_predictions,
    _to_float,
    _mse,
    _safe_percentile,
)


def _extract_key_metrics(pred: np.ndarray, target: np.ndarray,
                         direction: np.ndarray, phase: np.ndarray,
                         records: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract key audit metrics from predictions."""
    dim6_error = pred[:, 6] - target[:, 6]
    dim6_abs = np.abs(dim6_error)
    target_close = target[:, 6] > 0.0
    pred_close = pred[:, 6] > 0.0
    sign_ok = target_close == pred_close
    per_sample_mse = ((pred - target) ** 2).mean(axis=1)

    # Overall
    result: dict[str, Any] = {
        "action_dim6_mse": _mse(dim6_error),
        "action_dim6_abs_mean": _to_float(np.mean(dim6_abs)),
        "action_dim6_abs_p95": _safe_percentile(dim6_abs, 95),
        "sign_accuracy": _to_float(np.mean(sign_ok.astype(float))),
        "target_close_ratio": _to_float(np.mean(target_close.astype(float))),
        "pred_close_ratio": _to_float(np.mean(pred_close.astype(float))),
        "sample_mse": _to_float(np.mean(per_sample_mse)),
        "num_samples": int(len(pred)),
    }

    # Direction 1 / Phase 1
    d1p1_mask = (direction == 1) & (phase == 1)
    if d1p1_mask.sum() > 0:
        idx = np.where(d1p1_mask)[0]
        d1p1_dim6_err = pred[idx, 6] - target[idx, 6]
        d1p1_dim6_abs = np.abs(d1p1_dim6_err)
        d1p1_target_close = target[idx, 6] > 0.0
        d1p1_pred_close = pred[idx, 6] > 0.0
        d1p1_sign_ok = d1p1_target_close == d1p1_pred_close
        d1p1_sample_mse = ((pred[idx] - target[idx]) ** 2).mean(axis=1)

        result["direction1_phase1"] = {
            "action_dim6_mse": _mse(d1p1_dim6_err),
            "action_dim6_abs_mean": _to_float(np.mean(d1p1_dim6_abs)),
            "action_dim6_abs_p95": _safe_percentile(d1p1_dim6_abs, 95),
            "sign_accuracy": _to_float(np.mean(d1p1_sign_ok.astype(float))),
            "target_close_ratio": _to_float(np.mean(d1p1_target_close.astype(float))),
            "pred_close_ratio": _to_float(np.mean(d1p1_pred_close.astype(float))),
            "sample_mse": _to_float(np.mean(d1p1_sample_mse)),
            "num_samples": int(len(idx)),
        }

    # Per-direction
    for d in [0, 1]:
        d_mask = direction == d
        if d_mask.sum() > 0:
            idx = np.where(d_mask)[0]
            d_dim6_err = pred[idx, 6] - target[idx, 6]
            result[f"direction_{d}"] = {
                "action_dim6_mse": _mse(d_dim6_err),
                "sample_mse": _to_float(np.mean(((pred[idx] - target[idx]) ** 2).mean(axis=1))),
                "num_samples": int(len(idx)),
            }

    # Per-phase
    for p in range(6):
        p_mask = phase == p
        if p_mask.sum() > 0:
            idx = np.where(p_mask)[0]
            p_dim6_err = pred[idx, 6] - target[idx, 6]
            result[f"phase_{p}"] = {
                "action_dim6_mse": _mse(p_dim6_err),
                "sample_mse": _to_float(np.mean(((pred[idx] - target[idx]) ** 2).mean(axis=1))),
                "num_samples": int(len(idx)),
            }

    return result


def run_comparison(
    dataset_dir: str,
    checkpoint_a: str,
    checkpoint_b: str,
    label_a: str,
    label_b: str,
    split: str = "val",
    batch_size: int = 512,
    device: str = "cuda",
    top_k: int = 10,
) -> dict[str, Any]:
    """Run Q vs S comparison audit."""
    print(f"Loading dataset: {dataset_dir}")
    print(f"Checkpoint A ({label_a}): {checkpoint_a}")
    print(f"Checkpoint B ({label_b}): {checkpoint_b}")

    # Collect predictions for both
    print(f"\n--- Collecting predictions for {label_a} ---")
    result_a = _collect_predictions(
        dataset_dir=dataset_dir, checkpoint=checkpoint_a, split=split, batch_size=batch_size, device_name=device)
    pred_a = result_a["prediction"]
    target_a = result_a["target"]
    records_a = result_a["records"]
    direction_a = result_a["direction"]
    phase_a = result_a["phase"]
    print(f"  {label_a}: {len(pred_a)} samples")

    print(f"\n--- Collecting predictions for {label_b} ---")
    result_b = _collect_predictions(
        dataset_dir=dataset_dir, checkpoint=checkpoint_b, split=split, batch_size=batch_size, device_name=device)
    pred_b = result_b["prediction"]
    target_b = result_b["target"]
    records_b = result_b["records"]
    direction_b = result_b["direction"]
    phase_b = result_b["phase"]
    print(f"  {label_b}: {len(pred_b)} samples")

    # Extract key metrics
    metrics_a = _extract_key_metrics(pred_a, target_a, direction_a, phase_a, records_a)
    metrics_b = _extract_key_metrics(pred_b, target_b, direction_b, phase_b, records_b)

    # Comparison
    d1p1_a = metrics_a.get("direction1_phase1", {})
    d1p1_b = metrics_b.get("direction1_phase1", {})

    comparison = {
        "label_a": label_a,
        "label_b": label_b,
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "dataset_dir": dataset_dir,
        "split": split,
        f"metrics_{label_a}": metrics_a,
        f"metrics_{label_b}": metrics_b,
        "deltas": {
            "action_dim6_mse": _safe_delta(d1p1_a, d1p1_b, "action_dim6_mse"),
            "direction1_phase1_action_dim6_mse": _safe_delta(d1p1_a, d1p1_b, "action_dim6_mse"),
            "direction1_phase1_abs_error_p95": _safe_delta(d1p1_a, d1p1_b, "action_dim6_abs_p95"),
            "direction1_phase1_sign_accuracy": _safe_delta(d1p1_a, d1p1_b, "sign_accuracy"),
            "direction1_phase1_sample_mse": _safe_delta(d1p1_a, d1p1_b, "sample_mse"),
            "direction1_phase1_abs_error_mean": _safe_delta(d1p1_a, d1p1_b, "action_dim6_abs_mean"),
        },
        "verdict": _build_verdict(metrics_a, metrics_b),
    }

    return comparison


def _safe_delta(a: dict, b: dict, key: str) -> dict[str, Any]:
    va = a.get(key)
    vb = b.get(key)
    if va is None or vb is None:
        return {"a": va, "b": vb, "delta": None, "pct_change": None}
    return {
        "a": _to_float(va),
        "b": _to_float(vb),
        "delta": _to_float(vb - va),
        "pct_change": _to_float((vb - va) / va * 100) if va != 0 else None,
    }


def _build_verdict(metrics_a: dict, metrics_b: dict) -> dict[str, Any]:
    """Determine whether S improves over Q."""
    d1p1_a = metrics_a.get("direction1_phase1", {})
    d1p1_b = metrics_b.get("direction1_phase1", {})

    improvements = []
    regressions = []

    def _check(key: str, name: str, lower_is_better: bool = True):
        va = d1p1_a.get(key)
        vb = d1p1_b.get(key)
        if va is not None and vb is not None:
            if lower_is_better:
                if vb < va:
                    improvements.append(f"{name}: {va:.6f} → {vb:.6f} (improved)")
                else:
                    regressions.append(f"{name}: {va:.6f} → {vb:.6f} (worse)")
            else:
                if vb > va:
                    improvements.append(f"{name}: {va:.6f} → {vb:.6f} (improved)")
                else:
                    regressions.append(f"{name}: {va:.6f} → {vb:.6f} (worse)")

    _check("action_dim6_mse", "d1p1 action_dim6_mse", lower_is_better=True)
    _check("action_dim6_abs_mean", "d1p1 abs_mean", lower_is_better=True)
    _check("action_dim6_abs_p95", "d1p1 abs_p95", lower_is_better=True)
    _check("sign_accuracy", "d1p1 sign_accuracy", lower_is_better=False)
    _check("sample_mse", "d1p1 sample_mse", lower_is_better=True)

    n_improved = len(improvements)
    n_total = n_improved + len(regressions)

    return {
        "improvements": improvements,
        "regressions": regressions,
        "improved_count": n_improved,
        "total_count": n_total,
        "is_improvement": n_improved >= n_total * 0.5 if n_total > 0 else None,
    }


def build_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Generate markdown comparison report."""
    la = comparison["label_a"]
    lb = comparison["label_b"]
    ma = comparison[f"metrics_{la}"]
    mb = comparison[f"metrics_{lb}"]
    deltas = comparison["deltas"]
    verdict = comparison["verdict"]
    d1p1_a = ma.get("direction1_phase1", {})
    d1p1_b = mb.get("direction1_phase1", {})

    lines = [
        f"# Stage 20-T1: {la} vs {lb} Grasp Audit",
        "",
        f"**Date**: 2026-05-30",
        f"**Status**: {'IMPROVED' if verdict.get('is_improvement') else 'MIXED'}",
        "",
        "## Checkpoints",
        "",
        f"| Label | Path |",
        f"|-------|------|",
        f"| {la} | `{comparison['checkpoint_a']}` |",
        f"| {lb} | `{comparison['checkpoint_b']}` |",
        "",
        f"**Dataset**: `{comparison['dataset_dir']}`  ",
        f"**Split**: `{comparison['split']}`  ",
        f"**Samples**: {la}={ma['num_samples']}, {lb}={mb['num_samples']}",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | {la} | {lb} | Delta |",
        f"|--------|------|------|-------|",
    ]

    for key, name in [
        ("action_dim6_mse", "Action dim6 MSE"),
        ("action_dim6_abs_mean", "Action dim6 abs mean"),
        ("action_dim6_abs_p95", "Action dim6 abs p95"),
        ("sign_accuracy", "Sign accuracy"),
        ("sample_mse", "Sample MSE"),
    ]:
        va = ma.get(key)
        vb = mb.get(key)
        d = _to_float(vb - va) if va is not None and vb is not None else None
        lines.append(f"| {name} | {_fmt(va)} | {_fmt(vb)} | {_fmt_delta(d)} |")

    lines.extend([
        "",
        "## Direction 1 / Phase 1 (Key Bottleneck)",
        "",
        f"| Metric | {la} | {lb} | Delta | % Change |",
        f"|--------|------|------|-------|----------|",
    ])
    for key, name in [
        ("action_dim6_mse", "Action dim6 MSE"),
        ("action_dim6_abs_mean", "Abs error mean"),
        ("action_dim6_abs_p95", "Abs error p95"),
        ("sign_accuracy", "Sign accuracy"),
        ("sample_mse", "Sample MSE"),
    ]:
        d = deltas.get(f"direction1_phase1_{key}", {})
        lines.append(
            f"| {name} | {_fmt(d.get('a'))} | {_fmt(d.get('b'))} | "
            f"{_fmt_delta(d.get('delta'))} | {_fmt_pct(d.get('pct_change'))} |"
        )

    lines.extend([
        "",
        "## Per-Direction Comparison",
        "",
        f"| Direction | Metric | {la} | {lb} | Delta |",
        f"|-----------|--------|------|------|-------|",
    ])
    for d in [0, 1]:
        for metric, metric_name in [("action_dim6_mse", "dim6 MSE"), ("sample_mse", "Sample MSE")]:
            va = ma.get(f"direction_{d}", {}).get(metric)
            vb = mb.get(f"direction_{d}", {}).get(metric)
            delta = _to_float(vb - va) if va is not None and vb is not None else None
            lines.append(f"| {d} | {metric_name} | {_fmt(va)} | {_fmt(vb)} | {_fmt_delta(delta)} |")

    lines.extend([
        "",
        "## Per-Phase Action dim6 MSE",
        "",
        f"| Phase | {la} | {lb} | Delta |",
        f"|-------|------|------|-------|",
    ])
    for p in range(6):
        va = ma.get(f"phase_{p}", {}).get("action_dim6_mse")
        vb = mb.get(f"phase_{p}", {}).get("action_dim6_mse")
        delta = _to_float(vb - va) if va is not None and vb is not None else None
        lines.append(f"| {p} | {_fmt(va)} | {_fmt(vb)} | {_fmt_delta(delta)} |")

    lines.extend([
        "",
        "## Verdict",
        "",
        f"**{verdict.get('improved_count', 0)}/{verdict.get('total_count', 0)} direction1/phase1 metrics improved**",
        "",
    ])
    for imp in verdict.get("improvements", []):
        lines.append(f"- ✅ {imp}")
    for reg in verdict.get("regressions", []):
        lines.append(f"- ❌ {reg}")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Audit is offline (on validation set), not online rollout.",
        "- No claim of closed-loop success.",
        "- holding_signal is obs[4] heuristic, not true MuJoCo contact.",
        "- Observed improvements are correlational — online validation needed.",
    ])

    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.6f}"


def _fmt_delta(d: Any) -> str:
    if d is None:
        return "N/A"
    val = float(d)
    if val < 0:
        return f"**{val:+.6f}** ✅"
    return f"{val:+.6f}"


def _fmt_pct(p: Any) -> str:
    if p is None:
        return "N/A"
    return f"{float(p):+.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Stage 20-T1: Q vs S grasp audit")
    parser.add_argument("--dataset_dir",
                        default="generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled_holding_obs4_stage20_s")
    parser.add_argument("--checkpoint_a",
                        default="generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_gripper_magnitude_weight025_stage20_q/best.pt")
    parser.add_argument("--checkpoint_b",
                        default="generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_holding_aux_stage20_s/best.pt")
    parser.add_argument("--label_a", default="Stage20Q")
    parser.add_argument("--label_b", default="Stage20S")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default="generated/results/stage_20/stage20_t_q_vs_s_grasp_audit.json")
    parser.add_argument("--output_md", default="generated/results/stage_20/stage20_t_q_vs_s_grasp_audit.md")
    args = parser.parse_args()

    comparison = run_comparison(
        dataset_dir=args.dataset_dir,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        label_a=args.label_a,
        label_b=args.label_b,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_json, comparison)
    output_md.write_text(build_comparison_markdown(comparison), encoding="utf-8")
    print(f"\nWrote {output_json}")
    print(f"Wrote {output_md}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
