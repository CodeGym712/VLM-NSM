#!/usr/bin/env python3
"""Matrix runner for language variation experiments.

Runs all combinations of model, task, prompt_variant, alignment_mode,
skipping existing results when --skip_existing is set.

Usage:
  # Mock test
  python -m vlm_nsm.experiments.run_language_matrix \\
    --models mock --tasks hanoi,stacking --difficulty hard \\
    --prompt_variants full,medium,weak \\
    --alignment_modes full,no_alignment \\
    --skip_existing false

  # Qwen2.5-32B hard matrix
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlm_nsm.experiments.run_language_matrix \\
    --models qwen25_32b \\
    --model_paths qwen25_32b=/path/to/Qwen2.5-32B-Instruct \\
    --tasks hanoi,stacking --difficulty hard \\
    --prompt_variants full,medium,weak \\
    --alignment_modes full,no_alignment,no_repair,no_alias_repair,no_completeness_repair \\
    --skip_existing true
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir

RESULTS_DIR = ensure_dir(PROJECT_ROOT / "generated" / "results")


def _result_exists(results_dir: Path, model_name: str, task: str, difficulty: str,
                   prompt_variant: str, alignment_mode: str) -> bool:
    fname = f"language_variation_{task}_{difficulty}_{model_name}_{prompt_variant}_{alignment_mode}.json"
    return (results_dir / fname).exists()


def _run_single(spec: dict, cuda_devices: str | None, output_dir: Path | None = None) -> dict:
    """Run one experiment combination via subprocess."""
    model = spec["model"]
    task = spec["task"]
    difficulty = spec["difficulty"]
    prompt_variant = spec["prompt_variant"]
    alignment_mode = spec["alignment_mode"]
    model_path = spec.get("model_path")

    parser = "qwen_text" if model_path else "mock"

    cmd = [
        sys.executable, "-m", "vlm_nsm.experiments.run_language_variation",
        "--task", task,
        "--n_blocks", "3",
        "--parser", parser,
        "--difficulty", difficulty,
        "--prompt_variant", prompt_variant,
        "--alignment_mode", alignment_mode,
        "--model_name", model,
    ]
    if model_path:
        cmd += ["--model_path", model_path]
    if output_dir:
        cmd += ["--output_dir", str(output_dir)]

    env = None
    if cuda_devices:
        env = {**dict(subprocess.os.environ), "CUDA_VISIBLE_DEVICES": cuda_devices}

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {model} / {task} / {difficulty} / {prompt_variant} / {alignment_mode}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
        elapsed = time.time() - start
        if result.returncode == 0:
            print(result.stdout)
            return {"status": "completed", "elapsed": round(elapsed, 1), "error": ""}
        else:
            stderr = result.stderr[-500:] if result.stderr else ""
            print(f"FAILED (rc={result.returncode}, {elapsed:.1f}s)")
            if result.stdout:
                print(result.stdout[-300:])
            if stderr:
                print(f"stderr: {stderr}")
            return {"status": "failed", "elapsed": round(elapsed, 1), "error": stderr}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.1f}s")
        return {"status": "timeout", "elapsed": round(elapsed, 1), "error": "timeout"}
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.1f}s: {e}")
        return {"status": "error", "elapsed": round(elapsed, 1), "error": str(e)}


def main():
    p = argparse.ArgumentParser(description="Run language variation experiment matrix")
    p.add_argument("--models", required=True, help="Comma-separated model names (e.g. mock,qwen25_32b)")
    p.add_argument("--model_paths", default="", help="Comma-separated name=path mappings (e.g. qwen25_32b=/path)")
    p.add_argument("--tasks", default="hanoi,stacking", help="Comma-separated tasks")
    p.add_argument("--difficulty", default="hard", help="Difficulty level")
    p.add_argument("--prompt_variants", default="full,medium,weak", help="Comma-separated prompt variants")
    p.add_argument("--alignment_modes", default="full,no_alignment", help="Comma-separated alignment modes")
    p.add_argument("--skip_existing", default="true", choices=["true", "false"], help="Skip if result file exists")
    p.add_argument("--cuda_devices", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--output_dir", default=None, help="Custom output directory for results")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    difficulty = args.difficulty
    prompt_variants = [v.strip() for v in args.prompt_variants.split(",")]
    alignment_modes = [m.strip() for m in args.alignment_modes.split(",")]
    skip_existing = args.skip_existing == "true"
    cuda_devices = args.cuda_devices
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Parse model_paths
    model_path_map: dict[str, str | None] = {}
    for m in models:
        model_path_map[m] = None
    if args.model_paths:
        for entry in args.model_paths.split(","):
            if "=" in entry:
                name, path = entry.split("=", 1)
                model_path_map[name.strip()] = path.strip()

    # Generate all combinations
    combos = list(itertools.product(models, tasks, prompt_variants, alignment_modes))
    print(f"Matrix: {len(models)} models x {len(tasks)} tasks x "
          f"{len(prompt_variants)} prompts x {len(alignment_modes)} alignments = {len(combos)} total")

    manifest: list[dict] = []
    run_id = 0

    for model, task, prompt_variant, alignment_mode in combos:
        run_id += 1
        exists = _result_exists(output_dir, model, task, difficulty, prompt_variant, alignment_mode)

        entry = {
            "run_id": run_id,
            "model_name": model,
            "task": task,
            "difficulty": difficulty,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "result_path": str(output_dir / f"language_variation_{task}_{difficulty}_{model}_{prompt_variant}_{alignment_mode}.json"),
            "status": "",
            "error": "",
        }

        if skip_existing and exists:
            print(f"\n[{run_id}/{len(combos)}] SKIP (exists): {model} / {task} / {prompt_variant} / {alignment_mode}")
            entry["status"] = "skipped"
            manifest.append(entry)
            continue

        print(f"\n[{run_id}/{len(combos)}] {model} / {task} / {prompt_variant} / {alignment_mode}")

        spec = {
            "model": model,
            "task": task,
            "difficulty": difficulty,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "model_path": model_path_map.get(model),
        }

        result = _run_single(spec, cuda_devices, output_dir)
        entry["status"] = result["status"]
        entry["error"] = result["error"]
        manifest.append(entry)

        # Brief pause between runs
        time.sleep(2)

    # Write manifest
    manifest_path = output_dir / "language_matrix_manifest.csv"
    fieldnames = [
        "run_id", "model_name", "task", "difficulty",
        "prompt_variant", "alignment_mode", "result_path", "status", "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\nManifest written to {manifest_path}")

    # Summary
    completed = sum(1 for e in manifest if e["status"] == "completed")
    skipped = sum(1 for e in manifest if e["status"] == "skipped")
    failed = sum(1 for e in manifest if e["status"] in ("failed", "timeout", "error"))
    print(f"\nMatrix summary: {completed} completed, {skipped} skipped, {failed} failed")

    if failed > 0:
        print("Failed entries:")
        for e in manifest:
            if e["status"] in ("failed", "timeout", "error"):
                print(f"  #{e['run_id']}: {e['model_name']}/{e['task']}/{e['prompt_variant']}/{e['alignment_mode']} -> {e['status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
