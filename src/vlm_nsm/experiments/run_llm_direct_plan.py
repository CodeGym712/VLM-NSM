from __future__ import annotations

import argparse
import json

from vlm_nsm.baselines.llm_direct_plan import run_baseline
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["hanoi", "stacking"])
    parser.add_argument("--n_blocks", type=int, required=True, choices=[3, 4, 5])
    parser.add_argument("--mode", default="oracle_direct_plan", choices=["oracle_direct_plan", "corrupted_direct_plan"])
    args = parser.parse_args()

    result = run_baseline(args.task, args.n_blocks, args.mode)
    mode_tag = "oracle" if args.mode == "oracle_direct_plan" else "corrupted"
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"llm_direct_plan_{args.task}_{args.n_blocks}_{mode_tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"result_path={out_path}")


if __name__ == "__main__":
    main()
