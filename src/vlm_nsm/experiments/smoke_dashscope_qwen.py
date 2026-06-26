"""Smoke test an Alibaba DashScope / OpenAI-compatible Qwen endpoint.

This script does not run Robosuite or the planner. It only verifies that
the configured chat completion endpoint can be reached with the OpenAI
Python client and returns text.
"""

from __future__ import annotations

import argparse
import os
import time

from vlm_nsm.language.online_qwen_parser import DEFAULT_DASHSCOPE_API_BASE


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test DashScope-compatible Qwen API")
    parser.add_argument("--api_base", default=DEFAULT_DASHSCOPE_API_BASE)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api_key_env", default="QWEN_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package is not installed. Install with: pip install openai>=1.30.0")
        return 2

    api_key = os.environ.get(args.api_key_env)
    if not api_key or "YOUR_" in api_key:
        print(f"ERROR: {args.api_key_env} is not set. Fill it with your DashScope API key.")
        return 2

    client = OpenAI(api_key=api_key, base_url=args.api_base)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": 'Return exactly this JSON: {"ok": true}',
                }
            ],
            temperature=0,
            max_tokens=64,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"ERROR: request failed: {exc}")
        return 1

    latency = time.perf_counter() - started
    text = response.choices[0].message.content or ""
    print(f"OK model={args.model} api_base={args.api_base} latency={latency:.2f}s")
    print(f"response_preview={text[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
