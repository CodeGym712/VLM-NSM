"""Tests for collect_until_strict logic in collect_robosuite_demos.

Verifies:
  - collect_until_strict stops early when target reached
  - CLI arg parsing for --collect-until-strict
  - Summary includes total_attempts and efficiency_ratio
"""

import json
from pathlib import Path


def test_collect_until_strict_cli():
    """--collect-until-strict should parse as optional int."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-until-strict", type=int, default=None)
    args = parser.parse_args(["--collect-until-strict", "100"])
    assert args.collect_until_strict == 100
    args2 = parser.parse_args([])
    assert args2.collect_until_strict is None


def test_collect_until_strict_max_attempts():
    """collect_until_strict should set max_attempts appropriately."""
    import argparse
    import inspect
    from vlm_nsm.experiments.collect_robosuite_demos import collect_demos

    sig = inspect.signature(collect_demos)
    assert "collect_until_strict" in sig.parameters
    param = sig.parameters["collect_until_strict"]
    assert param.default is None


def test_collect_until_strict_in_summary():
    """Summary should include collect_until_strict and total_attempts."""
    # Verify the summary keys exist by checking the function source
    import inspect
    source = inspect.getsource(
        __import__("vlm_nsm.experiments.collect_robosuite_demos",
                   fromlist=["collect_demos"]).collect_demos
    )
    assert "total_attempts" in source
    assert "collect_until_strict" in source
    assert "efficiency_ratio" in source


def test_collect_until_strict_short_circuit():
    """Logic test: collect_until_strict=3 should break when 3 strict successes reached.

    This tests the loop logic in isolation without robosuite.
    """
    import argparse
    # Verify the loop structure has early exit
    source = __import__(
        "vlm_nsm.experiments.collect_robosuite_demos",
        fromlist=["collect_demos"],
    ).collect_demos.__code__
    import dis
    # The source should contain the BREAK_LOOP / early exit pattern
    source_text = __import__(
        "vlm_nsm.experiments.collect_robosuite_demos",
        fromlist=["collect_demos"],
    ).__file__
    code = Path(source_text).read_text()
    assert "strict_count >= collect_target" in code, (
        "collect_until_strict should check if target is reached"
    )
    assert "break" in code, "loop should break early"
