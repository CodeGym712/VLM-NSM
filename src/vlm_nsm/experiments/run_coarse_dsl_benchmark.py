"""60-Task Coarse DSL Benchmark — compare four program generation modes.

Modes:
  A. goal-only + BFS planner (baseline)
  B. phase-level DSL (old: reach/grasp/lift/...)
  C. coarse pick/place DSL (new: pick/place/move/ensure/achieve)
  D. oracle coarse program (human-authored reference)

Metrics:
  - program validity (schema check)
  - precondition violation count
  - subgoal correctness (does plan achieve goal?)
  - forbidden op count (reach/grasp/lift in output = violation)
  - parse success rate
  - verification pass rate

Output: generated/results/coarse_dsl_benchmark/
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_nsm.core.goal_parser import MockGoalParser, extract_json_object
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.neuro_symbolic.verifier_bank import (
    VerifierBank, VerificationReport,
    GoalVerifier, SubgoalVerifier, PlanVerifier,
    OptionContractVerifier, RepairVerifier, CoarseOptionVerifier,
    standard_verifier_bank,
)
from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser
from vlm_nsm.language.coarse_dsl import CoarseTaskProgram, CoarseStep
from vlm_nsm.neuro_symbolic.coarse_planner_bridge import CoarsePlannerBridge

# ── 60 Task Instructions ─────────────────────────────────────────────────────
# Difficulty levels:
#   L1: simple 2-block (20 tasks)
#   L2: 3-block stacking (20 tasks)
#   L3: unstacking / rearrangement (10 tasks)
#   L4: complex multi-step (10 tasks)

BENCHMARK_TASKS = [
    # === L1: Simple 2-block (20 tasks) ===
    {"id": "L1_01", "instruction": "Stack cube1 on cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_02", "instruction": "Put cube1 on top of cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_03", "instruction": "Place cube2 on cube1",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L1_04", "instruction": "The red cube should be on the green cube",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_05", "instruction": "Pick up cube1 and put it down on cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_06", "instruction": "Make a tower: cube1 at bottom, cube2 on top",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L1_07", "instruction": "Move cube1 onto cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_08", "instruction": "Cube2 goes on top of cube1",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L1_09", "instruction": "Stack the cubes with cube1 on cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_10", "instruction": "Put cube2 under cube1",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_11", "instruction": "cube1 on cube2 please",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_12", "instruction": "Put the top cube on the bottom cube",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_13", "instruction": "Stacking: cube1 -> cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_14", "instruction": "Grab cube1 and stack it on cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_15", "instruction": "Put cube1 on cube2, keep cube2 on table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_16", "instruction": "I want cube2 supporting cube1",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_17", "instruction": "Can you put the left cube on the right cube?",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_18", "instruction": "Build a 2-block stack: cube2 on cube1",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L1_19", "instruction": "Make on(cube1, cube2) true",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L1_20", "instruction": "Place cube1 directly above cube2",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},

    # === L2: 3-block stacking (20 tasks) ===
    {"id": "L2_01", "instruction": "Stack cube1 on cube2, then cube2 on cube3",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_02", "instruction": "Make a 3-block tower: cube3 on table, cube2 on cube3, cube1 on cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_03", "instruction": "Stack all three cubes: cube1 on top, cube3 at bottom",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_04", "instruction": "Put cube1 on cube2 and cube2 on cube3",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_05", "instruction": "Stack in order: bottom cube3, middle cube2, top cube1",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_06", "instruction": "Cube1 on cube2, cube3 stays on table",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_07", "instruction": "Stack cube2 on cube3 first, then put cube1 on cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_08", "instruction": "Build tower: cube3, then cube1 on cube3, then cube2 on cube1",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube3"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L2_09", "instruction": "Place cube2 on cube1, place cube3 on cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"},
                {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube2"}]},
    {"id": "L2_10", "instruction": "I need cube3 on table, cube1 on cube3, cube2 on cube1",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube3"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L2_11", "instruction": "Stack the blocks: cube3 base, cube1 middle, cube2 top",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube3"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L2_12", "instruction": "Arrange: on(cube1, cube2), on(cube2, cube3), on(cube3, table)",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_13", "instruction": "Stack cube1 and cube2 on cube3, with cube1 above cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_14", "instruction": "Put the three cubes in a single stack",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_15", "instruction": "Make a tower with all cubes, cube1 at the very top",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_16", "instruction": "cube1 → cube2 → cube3",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_17", "instruction": "Stack with cube3 supporting cube1 supporting cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube3"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube1"}]},
    {"id": "L2_18", "instruction": "Arrange cubes so cube2 is on cube3, and cube1 is on cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_19", "instruction": "Stack cube3 under cube2 under cube1",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L2_20", "instruction": "Construct: cube1/cube2/cube3 from top to bottom",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},

    # === L3: Unstacking / rearrangement (10 tasks) ===
    {"id": "L3_01", "instruction": "Unstack cube1 from cube2 and put it on the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]},
    {"id": "L3_02", "instruction": "Take cube1 off cube2 and place it on the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]},
    {"id": "L3_03", "instruction": "Move cube2 from on top of cube1 to the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]},
    {"id": "L3_04", "instruction": "Swap: put cube1 where cube2 is and cube2 where cube1 is",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]},
    {"id": "L3_05", "instruction": "Clear cube2 by removing cube1 from it",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]},
    {"id": "L3_06", "instruction": "Put cube1 down on the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]},
    {"id": "L3_07", "instruction": "Return all cubes to the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]},
    {"id": "L3_08", "instruction": "Disassemble the stack: put both cubes on the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]},
    {"id": "L3_09", "instruction": "The top cube should be moved to the table",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]},
    {"id": "L3_10", "instruction": "Rearrange: cube2 on table, cube1 on table (both separate)",
     "objects": ["cube1", "cube2", "table"], "n_blocks": 2,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]},

    # === L4: Complex multi-step (10 tasks) ===
    {"id": "L4_01", "instruction": "Stack cube1 on cube2, then unstack cube1 to table, then stack cube1 on cube3",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube3"}]},
    {"id": "L4_02", "instruction": "Ensure cube2 is clear, then place cube3 on cube2, then put cube1 on cube3",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "ensure", "predicate": ["clear", "cube2"]},
                {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube2"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube3"}]},
    {"id": "L4_03", "instruction": "Pick cube3, place on table, pick cube1, place on cube2, pick cube3, place on cube1",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "table"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"},
                {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube1"}]},
    {"id": "L4_04", "instruction": "Build tower cube3/cube2/cube1, then move cube1 to table",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]},
    {"id": "L4_05", "instruction": "Move cube1 to table, move cube2 to table, stack cube3 on cube1",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"},
                {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube1"}]},
    {"id": "L4_06", "instruction": "Achieve on(cube1, cube2) and on(cube2, cube3)",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "achieve", "predicate": ["on", "cube2", "cube3"]},
                {"op": "achieve", "predicate": ["on", "cube1", "cube2"]}]},
    {"id": "L4_07", "instruction": "Make sure cube2 is on cube3, then verify cube1 is on cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
    {"id": "L4_08", "instruction": "Stack cube3 on cube1, and separately stack cube2 on table",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube1"}]},
    {"id": "L4_09", "instruction": "Build a pyramid base (no actual stacking), place all cubes separate on table",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"},
                {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"},
                {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "table"}]},
    {"id": "L4_10", "instruction": "Clear the table, then stack cube2 on cube3, then cube1 on cube2",
     "objects": ["cube1", "cube2", "cube3", "table"], "n_blocks": 3,
     "oracle": [{"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "cube3"},
                {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]},
]


# ── Benchmark Result ──────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str = ""
    instruction: str = ""
    mode: str = ""  # A | B | C | D
    success: bool = False
    parse_ok: bool = False
    verification_passed: bool = False
    n_steps: int = 0
    n_forbidden_ops: int = 0
    n_precondition_violations: int = 0
    n_step_failures: int = 0
    goal_achieved: bool = False
    error: str = ""
    raw_output: str = ""
    steps: list | None = None
    verification_report: dict | None = None
    latency_s: float = 0.0


# ── Benchmark Runner ──────────────────────────────────────────────────────────

class CoarseDSLBenchmark:
    """Run the 60-task coarse DSL benchmark."""

    def __init__(self, api_key: str | None = None, model: str = "qwen-plus",
                 api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 output_dir: str = "generated/results/coarse_dsl_benchmark"):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.model = model
        self.api_base = api_base
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verifier bank with all objects
        all_objects = ["cube1", "cube2", "cube3", "table"]
        self.verifier_bank = standard_verifier_bank(all_objects)

        # Parsers
        self.qwen_parser = OnlineQwenGoalParser(
            model=model, api_key=self.api_key, api_base=api_base,
            temperature=0.0, max_tokens=1024, retries=1)

        # Coarse verifier
        self.coarse_verifier = CoarseOptionVerifier(valid_objects=all_objects)

        # Planner bridge
        self.bridge = CoarsePlannerBridge()

    def run(self) -> dict:
        """Run full benchmark across all 60 tasks × 4 modes."""
        all_results: list[TaskResult] = []

        for task in BENCHMARK_TASKS:
            task_id = task["id"]
            instruction = task["instruction"]
            objects = task["objects"]
            oracle = task["oracle"]

            # ── Mode A: goal-only + BFS ──
            r_a = self._run_goal_only(task_id, instruction, objects)
            all_results.append(r_a)

            # ── Mode B: phase-level DSL (old) ──
            r_b = self._run_phase_dsl(task_id, instruction, objects)
            all_results.append(r_b)

            # ── Mode C: coarse DSL (new) ──
            r_c = self._run_coarse_dsl(task_id, instruction, objects)
            all_results.append(r_c)

            # ── Mode D: oracle coarse program ──
            r_d = self._run_oracle(task_id, instruction, objects, oracle)
            all_results.append(r_d)

            # Progress
            done = len([r for r in all_results if r.task_id == task_id])
            print(f"[{task_id}] {done}/4 modes complete")

        # Save results
        summary = self._summarize(all_results)
        self._save(all_results, summary)
        return summary

    def _run_goal_only(self, task_id: str, instruction: str,
                       objects: list[str]) -> TaskResult:
        """Mode A: goal-only parsing + BFS planner."""
        result = TaskResult(task_id=task_id, instruction=instruction, mode="A_goal_only")
        t0 = time.perf_counter()

        try:
            parse_result = self.qwen_parser.parse(
                instruction, objects=objects, task_type="stacking",
                n_blocks=len([o for o in objects if o != "table"]))
            result.latency_s = time.perf_counter() - t0
            result.raw_output = parse_result.raw_output

            if not parse_result.success:
                result.error = f"parse_failed: {parse_result.error}"
                return result
            result.parse_ok = True

            goal_json = parse_result.goal_json or {}
            report = self.verifier_bank.verify_all(
                goal_json=goal_json, task_type="stacking")
            result.verification_passed = report.passed
            result.verification_report = report.to_dict()
            result.success = report.passed

            # Check for forbidden ops
            raw = parse_result.raw_output or ""
            forbidden = ["reach", "grasp", "lift", "reach_place", "release_place"]
            result.n_forbidden_ops = sum(1 for f in forbidden if f in raw.lower())

            preds = goal_json.get("goal_predicates", [])
            result.n_steps = len(preds)
            result.steps = preds

        except Exception as e:
            result.error = str(e)
            result.latency_s = time.perf_counter() - t0

        return result

    def _run_phase_dsl(self, task_id: str, instruction: str,
                       objects: list[str]) -> TaskResult:
        """Mode B: phase-level DSL (reach/grasp/lift/...)."""
        result = TaskResult(task_id=task_id, instruction=instruction, mode="B_phase_dsl")
        t0 = time.perf_counter()

        try:
            # Use goal parser — phase DSL isn't generated by Qwen, it's the old format
            parse_result = self.qwen_parser.parse(
                instruction, objects=objects, task_type="stacking",
                n_blocks=len([o for o in objects if o != "table"]))
            result.latency_s = time.perf_counter() - t0
            result.raw_output = parse_result.raw_output

            if not parse_result.success:
                result.error = f"parse_failed: {parse_result.error}"
                return result
            result.parse_ok = True

            # Count forbidden ops in output — phase DSL ALLOWS reach/grasp/lift
            # so we check if any appear (they shouldn't in coarse)
            raw = parse_result.raw_output or ""
            phase_ops = ["reach", "grasp", "lift", "reach_place", "release_place", "retreat"]
            result.n_forbidden_ops = sum(1 for f in phase_ops if f in raw.lower())

            goal_json = parse_result.goal_json or {}
            report = self.verifier_bank.verify_all(
                goal_json=goal_json, task_type="stacking")
            result.verification_passed = report.passed
            result.verification_report = report.to_dict()

            # Phase DSL doesn't generate steps — just goals
            preds = goal_json.get("goal_predicates", [])
            result.n_steps = len(preds)
            result.steps = preds
            result.success = report.passed

        except Exception as e:
            result.error = str(e)
            result.latency_s = time.perf_counter() - t0

        return result

    def _run_coarse_dsl(self, task_id: str, instruction: str,
                        objects: list[str]) -> TaskResult:
        """Mode C: coarse pick/place DSL (new system)."""
        result = TaskResult(task_id=task_id, instruction=instruction, mode="C_coarse_dsl")
        t0 = time.perf_counter()

        try:
            parse_result = self.qwen_parser.parse(
                instruction, objects=objects, task_type="coarse_stacking",
                n_blocks=len([o for o in objects if o != "table"]))
            result.latency_s = time.perf_counter() - t0
            result.raw_output = parse_result.raw_output

            if not parse_result.success:
                result.error = f"parse_failed: {parse_result.error}"
                return result
            result.parse_ok = True

            data = parse_result.goal_json or {}
            steps = data.get("steps", [])

            # Count forbidden phase-level ops in output
            raw = parse_result.raw_output or ""
            forbidden = ["reach", "grasp", "lift", "reach_place", "release_place"]
            result.n_forbidden_ops = sum(1 for f in forbidden if f in raw.lower())

            # Validate each step structurally
            valid_objects = set(objects)
            for step in steps:
                if isinstance(step, dict):
                    op = step.get("op", "")
                    if op in forbidden:
                        result.n_forbidden_ops += 1
                    if op == "place" and step.get("object") == step.get("target"):
                        result.n_precondition_violations += 1

            # Coarse verifier check
            coarse_report = self.coarse_verifier.check(
                coarse_steps=steps)
            result.verification_passed = coarse_report.passed
            result.verification_report = coarse_report.to_dict()
            result.n_step_failures = len(coarse_report.failures)
            result.n_precondition_violations += len([
                f for f in coarse_report.failures
                if "pick" in f[0] or "place" in f[0] or "precondition" in f[1]
            ])

            # Goal check
            if steps:
                try:
                    program = CoarseTaskProgram.from_json({"steps": steps})
                    if program:
                        goals = program.to_goal_predicates()
                        if goals:
                            goal_json = {"goal_predicates": goals}
                            goal_report = self.verifier_bank.verify_all(
                                goal_json=goal_json, task_type="stacking")
                            result.goal_achieved = goal_report.passed
                except Exception:
                    pass

            result.n_steps = len(steps)
            result.steps = steps
            result.success = coarse_report.passed and result.n_forbidden_ops == 0

        except Exception as e:
            result.error = str(e)
            result.latency_s = time.perf_counter() - t0

        return result

    def _run_oracle(self, task_id: str, instruction: str,
                    objects: list[str], oracle_steps: list) -> TaskResult:
        """Mode D: oracle coarse program (ground truth)."""
        result = TaskResult(task_id=task_id, instruction=instruction, mode="D_oracle")
        t0 = time.perf_counter()

        coarse_report = self.coarse_verifier.check(coarse_steps=oracle_steps)
        result.latency_s = time.perf_counter() - t0
        result.verification_passed = coarse_report.passed
        result.verification_report = coarse_report.to_dict()
        result.n_steps = len(oracle_steps)
        result.steps = oracle_steps
        result.parse_ok = True
        result.n_forbidden_ops = 0
        result.n_step_failures = len(coarse_report.failures)
        result.success = coarse_report.passed

        # Goal check
        try:
            program = CoarseTaskProgram.from_json({"steps": oracle_steps})
            if program:
                goals = program.to_goal_predicates()
                if goals:
                    goal_json = {"goal_predicates": goals}
                    goal_report = self.verifier_bank.verify_all(
                        goal_json=goal_json, task_type="stacking")
                    result.goal_achieved = goal_report.passed
        except Exception:
            pass

        if not result.success:
            result.error = f"oracle_coarse_failures: {coarse_report.explain()}"

        return result

    def _summarize(self, results: list[TaskResult]) -> dict:
        """Compute aggregate statistics."""
        by_mode: dict[str, list[TaskResult]] = {}
        for r in results:
            by_mode.setdefault(r.mode, []).append(r)

        summary = {
            "n_tasks": len(BENCHMARK_TASKS),
            "n_total_results": len(results),
            "modes": {},
        }

        for mode, mode_results in sorted(by_mode.items()):
            n = len(mode_results)
            ok = [r for r in mode_results if r.success]
            parse_ok = [r for r in mode_results if r.parse_ok]
            verify_ok = [r for r in mode_results if r.verification_passed]
            goal_ok = [r for r in mode_results if r.goal_achieved]

            total_forbidden = sum(r.n_forbidden_ops for r in mode_results)
            total_violations = sum(r.n_precondition_violations for r in mode_results)
            avg_latency = sum(r.latency_s for r in mode_results) / n if n else 0
            avg_steps = sum(r.n_steps for r in mode_results) / n if n else 0

            # Per-difficulty breakdown
            by_level: dict[str, dict] = {}
            for r in mode_results:
                level = r.task_id.split("_")[0]
                if level not in by_level:
                    by_level[level] = {"n": 0, "n_success": 0}
                by_level[level]["n"] += 1
                if r.success:
                    by_level[level]["n_success"] += 1

            summary["modes"][mode] = {
                "n": n,
                "success_rate": len(ok) / n if n else 0,
                "parse_rate": len(parse_ok) / n if n else 0,
                "verification_rate": len(verify_ok) / n if n else 0,
                "goal_rate": len(goal_ok) / n if n else 0,
                "total_forbidden_ops": total_forbidden,
                "total_precondition_violations": total_violations,
                "avg_latency_s": round(avg_latency, 3),
                "avg_steps": round(avg_steps, 1),
                "by_difficulty": {lvl: {
                    "success_rate": d["n_success"] / d["n"] if d["n"] else 0
                } for lvl, d in sorted(by_level.items())},
            }

        return summary

    def _save(self, results: list[TaskResult], summary: dict):
        """Save results to disk."""
        # Summary
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        # Detailed results
        detail_path = self.output_dir / "detailed_results.json"
        detail_data = []
        for r in results:
            detail_data.append({
                "task_id": r.task_id,
                "instruction": r.instruction,
                "mode": r.mode,
                "success": r.success,
                "parse_ok": r.parse_ok,
                "verification_passed": r.verification_passed,
                "n_steps": r.n_steps,
                "n_forbidden_ops": r.n_forbidden_ops,
                "n_precondition_violations": r.n_precondition_violations,
                "n_step_failures": r.n_step_failures,
                "goal_achieved": r.goal_achieved,
                "error": r.error,
                "latency_s": r.latency_s,
                "steps": r.steps,
                "verification_report": r.verification_report,
            })
        detail_path.write_text(json.dumps(detail_data, indent=2, ensure_ascii=False))

        # Human-readable table
        table_path = self.output_dir / "summary_table.txt"
        lines = [
            "=" * 80,
            "60-Task Coarse DSL Benchmark Results",
            "=" * 80,
            "",
            f"{'Mode':<20} {'Success':>8} {'Parse':>8} {'Verify':>8} {'Forbid':>8} {'Violat':>8} {'Lat(s)':>8}",
            "-" * 80,
        ]
        for mode in ["A_goal_only", "B_phase_dsl", "C_coarse_dsl", "D_oracle"]:
            m = summary["modes"].get(mode, {})
            if not m:
                continue
            lines.append(
                f"{mode:<20} {m['success_rate']:>7.1%} {m['parse_rate']:>7.1%} "
                f"{m['verification_rate']:>7.1%} {m['total_forbidden_ops']:>8} "
                f"{m['total_precondition_violations']:>8} {m['avg_latency_s']:>8.2f}"
            )
        lines.append("-" * 80)
        lines.append("")
        lines.append("Per-difficulty breakdown:")
        for mode in ["A_goal_only", "B_phase_dsl", "C_coarse_dsl", "D_oracle"]:
            m = summary["modes"].get(mode, {})
            if not m:
                continue
            lines.append(f"  {mode}:")
            for lvl, stats in m.get("by_difficulty", {}).items():
                lines.append(f"    {lvl}: {stats['success_rate']:.1%}")
        lines.append("")
        table_path.write_text("\n".join(lines))

        print(f"Results saved to {self.output_dir}")
        print(f"  Summary: {summary_path}")
        print(f"  Details: {detail_path}")
        print(f"  Table:   {table_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="60-task coarse DSL benchmark")
    ap.add_argument("--api-key", type=str, default=None,
                    help="Qwen API key (or set QWEN_API_KEY env var)")
    ap.add_argument("--api-base", type=str,
                    default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--model", type=str, default="qwen-plus")
    ap.add_argument("--output-dir", type=str,
                    default="generated/results/coarse_dsl_benchmark")
    ap.add_argument("--mock", action="store_true",
                    help="Use mock parser (no API calls)")
    ap.add_argument("--tasks", type=str, default=None,
                    help="Comma-separated task IDs to run (default: all 60)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("QWEN_API_KEY", "")
    if not api_key and not args.mock:
        print("WARNING: No API key provided. Use --api-key or set QWEN_API_KEY.")
        print("Continuing with mock mode...")

    if args.mock or not api_key:
        # Mock mode: use MockGoalParser for all modes
        print("Running in MOCK mode (no API calls)")
        return _run_mock_benchmark(args)

    benchmark = CoarseDSLBenchmark(
        api_key=api_key, model=args.model, api_base=args.api_base,
        output_dir=args.output_dir)
    summary = benchmark.run()

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for mode in ["A_goal_only", "B_phase_dsl", "C_coarse_dsl", "D_oracle"]:
        m = summary["modes"].get(mode, {})
        if m:
            print(f"  {mode}: success={m['success_rate']:.1%} "
                  f"parse={m['parse_rate']:.1%} "
                  f"verify={m['verification_rate']:.1%} "
                  f"forbidden={m['total_forbidden_ops']} "
                  f"violations={m['total_precondition_violations']}")


def _run_mock_benchmark(args) -> dict:
    """Run benchmark with mock parser (no API cost)."""
    from vlm_nsm.core.goal_parser import MockGoalParser

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_objects = ["cube1", "cube2", "cube3", "table"]
    verifier_bank = standard_verifier_bank(all_objects)
    coarse_verifier = CoarseOptionVerifier(valid_objects=all_objects)
    mock = MockGoalParser()

    results = []
    for task in BENCHMARK_TASKS:
        tid = task["id"]
        if args.tasks:
            wanted = set(args.tasks.split(","))
            if tid not in wanted:
                continue

        instruction = task["instruction"]
        objects = task["objects"]
        oracle = task["oracle"]

        # Mode A: goal-only via mock
        r_a = TaskResult(task_id=tid, instruction=instruction, mode="A_goal_only")
        try:
            pr = mock.parse(instruction, task_type="stacking")
            r_a.parse_ok = pr.success
            if pr.goal_json:
                report = verifier_bank.verify_all(goal_json=pr.goal_json, task_type="stacking")
                r_a.verification_passed = report.passed
                r_a.success = report.passed
                r_a.n_steps = len(pr.goal_json.get("goal_predicates", []))
                r_a.steps = pr.goal_json.get("goal_predicates", [])
        except Exception as e:
            r_a.error = str(e)
        results.append(r_a)

        # Mode B: phase DSL (same as A — just checking forbidden ops)
        r_b = TaskResult(task_id=tid, instruction=instruction, mode="B_phase_dsl")
        try:
            pr = mock.parse(instruction, task_type="stacking")
            r_b.parse_ok = pr.success
            if pr.goal_json:
                report = verifier_bank.verify_all(goal_json=pr.goal_json, task_type="stacking")
                r_b.verification_passed = report.passed
                r_b.success = report.passed
                r_b.n_steps = len(pr.goal_json.get("goal_predicates", []))
            # Check raw output for forbidden ops (mock returns structured data, not raw)
        except Exception as e:
            r_b.error = str(e)
        results.append(r_b)

        # Mode C: coarse DSL — mock program
        r_c = TaskResult(task_id=tid, instruction=instruction, mode="C_coarse_dsl")
        # Generate a mock coarse program from the oracle (simulating Qwen output)
        coarse_report = coarse_verifier.check(coarse_steps=oracle)
        r_c.verification_passed = coarse_report.passed
        r_c.n_steps = len(oracle)
        r_c.steps = oracle
        r_c.n_step_failures = len(coarse_report.failures)
        r_c.parse_ok = True
        r_c.success = coarse_report.passed
        try:
            program = CoarseTaskProgram.from_json({"steps": oracle})
            if program:
                goals = program.to_goal_predicates()
                if goals:
                    goal_report = verifier_bank.verify_all(
                        goal_json={"goal_predicates": goals}, task_type="stacking")
                    r_c.goal_achieved = goal_report.passed
        except Exception:
            pass
        results.append(r_c)

        # Mode D: oracle
        r_d = TaskResult(task_id=tid, instruction=instruction, mode="D_oracle")
        oracle_report = coarse_verifier.check(coarse_steps=oracle)
        r_d.verification_passed = oracle_report.passed
        r_d.n_steps = len(oracle)
        r_d.steps = oracle
        r_d.parse_ok = True
        r_d.success = oracle_report.passed
        try:
            program = CoarseTaskProgram.from_json({"steps": oracle})
            if program:
                goals = program.to_goal_predicates()
                if goals:
                    goal_report = verifier_bank.verify_all(
                        goal_json={"goal_predicates": goals}, task_type="stacking")
                    r_d.goal_achieved = goal_report.passed
        except Exception:
            pass
        results.append(r_d)

    # Summarize
    by_mode: dict[str, list] = {}
    for r in results:
        by_mode.setdefault(r.mode, []).append(r)

    summary = {"n_tasks": len([t for t in BENCHMARK_TASKS
                               if not args.tasks or t["id"] in args.tasks.split(",")]),
               "n_total_results": len(results), "modes": {}}
    for mode, mrs in sorted(by_mode.items()):
        n = len(mrs)
        ok = [r for r in mrs if r.success]
        summary["modes"][mode] = {
            "n": n,
            "success_rate": len(ok) / n if n else 0,
            "verification_rate": len([r for r in mrs if r.verification_passed]) / n if n else 0,
            "total_forbidden_ops": sum(r.n_forbidden_ops for r in mrs),
            "total_precondition_violations": sum(r.n_precondition_violations for r in mrs),
        }

    # Save
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    detail = []
    for r in results:
        detail.append({
            "task_id": r.task_id, "instruction": r.instruction,
            "mode": r.mode, "success": r.success,
            "parse_ok": r.parse_ok, "verification_passed": r.verification_passed,
            "n_steps": r.n_steps, "n_forbidden_ops": r.n_forbidden_ops,
            "n_precondition_violations": r.n_precondition_violations,
            "n_step_failures": r.n_step_failures,
            "goal_achieved": r.goal_achieved, "error": r.error,
            "steps": r.steps,
        })
    (output_dir / "detailed_results.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False))

    # Print
    print("\n60-Task Mock Benchmark Results:")
    for mode in ["A_goal_only", "B_phase_dsl", "C_coarse_dsl", "D_oracle"]:
        m = summary["modes"].get(mode, {})
        if m:
            print(f"  {mode}: success={m['success_rate']:.1%} "
                  f"verify={m['verification_rate']:.1%} "
                  f"forbidden={m['total_forbidden_ops']} "
                  f"violations={m['total_precondition_violations']}")

    return summary


if __name__ == "__main__":
    main()
