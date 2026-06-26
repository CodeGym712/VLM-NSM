"""TaskComposer — NL + SceneGraph → Coarse TaskProgram.

Generates coarse-option task programs (pick/place/move/ensure/achieve)
from natural language instructions via Qwen API + VerifierBank.

Phase-level skills (reach/grasp/lift/reach_place/release_place) are
FORBIDDEN as upper-layer actions. They exist only inside AtomicPick/AtomicPlace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vlm_nsm.neuro_symbolic.verifier_bank import VerifierBank, VerificationReport
from vlm_nsm.core.predicate_scene_graph import PredicateSceneGraph


# ── Coarse DSL Prompt Template ───────────────────────────────────────────────

COARSE_DSL_SYSTEM_PROMPT = """You are a robot task planner. Convert natural language instructions into a coarse option program.

You output ONLY coarse semantic options:
  - pick(object)          — grasp and lift the object
  - place(object, target) — put the held object on a target (block or table)
  - move(object, source, target) — pick from source and place on target
  - ensure(predicate)     — verify a predicate holds; if not, replan
  - achieve(predicate)    — do whatever is needed to make a predicate true

FORBIDDEN options (phase-level, do NOT output these):
  reach, grasp, lift, reach_place, release_place, retreat

Valid predicates: on(obj, support), ontable(obj), clear(obj), holding(obj), handempty, graspable(obj)

Output JSON schema:
{
  "version": "1.0",
  "steps": [
    {"op": "pick", "object": "cube1"},
    {"op": "place", "object": "cube1", "target": "cube2"}
  ]
}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no code fences.
- Use ONLY the valid object IDs listed in the scene.
- Do NOT output reach/grasp/lift/reach_place/release_place/retreat.
- pick(obj) requires obj to be clear and on the table (or a known position).
- place(obj, target) requires you to be holding obj, and target to be clear.
- place(obj, table) is allowed for putting objects down.
- ensure(predicate) is a verification gate — use before pick to make sure the object is reachable.
- achieve(predicate) is a goal — the executor will plan how to reach it.
- Every object placed on another object creates an on(A, B) predicate.
- No self-place (e.g., place(cube1, cube1)).

Examples:

Scene: cube1 on table, cube2 on table, hand empty
Instruction: "Stack cube1 on cube2"
{"version": "1.0", "steps": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]}

Scene: cube1 on table, cube2 on cube1, hand empty
Instruction: "Move cube2 from cube1 to the table"
{"version": "1.0", "steps": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]}

Scene: cube1 on cube2, cube2 on table, hand empty
Instruction: "Unstack cube1 and put it on the table"
{"version": "1.0", "steps": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]}

Scene: cube1 on table, cube2 on table, cube3 on table, hand empty
Instruction: "Stack cube1 on cube2, then put cube3 on cube1"
{"version": "1.0", "steps": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}, {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube1"}]}
"""


# ── Composition Result ───────────────────────────────────────────────────────

@dataclass
class CompositionResult:
    """Result from TaskComposer.compose()."""
    mode: str = "failed"  # "full" | "goal_only" | "coarse" | "failed"
    task_program: Any = None          # CoarseTaskProgram if mode="coarse"
    goal_json: dict | None = None     # Goal predicates for BFS fallback
    coarse_steps: list | None = None  # Raw coarse steps list
    verification_report: VerificationReport | None = None
    error: str | None = None
    raw_output: str | None = None


class TaskComposer:
    """Compose coarse task programs from NL + scene graph.

    Supports three modes:
      A. goal-only + BFS planner (backward compat)
      B. coarse option program (new, via CoarseTaskProgram)
      C. oracle coarse program (reference, manually specified)

    Args:
        parser: Goal parser with .parse() method (OnlineQwenGoalParser or mock).
        verifier_bank: VerifierBank for output verification.
        enable_coarse: Whether to generate coarse DSL programs.
    """

    def __init__(self, parser, verifier_bank: VerifierBank,
                 enable_coarse: bool = True):
        self.parser = parser
        self.verifier_bank = verifier_bank
        self.enable_coarse = enable_coarse

    def compose(self, instruction: str, scene_graph: PredicateSceneGraph,
                objects: list[str],
                oracle_coarse_steps: list | None = None) -> CompositionResult:
        """Compose task from NL instruction.

        Priority:
          1. If oracle_coarse_steps provided, use directly (mode="coarse").
          2. If enable_coarse, try to generate coarse program via parser.
          3. Fall back to goal-only + BFS planner.

        Returns CompositionResult with mode indicating result level.
        """
        # ── Mode D: Oracle ──
        if oracle_coarse_steps is not None:
            return self._verify_coarse(oracle_coarse_steps, objects,
                                       instruction, is_oracle=True)

        # ── Mode C: Coarse DSL via Qwen ──
        if self.enable_coarse:
            result = self._try_coarse_compose(instruction, scene_graph, objects)
            if result and result.mode == "coarse":
                return result

        # ── Mode A: Goal-only fallback ──
        return self._compose_goal_only(instruction, scene_graph, objects)

    def _try_coarse_compose(self, instruction: str,
                            scene_graph: PredicateSceneGraph,
                            objects: list[str]) -> CompositionResult | None:
        """Try to get a coarse program from the parser."""
        try:
            parse_result = self.parser.parse(
                instruction, task_type="coarse_stacking",
                objects=objects,
                n_blocks=len([o for o in objects if o != "table"]),
                scene_description=self._build_scene_description(scene_graph, objects))
        except Exception as e:
            try:
                parse_result = self.parser.parse(instruction)
            except Exception as e2:
                return None

        if not parse_result.success or not parse_result.goal_json:
            return None

        data = parse_result.goal_json
        steps = data.get("steps", [])
        if not steps:
            # Maybe it's a goal-only output — don't force it
            return None

        return self._verify_coarse(steps, objects, instruction,
                                   is_oracle=False,
                                   raw_output=parse_result.raw_output)

    def _verify_coarse(self, steps: list, objects: list[str],
                       instruction: str, is_oracle: bool = False,
                       raw_output: str | None = None) -> CompositionResult:
        """Verify coarse steps through CoarseOptionVerifier."""
        valid_objects_set = set(objects)

        # Static check first
        from vlm_nsm.neuro_symbolic.verifier_bank import CoarseOptionVerifier
        static_ok, static_errors = CoarseOptionVerifier.check_static(
            steps, valid_objects_set)

        if not static_ok:
            # Try to build CoarseTaskProgram — from_json validates per-step
            from vlm_nsm.language.coarse_dsl import CoarseTaskProgram
            program = CoarseTaskProgram.from_json({"steps": steps})
            if program is None:
                return CompositionResult(
                    mode="failed",
                    error=f"static_check_failed: {'; '.join(static_errors)}",
                    coarse_steps=steps,
                    raw_output=raw_output)

        # Full verification
        report = self.verifier_bank.verify_all(
            coarse_steps=steps,
            task_type="stacking")

        coarse_verifier = self.verifier_bank.get("CoarseOptionVerifier")
        if coarse_verifier:
            coarse_report = coarse_verifier.check(
                coarse_steps=steps)
            report.merge(coarse_report)

        if report.passed:
            mode = "coarse"
        else:
            # Check if failures are blocking
            has_forbidden = any("forbidden" in f[0] for f in report.failures)
            has_blocking = any(
                "pick" in f[0] or "place" in f[0] or "move" in f[0]
                for f in report.failures)
            if has_forbidden or has_blocking:
                return CompositionResult(
                    mode="failed",
                    verification_report=report,
                    coarse_steps=steps,
                    error=f"coarse_verification_failed: {report.explain()}",
                    raw_output=raw_output)
            mode = "coarse"  # warnings only

        # Build CoarseTaskProgram
        from vlm_nsm.language.coarse_dsl import CoarseTaskProgram
        program = CoarseTaskProgram.from_json({"steps": steps})

        # Extract goal predicates for BFS
        goal_preds = program.to_goal_predicates() if program else []

        return CompositionResult(
            mode=mode,
            task_program=program,
            goal_json={"goal_predicates": goal_preds} if goal_preds else None,
            coarse_steps=steps,
            verification_report=report,
            raw_output=raw_output)

    def _compose_goal_only(self, instruction: str,
                           scene_graph: PredicateSceneGraph,
                           objects: list[str]) -> CompositionResult:
        """Goal-only fallback using existing goal parser."""
        try:
            parse_result = self.parser.parse(
                instruction, task_type="stacking",
                objects=objects,
                n_blocks=len([o for o in objects if o != "table"]))
        except Exception:
            try:
                parse_result = self.parser.parse(instruction)
            except Exception as e:
                return CompositionResult(mode="failed", error=f"parse: {e}")

        if not parse_result.success:
            return CompositionResult(mode="failed", error=parse_result.error)

        goal_json = parse_result.goal_json or {}
        goal_preds = goal_json.get("goal_predicates", [])

        if not goal_preds:
            return CompositionResult(mode="failed", error="no goal_predicates")

        report = self.verifier_bank.verify_all(
            goal_json={"goal_predicates": goal_preds}, task_type="stacking")

        if report.passed:
            return CompositionResult(mode="full", goal_json=goal_json,
                                     verification_report=report)
        elif goal_preds:
            return CompositionResult(mode="goal_only", goal_json=goal_json,
                                     verification_report=report)
        return CompositionResult(mode="failed", goal_json=goal_json,
                                 verification_report=report,
                                 error="verification failed, no fallback")

    @staticmethod
    def _build_scene_description(scene_graph: PredicateSceneGraph,
                                  objects: list[str]) -> str:
        """Build a textual scene description from the predicate scene graph."""
        if scene_graph is None:
            return "No scene information available."
        lines = []
        if hasattr(scene_graph, 'predicates') and scene_graph.predicates:
            lines.append("Current scene predicates:")
            for p in scene_graph.predicates:
                p_list = [str(a) for a in p]
                lines.append(f"  {p_list[0]}({', '.join(p_list[1:])})")
        lines.append(f"Available objects: {objects}")
        lines.append(f"Valid object IDs: {[o for o in objects if o != 'table']}")
        return "\n".join(lines)
