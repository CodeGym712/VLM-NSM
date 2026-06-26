"""Coarse Planner Bridge — maps coarse TaskProgram steps to BFS symbolic Actions.

Connects the upper neuro-symbolic coarse DSL (pick/place/move/ensure/achieve)
to the BFS SymbolicPlanner. Phase-level skills (reach/grasp/lift/...) are
NOT exposed through this bridge — they're internal to AtomicPick/AtomicPlace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundedStep:
    """A coarse step grounded into executable actions.

    For pick/place: maps to symbolic move Action that BFS planner can use.
    For ensure/achieve: may decompose into sub-goals.
    For move: maps to pick+place pair.
    """
    op: str = ""
    object: str = ""
    target: str = ""
    source: str = ""
    symbolic_actions: list = field(default_factory=list)
    # For achieve: the parsed goal predicate
    goal_predicate: list | None = None
    # Whether this step requires BFS planning
    requires_planning: bool = False


class CoarsePlannerBridge:
    """Bridge between coarse TaskProgram and BFS SymbolicPlanner.

    Responsibilities:
      1. Validate coarse steps are physically meaningful.
      2. Map pick/place to symbolic move actions.
      3. Decompose ensure/achieve into goal predicates for BFS.
      4. Convert CoarseTaskProgram → list[SymbolicAction] for execution.
    """

    def __init__(self, symbolic_planner=None):
        """
        Args:
            symbolic_planner: BFS SymbolicPlanner instance (optional, can be set later).
        """
        self.planner = symbolic_planner

    def ground(self, coarse_steps: list,
               objects: list[str],
               initial_predicates: list[list[str]] | None = None) -> list[GroundedStep]:
        """Ground coarse steps into GroundedStep objects.

        Args:
            coarse_steps: List of dicts or CoarseStep objects.
            objects: Valid object IDs.
            initial_predicates: Optional initial state predicates.

        Returns:
            List of GroundedStep objects ready for execution planning.
        """
        grounded = []

        for i, step in enumerate(coarse_steps):
            if hasattr(step, 'op'):
                op, obj = step.op, step.object
                tgt = getattr(step, 'target', '')
                src = getattr(step, 'source', '')
                pred = getattr(step, 'predicate', [])
            elif isinstance(step, dict):
                op = step.get("op", "")
                obj = step.get("object", "")
                tgt = step.get("target", "")
                src = step.get("source", "")
                pred = step.get("predicate", [])
            else:
                grounded.append(GroundedStep(op="unknown"))
                continue

            gs = GroundedStep(op=op, object=obj, target=tgt, source=src)

            if op == "pick":
                # pick(obj) → symbolic move(obj, current_location, hand)
                # The BFS planner handles this internally
                gs.requires_planning = True

            elif op == "place":
                # place(obj, target) → symbolic move(obj, hand, target)
                gs.symbolic_actions = [("place", obj, tgt)]
                gs.goal_predicate = ["on", obj, tgt] if tgt != "table" else ["ontable", obj]

            elif op == "move":
                # move(obj, source, target) → pick + place
                gs.symbolic_actions = [
                    ("pick_from", obj, src),
                    ("place_on", obj, tgt),
                ]
                gs.goal_predicate = ["on", obj, tgt] if tgt != "table" else ["ontable", obj]
                gs.requires_planning = True  # May need intermediate steps

            elif op == "ensure":
                # ensure(predicate) → verification gate, pass through
                if pred:
                    gs.goal_predicate = [str(a) for a in pred]
                gs.requires_planning = False

            elif op == "achieve":
                # achieve(predicate) → goal for BFS planner
                if pred:
                    gs.goal_predicate = [str(a) for a in pred]
                gs.requires_planning = True

            grounded.append(gs)

        return grounded

    def to_symbolic_actions(self, coarse_steps: list,
                            objects: list[str]) -> list:
        """Convert coarse steps to symbolic actions for BFS planner.

        NOTE: This is for LOGGING/TRACE only. The BFS planner uses its own
        MoveOn/MoveToTable operators internally via PlannerAdapter("bfs").solve().
        Do NOT feed these dicts directly to the BFS planner.

        Maps:
          pick(obj)   → move(obj, ?, hand)  [source inferred from context]
          place(obj,t)→ move(obj, hand, t)
          move(obj,s,t)→ move(obj, s, t)
          ensure/achieve → pass (not actions)

        Args:
            coarse_steps: Coarse step list (dicts or CoarseStep objects).
            objects: Valid object IDs.

        Returns:
            List of dicts for logging/trace, NOT for planner input.
        """
        actions = []
        holding = None

        for step in coarse_steps:
            if hasattr(step, 'op'):
                op, obj = step.op, step.object
                tgt = getattr(step, 'target', '')
                src = getattr(step, 'source', '')
            elif isinstance(step, dict):
                op = step.get("op", "")
                obj = step.get("object", "")
                tgt = step.get("target", "")
                src = step.get("source", "")
            else:
                continue

            if op == "pick":
                # Source is inferred: if holding is None, object was wherever it was.
                # The BFS planner handles this; this is just for trace logging.
                src = src or "?"
                actions.append({"name": "pick", "args": [obj, src], "note": "source inferred"})
                holding = obj

            elif op == "place":
                actions.append({"name": "place", "args": [obj, tgt]})
                holding = None

            elif op == "move":
                actions.append({"name": "move", "args": [obj, src or "?", tgt]})

            elif op in ("ensure", "achieve"):
                # Verification/planning gates, not direct actions
                pass

        return actions

    def decompose_achieve(self, predicate: list,
                          objects: list[str]) -> list:
        """Decompose an achieve(predicate) step using BFS planning.

        Uses PlannerAdapter("bfs").solve() which builds SymbolicSceneGraph
        and SymbolicGoal internally, then converts results back to Action objects.

        Args:
            predicate: Goal predicate e.g. ["on", "cube1", "cube2"].
            objects: Available object IDs (e.g. ["cube1", "cube2", "cube3"]).

        Returns:
            List of Action objects from BFS planner, or empty list if unsolvable.
        """
        if not predicate or len(predicate) < 2:
            return []

        cubes = [o for o in objects if o != "table"]
        if not cubes:
            return []

        # Extract target stack ordering from predicate
        # For ["on", "cube1", "cube2"]: target_stack = ["cube2", "cube1"] (bottom→top)
        target_stack = []
        pred_name = str(predicate[0]).lower()
        if pred_name == "on" and len(predicate) >= 3:
            top, bottom = predicate[1], predicate[2]
            if bottom != "table":
                target_stack = [str(bottom), str(top)]
            else:
                target_stack = [str(top)]
        elif pred_name == "clear" and len(predicate) >= 2:
            # achieve clear(X) → move anything on X elsewhere
            target_stack = []  # No stacking goal, just need to clear
        elif pred_name == "ontable" and len(predicate) >= 2:
            target_stack = [str(predicate[1])]
        else:
            return []

        try:
            from vlm_nsm.core.planner_adapter import PlannerAdapter
            planner = PlannerAdapter("bfs")
            result = planner.solve(cubes=cubes, target_stack=target_stack)
            if result.success:
                return list(result.actions)
        except Exception:
            pass

        return []

    def execution_sequence(self, coarse_steps: list,
                           objects: list[str]) -> list[dict]:
        """Generate the full execution sequence from coarse steps.

        Output format (consumed by embodied executor):
          {"skill": "atomic_pick", "object": "cube1"}
          {"skill": "atomic_place", "object": "cube1", "target": "cube2"}

        ensure(predicate) → NO-OP (pure verification gate, filtered out).
        achieve(predicate) → NO-OP (must be decomposed by BFS before this point).

        Returns ONLY atomic_pick / atomic_place skill dicts.
        No reach/grasp/lift/reach_place/release_place.

        Args:
            coarse_steps: Coarse step list.
            objects: Valid object IDs.

        Returns:
            List of executable skill dicts (atomic_pick / atomic_place only).
        """
        sequence = []
        for step in coarse_steps:
            if hasattr(step, 'op'):
                op, obj = step.op, step.object
                tgt = getattr(step, 'target', '')
                pred = getattr(step, 'predicate', [])
            elif isinstance(step, dict):
                op = step.get("op", "")
                obj = step.get("object", "")
                tgt = step.get("target", "")
                pred = step.get("predicate", [])
            else:
                continue

            if op == "pick":
                sequence.append({"skill": "atomic_pick", "object": obj})
            elif op == "place":
                sequence.append({"skill": "atomic_place", "object": obj, "target": tgt})
            elif op == "move":
                sequence.append({"skill": "atomic_pick", "object": obj})
                sequence.append({"skill": "atomic_place", "object": obj, "target": tgt})
            elif op == "ensure":
                # NO-OP: ensure is a verification gate — nothing to execute.
                # The verifier already checked the predicate before we reach here.
                pass
            elif op == "achieve":
                # NO-OP: achieve must be decomposed by BFS planner before
                # reaching execution_sequence. If we see it here, it's an error
                # in the upstream pipeline.
                pass

        return sequence
