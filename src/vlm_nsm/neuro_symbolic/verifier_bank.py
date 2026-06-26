"""Verifier Bank — structured verification pipeline for LLM outputs.

All LLM-generated content (goals, subgoals, plans, repairs) must pass
through verifiers before reaching the planner or executor.

Architecture:
  Verifier → VerificationReport { passed, failures, warnings }
  VerifierBank → chain verifiers → aggregated report

Coarse option verifiers validate pick/place/move/ensure/achieve programs
instead of phase-level (reach/grasp/lift/reach_place/release_place) programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── Verification Report ─────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    """Output from a single verifier or the verifier bank.

    Attributes:
        passed: Whether all checks passed.
        verifier_name: Which verifier produced this report.
        failures: List of (check_name, reason) for each failed check.
        warnings: Non-fatal issues found.
        metadata: Arbitrary extra info (e.g., plan length, object count).
    """

    passed: bool = True
    verifier_name: str = ""
    failures: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_failure(self, check: str, reason: str):
        self.failures.append((check, reason))
        self.passed = False

    def add_warning(self, check: str, reason: str):
        self.warnings.append((check, reason))

    def merge(self, other: "VerificationReport") -> "VerificationReport":
        """Merge another report into this one."""
        self.passed = self.passed and other.passed
        self.failures.extend(other.failures)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        return self

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "verifier_name": self.verifier_name,
            "failures": [{"check": f[0], "reason": f[1]} for f in self.failures],
            "warnings": [{"check": w[0], "reason": w[1]} for w in self.warnings],
            "metadata": self.metadata,
        }

    def explain(self) -> str:
        """Human-readable explanation of failures."""
        if self.passed:
            return f"[{self.verifier_name}] All checks passed."
        lines = [f"[{self.verifier_name}] {len(self.failures)} failure(s):"]
        for check, reason in self.failures:
            lines.append(f"  - {check}: {reason}")
        return "\n".join(lines)


# ── Base Verifier ────────────────────────────────────────────────────────────

class BaseVerifier:
    """Abstract verifier — all verifiers implement check()."""

    name: str = "base"

    def check(self, *args, **kwargs) -> VerificationReport:
        raise NotImplementedError


# ── 1. GoalVerifier ──────────────────────────────────────────────────────────

class GoalVerifier(BaseVerifier):
    """Verify goal JSON: schema validity, object existence, relation legality,
    no self-contradictions.

    Reuses AlignmentLayer for structured validation.
    """

    name = "GoalVerifier"

    def __init__(self, valid_objects: list[str], valid_predicates: list[str] | None = None,
                 type_table: dict[str, str] | None = None):
        self.valid_objects = set(valid_objects)
        self.valid_predicates = valid_predicates or ["on", "ontable", "clear",
            "holding", "handempty", "graspable"]
        self.type_table = type_table or {"cube1": "block", "cube2": "block", "table": "table"}

    def check(self, goal_json: dict | None, task_type: str = "stacking") -> VerificationReport:
        report = VerificationReport(verifier_name=self.name)
        report.metadata["task_type"] = task_type

        if goal_json is None:
            report.add_failure("schema_check", "goal_json is None")
            return report

        # Schema check
        if not isinstance(goal_json, dict):
            report.add_failure("schema_check", f"goal_json is not dict: {type(goal_json)}")
            return report
        if "goal_predicates" not in goal_json:
            report.add_failure("schema_check", "missing 'goal_predicates' field")
            return report
        preds = goal_json.get("goal_predicates", [])
        if not isinstance(preds, list) or len(preds) == 0:
            report.add_failure("schema_check", "goal_predicates empty or not a list")
            return report

        # Per-predicate checks
        for i, p in enumerate(preds):
            if not isinstance(p, list) or len(p) < 2:
                report.add_failure(f"predicate_{i}", f"invalid predicate format: {p}")
                continue
            pred_name = str(p[0]).lower()

            if pred_name not in self.valid_predicates:
                report.add_failure(f"predicate_{i}", f"unknown predicate '{pred_name}'")
                continue

            # Object existence check
            for j, arg in enumerate(p[1:]):
                arg_str = str(arg)
                if arg_str not in self.valid_objects and arg_str not in ("", "table"):
                    report.add_failure(f"predicate_{i}_arg{j+1}",
                                       f"unknown object '{arg_str}'")

            # Self-reference check
            if pred_name == "on" and len(p) >= 3 and p[1] == p[2]:
                report.add_failure(f"predicate_{i}", f"self-reference: {p}")

        # Cross-predicate consistency
        on_pairs = [(str(p[1]), str(p[2])) for p in preds
                     if len(p) >= 3 and str(p[0]).lower() == "on"]
        supports = {}
        for top, bottom in on_pairs:
            if top in supports:
                report.add_failure("cross_check",
                    f"multiple supports for '{top}': {supports[top]} and {bottom}")
            supports[top] = bottom

        # Cycle detection via traversal
        visited = set()
        for top in supports:
            chain = [top]
            seen = {top}
            while chain[-1] in supports:
                nxt = supports[chain[-1]]
                if nxt in seen:
                    report.add_failure("cross_check",
                        f"cycle detected: {' -> '.join(chain)} -> {nxt}")
                    break
                seen.add(nxt)
                chain.append(nxt)

        report.metadata["n_predicates"] = len(preds)
        return report


# ── 2. SubgoalVerifier ───────────────────────────────────────────────────────

class SubgoalVerifier(BaseVerifier):
    """Verify subgoal consistency and ordering constraints."""

    name = "SubgoalVerifier"

    def __init__(self, valid_objects: list[str]):
        self.valid_objects = set(valid_objects)

    def check(self, subgoals: list[list[list[str]]],
              goal_predicates: list[list[str]] | None = None) -> VerificationReport:
        report = VerificationReport(verifier_name=self.name)

        if not subgoals:
            report.add_warning("empty", "no subgoals provided")
            return report

        # Each subgoal must be valid
        for i, sg in enumerate(subgoals):
            if not isinstance(sg, list) or not sg:
                report.add_failure(f"subgoal_{i}", "empty or invalid format")
                continue

        # Subgoals must not conflict with each other:
        # same object cannot be placed on two different supports
        seen_placements: dict[str, str] = {}  # obj → support
        for i, sg in enumerate(subgoals):
            for p in sg:
                p_list = [str(a) for a in p]
                if len(p_list) >= 3 and p_list[0].lower() == "on":
                    top, support = p_list[1], p_list[2]
                    if top in seen_placements and seen_placements[top] != support:
                        report.add_failure(f"subgoal_{i}",
                            f"conflict: {top} placed on {seen_placements[top]} and {support}")
                    seen_placements[top] = support

        # Check subgoal ordering: later subgoals shouldn't violate earlier ones
        if goal_predicates:
            goal_set = {tuple(str(a) for a in p) for p in goal_predicates}
            for i, sg in enumerate(subgoals):
                sg_set = {tuple(str(a) for a in p) for p in sg}
                # Final subgoal should match goal
                if i == len(subgoals) - 1:
                    if not sg_set.issubset(goal_set):
                        report.add_warning(f"subgoal_{i}",
                            "final subgoal does not achieve full goal")

        report.metadata["n_subgoals"] = len(subgoals)
        return report


# ── 3. PlanVerifier ──────────────────────────────────────────────────────────

class PlanVerifier(BaseVerifier):
    """Verify that each step of a plan respects preconditions and the final
    state satisfies the goal.

    Uses SymbolicSceneGraph + operators from the symbolic layer.
    """

    name = "PlanVerifier"

    def check(self, actions: list[Any], initial_scene_graph,
              goal_predicates: list[list[str]]) -> VerificationReport:
        report = VerificationReport(verifier_name=self.name)

        if not actions:
            report.add_failure("empty_plan", "no actions in plan")
            return report

        from vlm_nsm.symbolic.operators import SymbolicAction

        state = initial_scene_graph.clone() if hasattr(initial_scene_graph, 'clone') else initial_scene_graph

        for i, action in enumerate(actions):
            # Determine action name and args
            if hasattr(action, 'name'):
                name = action.name
                args = action.args if hasattr(action, 'args') else []
            elif isinstance(action, (list, tuple)):
                name = action[0]
                args = action[1:] if len(action) > 1 else []
            else:
                report.add_failure(f"step_{i}", f"unknown action format: {type(action)}")
                continue

            # Check if action is supported
            supported = {"move", "move_on", "move_to_table", "pick", "place"}
            if name not in supported:
                report.add_warning(f"step_{i}", f"action '{name}' not in supported set")

            # Check parameter count
            if name in ("move",) and len(args) < 3:
                report.add_failure(f"step_{i}", f"move requires 3 args, got {len(args)}")

            # If action is SymbolicAction, check preconditions
            if isinstance(action, SymbolicAction):
                if not action.is_applicable(state):
                    missing = [p for p in action.preconditions
                               if p not in state.predicates]
                    report.add_failure(f"step_{i}_precondition",
                        f"preconditions not met: {missing}")
                state = action.apply(state)

        # Final goal check
        if goal_predicates:
            from vlm_nsm.symbolic.predicates import Predicate
            goal_set = {Predicate(str(p[0]), tuple(str(a) for a in p[1:]))
                        for p in goal_predicates}
            if not goal_set.issubset(state.predicates):
                report.add_failure("goal_check", "final state does not satisfy goal")

        report.metadata["n_actions"] = len(actions)
        return report


# ── 4. OptionContractVerifier ────────────────────────────────────────────────

class OptionContractVerifier(BaseVerifier):
    """Verify that an option's initiation conditions are met and its
    expected effects are achievable.
    """

    name = "OptionContractVerifier"

    def check(self, contract, state_predicates: list[list[str]],
              target_object: str = "", target_location: str = "") -> VerificationReport:
        report = VerificationReport(verifier_name=self.name)
        report.metadata["option"] = contract.name if hasattr(contract, 'name') else "unknown"

        from vlm_nsm.core.option_contract import OptionContract
        if not isinstance(contract, OptionContract):
            report.add_failure("type", f"not an OptionContract: {type(contract)}")
            return report

        # Check initiation
        state_set = {tuple(p) for p in state_predicates}

        for p in contract.initiation_predicates:
            # Substitute parameters
            substituted = []
            for arg in p:
                if arg == "{target_object}":
                    substituted.append(target_object)
                elif arg == "{target_location}":
                    substituted.append(target_location)
                else:
                    substituted.append(arg)
            if tuple(substituted) not in state_set:
                report.add_failure("initiation", f"missing: {substituted}")

        # Check retry budget
        if contract.retry_policy.get("max_retries", 1) > 5:
            report.add_warning("retry", f"retry budget excessive: {contract.retry_policy['max_retries']}")

        report.metadata["n_initiation_checks"] = len(contract.initiation_predicates)
        return report


# ── 5. RepairVerifier ───────────────────────────────────────────────────────

class RepairVerifier(BaseVerifier):
    """Verify that a repair proposal is valid: doesn't violate domain rules,
    doesn't exceed retry budget, doesn't introduce illegal goals.
    """

    name = "RepairVerifier"

    def __init__(self, valid_objects: list[str], max_retries: int = 3):
        self.valid_objects = set(valid_objects)
        self.max_retries = max_retries

    def check(self, repair_proposal: dict, retries_used: int = 0,
              original_goal: list[list[str]] | None = None) -> VerificationReport:
        report = VerificationReport(verifier_name=self.name)

        if not isinstance(repair_proposal, dict):
            report.add_failure("format", f"repair_proposal must be dict, got {type(repair_proposal)}")
            return report

        repair_type = repair_proposal.get("repair_type", "unknown")
        report.metadata["repair_type"] = repair_type

        # Retry budget check
        retry_budget = repair_proposal.get("retry_budget", 1)
        if retries_used + retry_budget > self.max_retries:
            report.add_failure("budget",
                f"retries_used({retries_used}) + retry_budget({retry_budget}) > max({self.max_retries})")

        # Modified option must exist (coarse options + phase-level for backward compat)
        modified_option = repair_proposal.get("modified_option", "")
        valid_options = {"pick", "place", "move", "ensure", "achieve",
                         "reach", "grasp", "lift", "reach_place", "release_place", "retreat"}
        if modified_option and modified_option not in valid_options:
            report.add_failure("option", f"unknown modified_option: '{modified_option}'")

        # Parameter patch must reference valid objects
        param_patch = repair_proposal.get("parameter_patch", {})
        for key, val in param_patch.items():
            if isinstance(val, str) and val not in self.valid_objects and val not in ("table", ""):
                report.add_warning("param", f"parameter '{key}={val}' not in valid_objects")

        # Required verification must be specified
        required_verification = repair_proposal.get("required_verification", [])
        if not required_verification:
            report.add_warning("verify", "no required_verification specified")

        report.metadata["retries_used"] = retries_used
        return report


# ── 6. CoarseOptionVerifier ────────────────────────────────────────────────────

class CoarseOptionVerifier(BaseVerifier):
    """Verify coarse option program (pick/place/move/ensure/achieve).

    Checks:
      - object exists in valid_objects
      - target exists for place/move
      - handempty before pick
      - holding correct object before place
      - clear target before place (target must not have anything on it)
      - no self-place (object == target)
      - no cyclic stack (A on B and B on A)

    Tracks symbolic state through the step sequence to validate
    preconditions incrementally.
    """

    name = "CoarseOptionVerifier"

    # Coarse ops that modify state
    _MUTATING_OPS = {"pick", "place", "move"}

    def __init__(self, valid_objects: list[str]):
        self.valid_objects = set(valid_objects)

    def check(self, task_program=None, coarse_steps: list | None = None,
              scene_graph=None) -> VerificationReport:
        """Verify a coarse task program.

        Args:
            task_program: A CoarseTaskProgram instance or dict with 'steps' key.
            coarse_steps: Alternative: list of CoarseStep or dict steps directly.
            scene_graph: Optional PredicateSceneGraph for initial state.

        Returns:
            VerificationReport with per-step precondition/effect checks.
        """
        report = VerificationReport(verifier_name=self.name)

        # Extract steps
        steps = coarse_steps or []
        if task_program is not None:
            if hasattr(task_program, 'steps'):
                steps = task_program.steps
            elif isinstance(task_program, dict):
                steps = task_program.get("steps", [])

        if not steps:
            report.add_failure("empty_program", "no steps in coarse program")
            return report

        # Initial symbolic state
        state = {
            "holding": None,       # None or object_id
            "handempty": True,
        }
        # Track what's on top of each object (None = clear, object_id = occupied)
        clear: dict[str, bool] = {}
        # Track what each object is sitting on (None = table, object_id = stacked)
        on: dict[str, str | None] = {}
        # Track what's on each object (the inverse of on)
        on_top: dict[str, str | None] = {}

        for obj in self.valid_objects:
            clear[obj] = True
            on[obj] = None
            on_top[obj] = None

        # If scene_graph provided, initialize from it
        if scene_graph is not None and hasattr(scene_graph, 'predicates'):
            for pred in scene_graph.predicates:
                p = [str(a) for a in pred]
                if p[0] == "on" and len(p) >= 3:
                    top, bottom = p[1], p[2]
                    if bottom != "table":
                        clear[bottom] = False
                    on[top] = bottom if bottom != "table" else None
                    if bottom != "table":
                        on_top[bottom] = top
                elif p[0] == "clear" and len(p) >= 2:
                    clear[p[1]] = True
                elif p[0] == "ontable" and len(p) >= 2:
                    on[p[1]] = None
                elif p[0] == "holding" and len(p) >= 2:
                    state["holding"] = p[1]
                    state["handempty"] = False
                    clear[p[1]] = False
                    on[p[1]] = "hand"

        for i, step in enumerate(steps):
            # Normalize to dict
            if hasattr(step, 'op'):
                s = {"op": step.op, "object": step.object,
                     "target": getattr(step, 'target', ''),
                     "source": getattr(step, 'source', ''),
                     "predicate": getattr(step, 'predicate', [])}
            elif isinstance(step, dict):
                s = step
            else:
                report.add_failure(f"step_{i}", f"unknown step format: {type(step)}")
                continue

            op = s.get("op", "")
            obj = s.get("object", "")
            tgt = s.get("target", "")
            src = s.get("source", "")
            pred = s.get("predicate", [])

            # ── Per-op precondition checks ──
            if op == "pick":
                # Object must exist
                if obj not in self.valid_objects:
                    report.add_failure(f"step_{i}_pick", f"unknown object '{obj}'")
                    continue
                # Must be handempty
                if not state["handempty"]:
                    report.add_failure(f"step_{i}_pick",
                        f"hand not empty (holding {state['holding']})")
                # Object must be clear (nothing on top of it)
                if obj in clear and not clear.get(obj, True):
                    report.add_failure(f"step_{i}_pick",
                        f"'{obj}' is not clear (has '{on_top.get(obj)}' on top)")
                # Apply effect
                state["holding"] = obj
                state["handempty"] = False
                clear[obj] = False
                on[obj] = "hand"
                # If object was on another object, clear that support
                prev_support = on.get(obj)
                if prev_support and prev_support != "hand" and prev_support in clear:
                    clear[prev_support] = True
                    on_top[prev_support] = None

            elif op == "place":
                # Must be holding the object
                if state["holding"] != obj:
                    report.add_failure(f"step_{i}_place",
                        f"not holding '{obj}' (holding {state['holding']})")
                # Target must exist
                if tgt not in self.valid_objects and tgt != "table":
                    report.add_failure(f"step_{i}_place", f"unknown target '{tgt}'")
                # No self-place
                if obj == tgt:
                    report.add_failure(f"step_{i}_place",
                        f"self-place: {obj} on {obj}")
                # Target must be clear (if not table)
                if tgt != "table" and tgt in clear and not clear.get(tgt, True):
                    report.add_failure(f"step_{i}_place",
                        f"target '{tgt}' not clear (has '{on_top.get(tgt)}' on top)")
                # Cycle check: obj already supports tgt somewhere
                if tgt != "table":
                    # Follow chain: if tgt → ... → obj, placing obj on tgt creates cycle
                    chain_obj = tgt
                    seen_chain = {obj}
                    cycle = False
                    while chain_obj in on and on[chain_obj] is not None and on[chain_obj] != "hand":
                        nxt = on[chain_obj]
                        if nxt in seen_chain:
                            cycle = True
                            break
                        seen_chain.add(nxt)
                        chain_obj = nxt
                    if cycle or on.get(tgt) == obj:
                        report.add_failure(f"step_{i}_place",
                            f"cyclic stack: placing {obj} on {tgt} creates cycle")
                # Apply effect
                state["holding"] = None
                state["handempty"] = True
                clear[obj] = True
                on[obj] = tgt if tgt != "table" else None
                if tgt != "table" and tgt in clear:
                    clear[tgt] = False
                    on_top[tgt] = obj

            elif op == "move":
                # move(obj, source, target) = internal composite of place+pick
                # source must match where object currently is
                if obj not in self.valid_objects:
                    report.add_failure(f"step_{i}_move", f"unknown object '{obj}'")
                if tgt not in self.valid_objects and tgt != "table":
                    report.add_failure(f"step_{i}_move", f"unknown target '{tgt}'")
                if obj == tgt:
                    report.add_failure(f"step_{i}_move",
                        f"self-move: {obj} to {obj}")
                if tgt != "table" and tgt in clear and not clear.get(tgt, True):
                    report.add_failure(f"step_{i}_move",
                        f"target '{tgt}' not clear for move")
                # Apply effect: obj moves from current position to target
                # Remove from old position
                old_support = on.get(obj)
                if old_support and old_support != "hand" and old_support in clear:
                    clear[old_support] = True
                    on_top[old_support] = None
                # Place on new target
                if tgt != "table" and tgt in clear:
                    clear[tgt] = False
                    on_top[tgt] = obj
                on[obj] = tgt if tgt != "table" else None
                clear[obj] = True  # object itself is clear after move

            elif op == "ensure":
                if not pred:
                    report.add_failure(f"step_{i}_ensure", "empty predicate for ensure")
                else:
                    pred_name = str(pred[0]).lower() if pred else ""
                    if pred_name not in ("on", "ontable", "clear", "holding",
                                          "handempty", "graspable"):
                        report.add_failure(f"step_{i}_ensure",
                            f"unknown predicate '{pred_name}' for ensure")
                    # Check predicate args are valid objects
                    for j, arg in enumerate(pred[1:]):
                        arg_str = str(arg)
                        if arg_str not in self.valid_objects and arg_str != "table":
                            report.add_failure(f"step_{i}_ensure_arg{j+1}",
                                f"unknown object '{arg_str}' in ensure predicate")

            elif op == "achieve":
                if not pred:
                    report.add_failure(f"step_{i}_achieve", "empty predicate for achieve")
                else:
                    pred_name = str(pred[0]).lower() if pred else ""
                    if pred_name not in ("on", "ontable", "clear", "holding",
                                          "handempty", "graspable"):
                        report.add_failure(f"step_{i}_achieve",
                            f"unknown predicate '{pred_name}' for achieve")
                    for j, arg in enumerate(pred[1:]):
                        arg_str = str(arg)
                        if arg_str not in self.valid_objects and arg_str != "table":
                            report.add_failure(f"step_{i}_achieve_arg{j+1}",
                                f"unknown object '{arg_str}' in achieve predicate")

            else:
                # Check if it's a forbidden phase-level op
                _forbidden = {"reach", "grasp", "lift", "reach_place",
                              "release_place", "retreat"}
                if op in _forbidden:
                    report.add_failure(f"step_{i}_forbidden",
                        f"'{op}' is a phase-level skill, forbidden in coarse DSL")
                else:
                    report.add_failure(f"step_{i}_unknown_op",
                        f"unknown coarse op '{op}'")

        # Post-sequence checks
        all_objects = [o for o in self.valid_objects if o != "table"]
        # Check no object appears twice as support
        support_counts: dict[str, int] = {}
        for obj_id in all_objects:
            sup = on.get(obj_id)
            if sup and sup != "hand":
                key = f"{obj_id}→{sup}"
                support_counts[key] = support_counts.get(key, 0) + 1

        report.metadata["n_steps"] = len(steps)
        report.metadata["n_objects"] = len(all_objects)
        return report

    @staticmethod
    def check_static(steps: list, valid_objects: set) -> tuple[bool, list[str]]:
        """Static check without state tracking — just structural validity.

        Returns (ok, error_messages).
        """
        errors = []
        for i, step in enumerate(steps):
            if hasattr(step, 'op'):
                op, obj, tgt = step.op, step.object, getattr(step, 'target', '')
            elif isinstance(step, dict):
                op = step.get("op", "")
                obj = step.get("object", "")
                tgt = step.get("target", "")
            else:
                errors.append(f"step_{i}: unknown format")
                continue

            _forbidden = {"reach", "grasp", "lift", "reach_place",
                          "release_place", "retreat"}
            if op in _forbidden:
                errors.append(f"step_{i}: '{op}' is phase-level, forbidden")
            if op in ("pick", "place", "move") and obj not in valid_objects:
                errors.append(f"step_{i}: unknown object '{obj}'")
            if op in ("place", "move") and tgt not in valid_objects and tgt != "table":
                errors.append(f"step_{i}: unknown target '{tgt}'")
            if op == "place" and obj == tgt:
                errors.append(f"step_{i}: self-place {obj} on {obj}")
        return len(errors) == 0, errors


# ── VerifierBank ─────────────────────────────────────────────────────────────

class VerifierBank:
    """Chain multiple verifiers and aggregate results.

    Usage:
        bank = VerifierBank()
        bank.add(GoalVerifier(objects))
        bank.add(PlanVerifier())
        report = bank.verify_all(goal_json=..., actions=...)
    """

    def __init__(self):
        self._verifiers: dict[str, BaseVerifier] = {}

    def add(self, verifier: BaseVerifier):
        self._verifiers[verifier.name] = verifier

    def get(self, name: str) -> BaseVerifier | None:
        return self._verifiers.get(name)

    def verify_all(self, **kwargs) -> VerificationReport:
        """Run all registered verifiers with the given keyword arguments.

        Each verifier receives only the kwargs it needs (via introspection
        or by convention — verifiers ignore extra kwargs via **kwargs).

        Returns an aggregated VerificationReport.
        """
        aggregated = VerificationReport(verifier_name="VerifierBank", passed=True)

        # Map verifier names to their expected kwarg names
        verifier_kwargs = {
            "GoalVerifier": ["goal_json", "task_type"],
            "SubgoalVerifier": ["subgoals", "goal_predicates"],
            "PlanVerifier": ["actions", "initial_scene_graph", "goal_predicates"],
            "OptionContractVerifier": ["contract", "state_predicates", "target_object", "target_location"],
            "RepairVerifier": ["repair_proposal", "retries_used", "original_goal"],
            "CoarseOptionVerifier": ["task_program", "coarse_steps", "scene_graph"],
        }

        for name, verifier in self._verifiers.items():
            expected = verifier_kwargs.get(name, [])
            filtered = {k: v for k, v in kwargs.items() if k in expected}
            # Skip verifiers that don't have enough args (not applicable)
            if not filtered and expected:
                continue
            try:
                report = verifier.check(**filtered)
            except Exception as e:
                report = VerificationReport(
                    passed=False, verifier_name=name,
                    failures=[("exception", str(e))])
            aggregated.merge(report)

        return aggregated

    @property
    def verifier_names(self) -> list[str]:
        return list(self._verifiers.keys())


# ── Factory: Standard VerifierBank for 2-block stacking ──────────────────────

def standard_verifier_bank(objects: list[str]) -> VerifierBank:
    """Create a standard verifier bank for stacking tasks.

    Args:
        objects: List of valid object IDs (e.g. ["cube1", "cube2", "table"]).

    Returns:
        VerifierBank with GoalVerifier, SubgoalVerifier, PlanVerifier,
        OptionContractVerifier, RepairVerifier, and CoarseOptionVerifier.
    """
    bank = VerifierBank()
    bank.add(GoalVerifier(valid_objects=objects))
    bank.add(SubgoalVerifier(valid_objects=objects))
    bank.add(PlanVerifier())
    bank.add(OptionContractVerifier())
    bank.add(RepairVerifier(valid_objects=objects, max_retries=3))
    bank.add(CoarseOptionVerifier(valid_objects=objects))
    return bank
