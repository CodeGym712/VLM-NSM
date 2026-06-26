"""OptionContract — defines initiation/termination conditions and
expected effects for each skill option.

Each skill (reach, grasp, lift, reach_place, release_place, retreat)
has a contract specifying:
  - initiation_predicates: what must be true before execution
  - termination_predicates: what must be true to consider it done
  - expected_effects: what predicates should change after success
  - timeout, retry_policy, failure_types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptionContract:
    """Formal contract for a single skill option.

    Attributes:
        name: Skill name (e.g. "reach", "grasp").
        initiation_predicates: Predicates that must hold before execution.
            Format: [["pred_name", "arg1", "arg2"], ...]
        termination_predicates: Predicates that indicate success.
        expected_effects: Predicates expected to be added/removed.
            Format: {"add": [...], "del": [...]}
        timeout_steps: Max steps before forced termination.
        retry_policy: {"max_retries": N, "strategy": "reset"|"continue"|"xy_realign"}
        failure_types: Known failure modes for this option.
        verifier_hooks: Optional list of verifier names to run post-execution.
        parameters_schema: Expected parameter fields.
    """

    name: str = ""
    initiation_predicates: list[list[str]] = field(default_factory=list)
    termination_predicates: list[list[str]] = field(default_factory=list)
    expected_effects: dict[str, list[list[str]]] = field(default_factory=lambda: {"add": [], "del": []})
    timeout_steps: int = 500
    retry_policy: dict[str, Any] = field(default_factory=lambda: {"max_retries": 1, "strategy": "reset"})
    failure_types: list[str] = field(default_factory=list)
    verifier_hooks: list[str] = field(default_factory=list)
    parameters_schema: list[str] = field(default_factory=lambda: ["target_object", "target_location"])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "initiation_predicates": self.initiation_predicates,
            "termination_predicates": self.termination_predicates,
            "expected_effects": self.expected_effects,
            "timeout_steps": self.timeout_steps,
            "retry_policy": self.retry_policy,
            "failure_types": self.failure_types,
        }

    def check_initiation(self, state_predicates: list[list[str]]) -> tuple[bool, list[list[str]]]:
        """Check if initiation conditions are satisfied.

        Returns:
            (ok, missing_predicates) — missing predicates that need to be satisfied.
        """
        state_set = {tuple(p) for p in state_predicates}
        missing = []
        for p in self.initiation_predicates:
            if tuple(p) not in state_set:
                missing.append(p)
        return len(missing) == 0, missing

    def check_effects(self, pre_predicates: list[list[str]],
                      post_predicates: list[list[str]]) -> tuple[bool, dict]:
        """Check if expected effects were achieved.

        Returns:
            (all_ok, details) where details maps each expected effect to its status.
        """
        pre_set = {tuple(p) for p in pre_predicates}
        post_set = {tuple(p) for p in post_predicates}
        details = {"add_ok": [], "add_fail": [], "del_ok": [], "del_fail": []}

        for p in self.expected_effects.get("add", []):
            if tuple(p) in post_set:
                details["add_ok"].append(p)
            else:
                details["add_fail"].append(p)

        for p in self.expected_effects.get("del", []):
            if tuple(p) not in post_set:
                details["del_ok"].append(p)
            else:
                details["del_fail"].append(p)

        all_ok = len(details["add_fail"]) == 0 and len(details["del_fail"]) == 0
        return all_ok, details


# ── Standard 2-Block Contracts ───────────────────────────────────────────────

# ── Phase-level contracts (reach/grasp/lift/reach_place/release_place/retreat)
#     Kept for backward compatibility and internal use by AtomicPick/AtomicPlace.
#     NOT exposed to upper neuro-symbolic layer.

def _contract_reach() -> OptionContract:
    return OptionContract(
        name="reach",
        initiation_predicates=[["handempty"], ["graspable", "{target_object}"]],
        termination_predicates=[],  # distance-based convergence
        # reach only moves EEF — no predicate changes expected
        expected_effects={"add": [], "del": []},
        timeout_steps=800,
        retry_policy={"max_retries": 1, "strategy": "reset"},
        failure_types=["timeout", "env_error", "reach_failed", "approach_blocked"],
        parameters_schema=["target_object", "approach_height"],
    )


def _contract_grasp() -> OptionContract:
    return OptionContract(
        name="grasp",
        initiation_predicates=[["clear", "{target_object}"], ["graspable", "{target_object}"]],
        termination_predicates=[],
        expected_effects={
            "add": [["holding", "{target_object}"]],
            "del": [["handempty"], ["ontable", "{target_object}"]],
        },
        timeout_steps=300,
        retry_policy={"max_retries": 3, "strategy": "continue"},
        failure_types=["nocontact", "slip", "prereach", "grasp_failed", "horizon"],
        parameters_schema=["target_object", "descend_offset"],
    )


def _contract_lift() -> OptionContract:
    return OptionContract(
        name="lift",
        initiation_predicates=[["holding", "{target_object}"]],
        termination_predicates=[],  # z-height based
        # lift only changes EEF z — no predicate changes expected
        expected_effects={"add": [], "del": []},
        timeout_steps=800,
        retry_policy={"max_retries": 1, "strategy": "reset"},
        failure_types=["timeout", "env_error", "lift_failed"],
        parameters_schema=["target_object", "lift_height"],
    )


def _contract_reach_place() -> OptionContract:
    return OptionContract(
        name="reach_place",
        initiation_predicates=[["holding", "{target_object}"]],
        termination_predicates=[],
        expected_effects={"add": [], "del": []},
        timeout_steps=1000,
        retry_policy={"max_retries": 1, "strategy": "reset"},
        failure_types=["timeout", "env_error", "reach_place_failed"],
        parameters_schema=["target_object", "target_location"],
    )


def _contract_release_place() -> OptionContract:
    return OptionContract(
        name="release_place",
        initiation_predicates=[["holding", "{target_object}"]],
        termination_predicates=[],
        # Effects depend on target_location:
        #   if target is object: add on(obj,target), del clear(target)
        #   if target is table: add ontable(obj) [NOT clear(table)]
        #   always: add clear(obj), add handempty, del holding(obj)
        expected_effects={
            "add": [["on", "{target_object}", "{target_location}"],
                    ["clear", "{target_object}"], ["handempty"]],
            "del": [["holding", "{target_object}"]],
        },
        timeout_steps=600,
        retry_policy={"max_retries": 2, "strategy": "continue"},
        failure_types=["preplace", "verify_failed", "place_failed", "horizon"],
        parameters_schema=["target_object", "target_location"],
    )


def _contract_retreat() -> OptionContract:
    return OptionContract(
        name="retreat",
        initiation_predicates=[["handempty"]],
        termination_predicates=[],
        expected_effects={"add": [], "del": []},
        timeout_steps=500,
        retry_policy={"max_retries": 1, "strategy": "reset"},
        failure_types=["timeout", "env_error", "retreat_failed"],
        parameters_schema=[],
    )


# ── Coarse option contracts (pick/place/move/ensure/achieve)
#     These are the UPPER-LAYER contracts exposed to TaskComposer / VerifierBank.
#     Reach/grasp/lift/reach_place/release_place are internal implementation only.

def _contract_pick() -> OptionContract:
    """pick(obj): reach + grasp + lift as one atomic option.

    Initiation: handempty, obj graspable and clear.
    Effect: holding(obj); not handempty; obj no longer ontable.
    """
    return OptionContract(
        name="pick",
        initiation_predicates=[
            ["handempty"],
            ["graspable", "{target_object}"],
            ["clear", "{target_object}"],
        ],
        termination_predicates=[],
        expected_effects={
            "add": [["holding", "{target_object}"]],
            "del": [["handempty"], ["ontable", "{target_object}"],
                    ["clear", "{target_object}"]],
        },
        timeout_steps=2000,  # reach+grasp+lift combined
        retry_policy={"max_retries": 2, "strategy": "reset"},
        failure_types=["timeout", "env_error", "slip", "nocontact",
                       "pick_failed"],
        parameters_schema=["target_object"],
    )


def _contract_place() -> OptionContract:
    """place(obj, target): reach_place + release_place as one atomic option.

    Initiation: holding(obj), target is clear (if not table).
    Effect: on(obj, target), clear(obj), handempty; not holding(obj).
    """
    return OptionContract(
        name="place",
        initiation_predicates=[
            ["holding", "{target_object}"],
        ],
        termination_predicates=[],
        expected_effects={
            "add": [["on", "{target_object}", "{target_location}"],
                    ["clear", "{target_object}"], ["handempty"]],
            "del": [["holding", "{target_object}"]],
        },
        timeout_steps=2000,  # reach_place+release_place combined
        retry_policy={"max_retries": 2, "strategy": "reset"},
        failure_types=["timeout", "env_error", "preplace", "verify_failed",
                       "place_failed"],
        parameters_schema=["target_object", "target_location"],
    )


def _contract_move() -> OptionContract:
    """move(obj, source, target): pick from source + place on target.

    Initiation: obj on source (or ontable), source clear for pick, target clear.
    Effect: on(obj, target), clear(obj), clear(source), handempty.
    """
    return OptionContract(
        name="move",
        initiation_predicates=[
            ["handempty"],
            ["graspable", "{target_object}"],
        ],
        termination_predicates=[],
        expected_effects={
            "add": [["on", "{target_object}", "{target_location}"],
                    ["clear", "{target_object}"], ["handempty"]],
            "del": [["holding", "{target_object}"]],
        },
        timeout_steps=5000,  # pick + place
        retry_policy={"max_retries": 2, "strategy": "reset"},
        failure_types=["timeout", "env_error", "pick_failed", "place_failed",
                       "move_failed"],
        parameters_schema=["target_object", "source_location", "target_location"],
    )


def _contract_ensure() -> OptionContract:
    """ensure(predicate): verify predicate holds; if not, replan.

    No state change. Purely a verification gate.
    """
    return OptionContract(
        name="ensure",
        initiation_predicates=[],
        termination_predicates=[],  # the predicate itself
        expected_effects={"add": [], "del": []},
        timeout_steps=100,
        retry_policy={"max_retries": 3, "strategy": "replan"},
        failure_types=["ensure_failed"],
        parameters_schema=["predicate"],
    )


def _contract_achieve() -> OptionContract:
    """achieve(predicate): all-out attempt to make predicate true.

    May decompose into sub-steps (planning) or use direct execution.
    """
    return OptionContract(
        name="achieve",
        initiation_predicates=[],
        termination_predicates=[],  # the target predicate
        expected_effects={"add": [], "del": []},
        timeout_steps=10000,
        retry_policy={"max_retries": 3, "strategy": "replan"},
        failure_types=["achieve_failed", "timeout"],
        parameters_schema=["predicate"],
    )


# ── Contract Registry ────────────────────────────────────────────────────────

_DEFAULT_CONTRACTS: dict[str, OptionContract] = {
    # Phase-level (internal use, backward compat)
    "reach": _contract_reach(),
    "grasp": _contract_grasp(),
    "lift": _contract_lift(),
    "reach_place": _contract_reach_place(),
    "release_place": _contract_release_place(),
    "retreat": _contract_retreat(),
    # Coarse options (upper-layer API)
    "pick": _contract_pick(),
    "place": _contract_place(),
    "move": _contract_move(),
    "ensure": _contract_ensure(),
    "achieve": _contract_achieve(),
}

# Coarse-only contracts (upper neuro-symbolic layer)
_COARSE_CONTRACTS: dict[str, OptionContract] = {
    "pick": _contract_pick(),
    "place": _contract_place(),
    "move": _contract_move(),
    "ensure": _contract_ensure(),
    "achieve": _contract_achieve(),
}


def get_contract(skill_name: str) -> OptionContract:
    """Get the standard option contract for a skill."""
    if skill_name not in _DEFAULT_CONTRACTS:
        raise KeyError(f"No contract for skill '{skill_name}'. Available: {list(_DEFAULT_CONTRACTS.keys())}")
    return _DEFAULT_CONTRACTS[skill_name]


def get_all_contracts() -> dict[str, OptionContract]:
    return dict(_DEFAULT_CONTRACTS)


def get_coarse_contracts() -> dict[str, OptionContract]:
    """Get only coarse option contracts (pick/place/move/ensure/achieve)."""
    return dict(_COARSE_CONTRACTS)


def get_coarse_contract(skill_name: str) -> OptionContract:
    """Get a coarse option contract."""
    if skill_name not in _COARSE_CONTRACTS:
        raise KeyError(f"No coarse contract for '{skill_name}'. Available: {list(_COARSE_CONTRACTS.keys())}")
    return _COARSE_CONTRACTS[skill_name]
