"""Convert symbolic planner actions into skill-level commands."""

from __future__ import annotations

from typing import Any

from vlm_nsm.skills.skill_types import SkillCommand, SkillPlan


def _extract_actions(symbolic_plan: Any) -> list[Any]:
    """Return a list of symbolic actions from common planner result shapes."""
    if symbolic_plan is None:
        return []
    if hasattr(symbolic_plan, "actions"):
        return list(symbolic_plan.actions)
    if isinstance(symbolic_plan, dict) and "actions" in symbolic_plan:
        return list(symbolic_plan["actions"])
    return list(symbolic_plan)


def _action_name_args(action: Any) -> tuple[str, list[str]]:
    """Normalize Action/dataclass/dict/list forms into ``(name, args)``."""
    if hasattr(action, "name") and hasattr(action, "args"):
        return str(action.name).lower(), [str(arg) for arg in action.args]

    if isinstance(action, dict):
        name = action.get("name") or action.get("action") or action.get("action_name")
        args = action.get("args") or action.get("arguments")
        if name is None:
            raise ValueError(f"Symbolic action dict missing name: {action}")
        if args is None:
            args = []
        return str(name).lower(), [str(arg) for arg in args]

    if isinstance(action, (list, tuple)) and action:
        name = str(action[0]).lower()
        if len(action) == 2 and isinstance(action[1], (list, tuple)):
            args = [str(arg) for arg in action[1]]
        else:
            args = [str(arg) for arg in action[1:]]
        return name, args

    raise ValueError(f"Unsupported symbolic action format: {action!r}")


# Fix C3: Supported symbolic action names (including BFS SymbolicPlanner output)
_SUPPORTED_ACTIONS = {"move", "move_on", "move_to_table"}


def _move_to_skill_command(action: Any) -> SkillCommand:
    name, args = _action_name_args(action)
    if name not in _SUPPORTED_ACTIONS:
        raise ValueError(
            f"Unsupported symbolic action '{name}'; expected one of {_SUPPORTED_ACTIONS}"
        )

    # Phase 2.0: extract operator semantics from SymbolicAction
    preconditions = _extract_predicate_dicts(action, "preconditions")
    add_effects = _extract_predicate_dicts(action, "effects_add")
    del_effects = _extract_predicate_dicts(action, "effects_del")

    if name == "move_to_table":
        # MoveToTable(x): place object x on the table
        if len(args) < 1:
            raise ValueError(f"move_to_table requires at least 1 arg (object), got {args}")
        obj = args[0]
        source = "table"  # source is wherever the object was
        target = "table"
        relation = "table"
    elif name in ("move", "move_on"):
        # move(obj, source, target) or MoveOn(x, y)
        if len(args) < 2:
            raise ValueError(
                f"{name} action requires at least 2 args (object, target); got {args}"
            )
        if name == "move" and len(args) >= 3:
            obj, source, target = args[0], args[1], args[2]
        else:
            # move_on(x, y): place x on y
            obj = args[0]
            source = "table"
            target = args[1] if len(args) >= 2 else "table"

        if not target or target == "table":
            relation = "table"
        else:
            relation = "on"
    else:
        raise ValueError(f"Unexpected action name: {name}")

    return SkillCommand(
        skill_name="pick_place",
        object_id=obj,
        source=source,
        target_object_id=target,
        relation=relation,
        metadata={"symbolic_action": {"name": name, "args": args}},
        preconditions=preconditions,
        expected_add_effects=add_effects,
        expected_del_effects=del_effects,
    )


def _extract_predicate_dicts(action: Any, attr: str) -> list[dict[str, Any]]:
    """Extract predicate list from a SymbolicAction attribute."""
    if not hasattr(action, attr):
        return []
    predicates = getattr(action, attr)
    if predicates is None:
        return []
    result = []
    for p in predicates:
        if hasattr(p, "name") and hasattr(p, "args"):
            result.append({"name": str(p.name), "args": list(p.args)})
        elif isinstance(p, dict):
            result.append(p)
    return result


def skill_plan_from_symbolic(symbolic_plan: Any) -> SkillPlan:
    """Convert symbolic move actions into a ``SkillPlan``.

    Currently supports stacking planner actions of the form
    ``move(object, source, target)`` and converts each one into a
    ``pick_place`` skill command.
    """
    actions = _extract_actions(symbolic_plan)
    commands = [_move_to_skill_command(action) for action in actions]
    return SkillPlan(
        commands=commands,
        source_plan=actions,
        metadata={
            "source": "symbolic_plan",
            "num_commands": len(commands),
            "supported_action": "move",
        },
    )
