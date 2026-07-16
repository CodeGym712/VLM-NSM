"""Evidence-constrained minimal clause edits for learned planning domains."""

from __future__ import annotations

import copy
from typing import Any


def apply_minimal_edit(domain: dict[str, Any], witness: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one witness-supported edit without changing unrelated clauses."""
    revised = copy.deepcopy(domain)
    operator_name = str(witness["operator"])
    literal = list(witness["literal"])
    operator = next(item for item in revised["operators"] if item["name"] == operator_name)
    edit_type = str(witness.get("edit_type", "add_positive_precondition"))
    if edit_type == "add_positive_precondition":
        field = "preconditions_pos"
        if literal not in operator[field]:
            operator[field].append(literal)
    elif edit_type == "add_negative_precondition":
        field = "preconditions_neg"
        if literal not in operator[field]:
            operator[field].append(literal)
    elif edit_type in {"add_effect", "delete_effect"}:
        field = "effects_add" if edit_type == "add_effect" else "effects_del"
        if literal not in operator[field]:
            operator[field].append(literal)
    else:
        raise ValueError(f"unsupported witness edit type: {edit_type}")
    edit = {
        "witness_id": witness.get("witness_id", "unassigned"),
        "operator": operator_name,
        "edit_type": edit_type,
        "literal": literal,
        "reason": witness.get("reason", "witness-supported minimal edit"),
        "minimality": "one literal; no unrelated operator changed",
    }
    return revised, edit
