from __future__ import annotations

from vlm_nsm.core.data_types import AlignmentResult, Predicate


DEFAULT_ALIASES = {
    "handempty": "free-gripper",
    "emptyhand": "free-gripper",
    "holding": "grasped",
    "placed_on": "on",
    "above": "on",
}


class AlignmentLayer:
    def __init__(self, schema: dict | None = None, aliases: dict[str, str] | None = None):
        self.schema = schema
        self.aliases = aliases if aliases is not None else DEFAULT_ALIASES

    def align_and_validate(
        self,
        goal_json: dict,
        valid_objects: list[str],
        valid_predicates: list[str],
        type_table: dict[str, str],
        predicate_signature: dict[str, list[list[str]]],
        task_type: str,
        target_peg: str | None = None,
        allow_hanoi_autocomplete: bool = False,
        n_blocks: int | None = None,
    ) -> AlignmentResult:
        schema_error = self._schema_error(goal_json)
        if schema_error:
            return AlignmentResult(False, [], False, "schema_error", {"message": schema_error})

        valid_obj_set = set(valid_objects)
        valid_pred_set = set(valid_predicates)
        repaired = False
        aligned: list[Predicate] = []
        for raw_pred in goal_json.get("goal_predicates", []):
            name = raw_pred[0].lower()
            canonical_name = self.aliases.get(name, name)
            if canonical_name != name:
                repaired = True
            if canonical_name not in valid_pred_set:
                return AlignmentResult(False, [], repaired, "invalid_predicate", {"predicate": name})
            args = [str(arg).lower() for arg in raw_pred[1:]]
            for arg in args:
                if arg not in valid_obj_set:
                    return AlignmentResult(False, [], repaired, "invalid_object", {"object": arg})
            if not self._type_check(canonical_name, args, type_table, predicate_signature):
                return AlignmentResult(False, [], repaired, "type_error", {"predicate": canonical_name, "args": args})
            aligned.append(Predicate(canonical_name, args))

        if task_type == "hanoi":
            completeness = self._check_hanoi_completeness(aligned, type_table, target_peg, n_blocks)
            if completeness[0] is False:
                if allow_hanoi_autocomplete and target_peg:
                    aligned = completeness[2]
                    repaired = True
                else:
                    return AlignmentResult(False, aligned, repaired, "incomplete_goal", {"missing": [p.to_pddl() for p in completeness[1]]})
        elif task_type == "stacking":
            completeness = self._check_stacking_completeness(aligned, type_table, n_blocks)
            if completeness[0] is False:
                return AlignmentResult(False, aligned, repaired, "incomplete_goal", {"missing": [p.to_pddl() for p in completeness[1]]})

        return AlignmentResult(True, aligned, repaired, None, {})

    def _schema_error(self, goal_json: dict) -> str | None:
        if not isinstance(goal_json, dict):
            return "goal_json must be an object"
        for key in ["task_type", "object_bindings", "goal_predicates"]:
            if key not in goal_json:
                return f"missing required key: {key}"
        if not isinstance(goal_json["task_type"], str):
            return "task_type must be a string"
        if not isinstance(goal_json["object_bindings"], dict):
            return "object_bindings must be an object"
        if not all(isinstance(v, str) for v in goal_json["object_bindings"].values()):
            return "object_bindings values must be strings"
        if not isinstance(goal_json["goal_predicates"], list):
            return "goal_predicates must be an array"
        for pred in goal_json["goal_predicates"]:
            if not isinstance(pred, list) or not pred:
                return "each goal predicate must be a non-empty array"
            if not all(isinstance(item, str) for item in pred):
                return "predicate items must be strings"
        return None

    def _type_check(self, predicate: str, args: list[str], type_table: dict[str, str], signature: dict[str, list[list[str]]]) -> bool:
        if predicate not in signature:
            return False
        alternatives = signature[predicate]
        for alt in alternatives:
            if len(alt) != len(args):
                continue
            ok = True
            for obj, allowed_types in zip(args, alt):
                allowed = set(allowed_types.split("|")) if isinstance(allowed_types, str) else set(allowed_types)
                if type_table.get(obj) not in allowed:
                    ok = False
                    break
            if ok:
                return True
        return False

    def _check_stacking_completeness(self, goals: list[Predicate], type_table: dict[str, str], n_blocks: int | None):
        """Validate on predicates form a complete single-stack chain (any ordering).

        Supports any block ordering (e.g. both ``on(cube1, cube2)`` and
        ``on(cube2, cube1)``), as long as the chain is a valid linear stack
        with exactly one block on table and all blocks connected.
        """
        blocks = sorted([obj for obj, typ in type_table.items() if typ == "block"], key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
        if n_blocks:
            blocks = blocks[:n_blocks]
        if not blocks:
            return True, [], goals

        block_set = set(blocks)

        # Build on relation map: upper_block -> support
        on_relations: dict[str, str] = {}
        for g in goals:
            if g.name == "on":
                upper, support = g.args[0], g.args[1]
                if upper in on_relations:
                    return False, [g], goals  # duplicate support
                on_relations[upper] = support

        # Only consider block-on-X (X may be block or table)
        filtered = {u: s for u, s in on_relations.items() if u in block_set}

        # Exactly one block must be on table
        on_table = [b for b, s in filtered.items() if s == "table"]
        if len(on_table) != 1:
            return False, [], goals

        bottom = on_table[0]

        # Build reverse map: support -> upper (linear chain check)
        supported_by: dict[str, str] = {}
        for upper, support in filtered.items():
            if support in block_set:
                if support in supported_by:
                    return False, [], goals  # multiple blocks on same support
                supported_by[support] = upper

        # Check for cycles and build chain from bottom up
        chain = [bottom]
        current = bottom
        while current in supported_by:
            upper = supported_by[current]
            if upper in chain:
                return False, [], goals  # cycle
            chain.append(upper)
            current = upper

        chain_set = set(chain)
        if chain_set != block_set:
            return False, [], goals  # disconnected blocks
        if len(chain_set) != len(chain):
            return False, [], goals  # duplicate in chain (safety)

        # Build expected predicates from chain
        # chain = [bottom, ..., top] → [on(top, next_top), ..., on(bottom, table)]
        expected = []
        for i in range(len(chain) - 1, 0, -1):
            expected.append(Predicate("on", [chain[i], chain[i - 1]]))
        expected.append(Predicate("on", [chain[0], "table"]))

        goal_set = {g.as_tuple() for g in goals}
        missing = [p for p in expected if p.as_tuple() not in goal_set]
        if missing:
            return False, missing, expected
        return True, [], goals

    def _check_hanoi_completeness(self, goals: list[Predicate], type_table: dict[str, str], target_peg: str | None, n_blocks: int | None):
        disks = sorted([obj for obj, typ in type_table.items() if typ == "disk"], key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
        if n_blocks:
            disks = disks[:n_blocks]
        if not disks or not target_peg:
            return True, [], goals
        expected = [Predicate("on", [disks[i], disks[i + 1]]) for i in range(len(disks) - 1)]
        expected.append(Predicate("on", [disks[-1], target_peg]))
        goal_set = {g.as_tuple() for g in goals}
        missing = [p for p in expected if p.as_tuple() not in goal_set]
        if missing:
            return False, missing, expected
        return True, [], goals
