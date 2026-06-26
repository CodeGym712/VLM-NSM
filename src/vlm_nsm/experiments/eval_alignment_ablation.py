"""Alignment Layer Ablation: inject bad goal_json directly, no parser/LLM needed.

Tests 6 alignment checks against 56 hand-crafted goal_json payloads.
Categories: valid, illegal_obj, illegal_pred, missing_target, table_as_pick,
            self_reference, empty, no_schema.

Usage: nsm/bin/python -m vlm_nsm.experiments.eval_alignment_ablation
"""

VALID_OBJECTS = ["cube1", "cube2"]
VALID_PREDS = {"on"}

TEST_CASES = [
    # === VALID (6) ===
    ("on(cube1,cube2)", [["on","cube1","cube2"]], "valid"),
    ("on(cube2,cube1)", [["on","cube2","cube1"]], "valid"),
    ("on(cube1,cube2)+ontable(cube2)", [["on","cube1","cube2"],["on","cube2","table"]], "valid"),
    ("on(cube2,table)", [["on","cube2","table"]], "valid"),
    ("single: on(cube1,cube2)", [["on","cube1","cube2"]], "valid"),
    ("two step stack", [["on","cube1","cube2"],["on","cube2","cube1"]], "valid"),

    # === ILLEGAL OBJECT (10) ===
    ("cube5 doesn't exist", [["on","cube5","cube2"]], "illegal_obj"),
    ("cube3 missing", [["on","cube3","cube1"]], "illegal_obj"),
    ("purple_cube unknown", [["on","purple_cube","cube2"]], "illegal_obj"),
    ("blockA not in scene", [["on","blockA","cube1"]], "illegal_obj"),
    ("cube999", [["on","cube999","table"]], "illegal_obj"),
    ("blue_cube", [["on","blue_cube","cube2"]], "illegal_obj"),
    ("yellow_block", [["on","yellow_block","cube1"]], "illegal_obj"),
    ("target is cube5", [["on","cube1","cube5"]], "illegal_obj"),
    ("both illegal", [["on","cube5","cube6"]], "illegal_obj"),
    ("obj_x", [["on","obj_x","cube1"]], "illegal_obj"),

    # === ILLEGAL PREDICATE (8) ===
    ("above is unsupported", [["above","cube1","cube2"]], "illegal_pred"),
    ("next_to unsupported", [["next_to","cube1","cube2"]], "illegal_pred"),
    ("under", [["under","cube2","cube1"]], "illegal_pred"),
    ("in_front_of", [["in_front_of","cube1","cube2"]], "illegal_pred"),
    ("beside", [["beside","cube1","cube2"]], "illegal_pred"),
    ("grasping is not goal predicate", [["grasping","cube1"]], "illegal_pred"),
    ("holding", [["holding","cube1"]], "illegal_pred"),
    ("inside", [["inside","cube1","cube2"]], "illegal_pred"),

    # === MISSING TARGET (6) ===
    ("1-arg: on(cube1)", [["on","cube1"]], "missing_target"),
    ("1-arg: on(cube2)", [["on","cube2"]], "missing_target"),
    ("incomplete: on", [["on"]], "missing_target"),
    ("mixed: one valid one short", [["on","cube1","cube2"],["on","cube2"]], "missing_target"),
    ("pred with 4 args", [["on","cube1","cube2","extra"]], "missing_target"),
    ("pred with 0 args", [[]], "missing_target"),

    # === TABLE AS PICK (5) ===
    ("table as pick object", [["on","table","cube1"]], "table_as_pick"),
    ("table on cube1", [["on","table","cube2"]], "table_as_pick"),
    ("table on table", [["on","table","table"]], "table_as_pick"),
    ("table as both", [["on","table","table"]], "table_as_pick"),
    ("table picked", [["on","table","cube1"],["on","cube2","table"]], "table_as_pick"),

    # === SELF-REFERENCE (5) ===
    ("cube1 on cube1", [["on","cube1","cube1"]], "self_ref"),
    ("cube2 on cube2", [["on","cube2","cube2"]], "self_ref"),
    ("in stack", [["on","cube1","cube2"],["on","cube2","cube2"]], "self_ref"),
    ("both self-ref", [["on","cube1","cube1"],["on","cube2","cube2"]], "self_ref"),
    ("table self-ref implicit", [["on","table","table"]], "self_ref"),

    # === EMPTY / NO SCHEMA (8) ===
    ("empty list", [], "empty"),
    ("null", None, "empty"),
    ("not a list of lists", {"action": "move"}, "empty"),
    ("string predicate", ["on(cube1,cube2)"], "empty"),
    ("nested dicts", [{"pred":"on","args":["cube1","cube2"]}], "empty"),
    ("numbers", [[1,2,3]], "empty"),
    ("mixed valid+invalid", [["on","cube1","cube2"],"bad"], "empty"),
    ("deeply nested", [[["on","cube1","cube2"]]], "empty"),

    # === EDGE CASES (8) ===
    ("table target is valid", [["on","cube1","table"]], "valid"),
    ("table target (2)", [["on","cube2","table"]], "valid"),
    ("duplicate predicates", [["on","cube1","cube2"],["on","cube1","cube2"]], "valid"),
    ("reversed pair", [["on","cube1","cube2"],["on","cube2","cube1"]], "valid"),
    ("with table target", [["on","cube1","table"]], "valid"),
    ("single table target", [["on","cube2","table"]], "valid"),
    ("multiple valid", [["on","cube1","cube2"],["on","cube2","table"]], "valid"),
    ("redundant table", [["on","cube1","cube2"],["on","cube2","table"]], "valid"),
]
# Total: 56 cases


def check(gj):
    preds = gj if isinstance(gj, list) else []
    r = {}
    r["schema"] = bool(preds) and all(isinstance(p, list) and len(p) == 3 for p in preds)
    if r["schema"]:
        objs = set()
        for p in preds:
            for i in [1, 2]:
                objs.add(str(p[i]))
        r["whitelist"] = objs.issubset(set(VALID_OBJECTS) | {"table"})
        r["predicate"] = all(p[0] in VALID_PREDS for p in preds)
        r["type"] = all(p[1] != "table" for p in preds)
        r["completeness"] = len(preds) >= 1
    else:
        for k in ["whitelist", "predicate", "type", "completeness"]:
            r[k] = False
    return r


def main():
    modes = [
        ("none", []),
        ("schema", ["schema"]),
        ("+whitelist", ["schema", "whitelist"]),
        ("+predicate", ["schema", "whitelist", "predicate"]),
        ("+type", ["schema", "whitelist", "predicate", "type"]),
        ("full(+completeness)", ["schema", "whitelist", "predicate", "type", "completeness"]),
    ]

    print(f"{'Condition':<22} {'Pass':>5} {'Reject':>7} {'Rate':>7}")
    print("-" * 45)
    for mode_name, required in modes:
        passed = 0; total = 0
        for _, gj, _ in TEST_CASES:
            if gj is None:
                total += 1
                continue
            c = check(gj)
            total += 1
            if all(c.get(r, False) for r in required):
                passed += 1
        print(f"{mode_name:<22} {passed:>5} {total-passed:>7} {passed/max(total,1)*100:>6.1f}%")

    # Category breakdown (full mode)
    required = modes[-1][1]
    cat = {}
    for label, gj, category in TEST_CASES:
        c = check(gj) if gj is not None else {"schema": False}
        ok = all(c.get(r, False) for r in required)
        cat.setdefault(category, {"ok": 0, "fail": 0, "examples": []})
        cat[category]["ok" if ok else "fail"] += 1
        if not ok:
            failed = [r for r in required if not c.get(r, False)]
            cat[category]["examples"].append(f"{label[:45]} [{','.join(failed)}]")

    print(f"\n{'='*50}")
    print("Per-category (full mode):")
    for cname in ["valid", "illegal_obj", "illegal_pred", "missing_target",
                  "table_as_pick", "self_ref", "empty"]:
        if cname in cat:
            s = cat[cname]
            print(f"\n{cname} ({s['ok']}/{s['ok']+s['fail']} pass):")
            for ex in s["examples"][:3]:
                print(f"  REJECT: {ex}")


if __name__ == "__main__":
    main()
