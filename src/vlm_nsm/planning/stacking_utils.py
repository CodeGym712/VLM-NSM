"""Utilities for converting aligned predicates to planner target stacks.

S14.5-A: target_stack_from_predicates — parse predicates into
bottom-to-top stack ordering for the internal stacking planner.
"""

from __future__ import annotations


def target_stack_from_predicates(
    predicates: list[tuple[str, str, str]],
    blocks: list[str],
    require_table_support: bool = True,
) -> list[str]:
    """Convert on(block, support) predicates into a bottom-to-top stack list.

    Args:
        predicates: List of (predicate_name, arg1, arg2) tuples where
                    predicate_name is "on", arg1 is the upper block,
                    arg2 is the lower block or "table".
        blocks: Valid block IDs (e.g. ["cube1", "cube2"]).
        require_table_support: If True, exactly one block must be on table.

    Returns:
        Bottom-to-top stack list, e.g. ["cube2", "cube1"].

    Raises:
        ValueError: If predicates are invalid (unknown objects, cycles,
                    duplicate support, disconnected chains, missing table).
    """
    # Filter to on predicates only
    on_relations: dict[str, str] = {}  # upper_block -> support
    block_set = set(blocks)

    for pred in predicates:
        name = str(pred[0])

        if name != "on":
            continue

        arg1 = str(pred[1]).lower()
        arg2 = str(pred[2]).lower()

        if arg1 not in block_set:
            raise ValueError(f"Unknown object in predicate: {arg1} not in {blocks}")
        if arg2 not in block_set and arg2 != "table":
            raise ValueError(f"Unknown support: {arg2} not in {blocks} and not 'table'")

        if arg1 in on_relations:
            raise ValueError(f"Duplicate support: {arg1} is on both {on_relations[arg1]} and {arg2}")
        on_relations[arg1] = arg2

    # Check for cycles via DFS
    visited = set()
    for block in blocks:
        current = block
        seen = set()
        while current in on_relations:
            if current in seen:
                cycle = " -> ".join(list(seen) + [current])
                raise ValueError(f"Cycle detected in predicates: {cycle}")
            seen.add(current)
            visited.add(current)
            current = on_relations[current]
            if current not in block_set:
                break  # ends at table or unknown

    # Count table-based blocks
    on_table = [b for b, s in on_relations.items() if s == "table"]

    if require_table_support:
        if len(on_table) == 0:
            raise ValueError("No block is on table (missing table support)")
        if len(on_table) > 1:
            table_blocks = ", ".join(on_table)
            raise ValueError(f"Multiple blocks on table: {table_blocks}")

    # Check that all blocks are in a single connected chain
    if require_table_support:
        bottom = on_table[0]
        # Build the chain from bottom up
        chain = [bottom]
        supported_by = {s: b for b, s in on_relations.items()}  # support -> upper_block
        current = bottom
        while current in supported_by:
            upper = supported_by[current]
            chain.append(upper)
            current = upper

        # Check all blocks are covered
        chain_set = set(chain)
        missing = block_set - chain_set
        if missing:
            raise ValueError(f"Blocks not in chain: {sorted(missing)}")

        # Check no duplicates in chain
        if len(chain_set) != len(chain):
            raise ValueError("Duplicate block in chain (cycle)")
    else:
        # Without table requirement, just sort blocks that appear
        chain = [b for b in blocks if b not in on_relations]  # blocks with no support
        if chain:
            supported_by = {s: b for b, s in on_relations.items()}
            current = chain[0]
            while current in supported_by:
                chain.append(supported_by[current])
                current = chain[-1]

    return chain
