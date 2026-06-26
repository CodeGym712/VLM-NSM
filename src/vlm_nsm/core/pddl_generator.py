from __future__ import annotations

import re
from pathlib import Path

from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


class PDDLGenerator:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = ensure_dir(output_dir or PROJECT_ROOT / "generated" / "pddl")

    def generate(self, problem: PlanningProblem) -> Path:
        if not problem.goal_predicates:
            raise ValueError("goal_predicates cannot be empty")
        init_preds = list(problem.init_predicates)
        init_preds = self._with_smaller_predicates(problem, init_preds)
        objects_block = self._objects_block(problem.types)
        init_block = self._predicates_block(init_preds, indent="    ")
        goal_block = self._predicates_block(problem.goal_predicates, indent="      ")
        text = f"""(define (problem {problem.problem_name})
  (:domain {problem.domain_name})
  (:objects
{objects_block}
  )
  (:init
{init_block}
  )
  (:goal
    (and
{goal_block}
    )
  )
)
"""
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", problem.problem_name)
        path = self.output_dir / f"{safe_name}.pddl"
        path.write_text(text, encoding="utf-8")
        return path

    def _objects_block(self, types: dict[str, str]) -> str:
        by_type: dict[str, list[str]] = {}
        for obj, typ in types.items():
            by_type.setdefault(typ, []).append(obj)
        lines = []
        for typ in sorted(by_type):
            objs = " ".join(sorted(by_type[typ], key=self._natural_key))
            lines.append(f"    {objs} - {typ}")
        return "\n".join(lines)

    def _predicates_block(self, predicates: list[Predicate], indent: str) -> str:
        if not predicates:
            raise ValueError("predicate block cannot be empty")
        return "\n".join(f"{indent}({pred.to_pddl()})" for pred in predicates)

    def _with_smaller_predicates(self, problem: PlanningProblem, init_preds: list[Predicate]) -> list[Predicate]:
        existing = {p.as_tuple() for p in init_preds}
        disks = sorted([obj for obj, typ in problem.types.items() if typ == "disk"], key=self._natural_key)
        pegs = sorted([obj for obj, typ in problem.types.items() if typ == "peg"], key=self._natural_key)
        blocks = sorted([obj for obj, typ in problem.types.items() if typ == "block"], key=self._natural_key)
        tables = sorted([obj for obj, typ in problem.types.items() if typ == "table"], key=self._natural_key)

        if disks:
            for i, disk in enumerate(disks):
                for larger_disk in disks[i + 1:]:
                    init_preds = self._maybe_add_smaller(init_preds, existing, disk, larger_disk)
                for peg in pegs:
                    init_preds = self._maybe_add_smaller(init_preds, existing, disk, peg)
        elif blocks:
            for i, block in enumerate(blocks):
                for larger_block in blocks[i + 1:]:
                    init_preds = self._maybe_add_smaller(init_preds, existing, block, larger_block)
                for table in tables:
                    init_preds = self._maybe_add_smaller(init_preds, existing, block, table)
        return init_preds

    @staticmethod
    def _maybe_add_smaller(init_preds, existing, obj_a, obj_b):
        pred = Predicate("smaller", [obj_a, obj_b])
        if pred.as_tuple() not in existing:
            init_preds.append(pred)
            existing.add(pred.as_tuple())
        return init_preds

    @staticmethod
    def _natural_key(value: str):
        return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", value)]
