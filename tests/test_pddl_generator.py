from pathlib import Path

from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.experiments.run_hanoi_symbolic import build_hanoi_problem


def test_hanoi_pddl_generator_3_4_5(tmp_path):
    for n in [3, 4, 5]:
        disks, pegs, objects, types, init = build_hanoi_problem(n)
        goals = [Predicate("on", [disks[i], disks[i + 1]]) for i in range(n - 1)] + [Predicate("on", [disks[-1], "peg3"])]
        path = PDDLGenerator(tmp_path).generate(PlanningProblem("hanoi", objects, types, init, goals, f"hanoi_{n}"))
        text = Path(path).read_text()
        assert "(:objects" in text
        assert "(:init" in text
        assert "(:goal" in text
        assert "(and" in text
        assert (("cube4" in text) if n >= 4 else ("cube4" not in text))
        assert (("cube5" in text) if n >= 5 else ("cube5" not in text))
        assert "(on" in text
