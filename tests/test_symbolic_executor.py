from vlm_nsm.core.data_types import Action, Predicate
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor
from vlm_nsm.experiments.run_hanoi_symbolic import build_hanoi_problem


def test_symbolic_executor_hanoi_3_manual_plan():
    disks, pegs, objects, types, init = build_hanoi_problem(3)
    init = PDDLGenerator()._with_smaller_predicates(type("P", (), {"types": types})(), list(init))
    goals = [Predicate("on", ["cube1", "cube2"]), Predicate("on", ["cube2", "cube3"]), Predicate("on", ["cube3", "peg3"])]
    actions = [
        Action("pick", ["cube1", "cube2"]), Action("place", ["cube1", "peg3"]),
        Action("pick", ["cube2", "cube3"]), Action("place", ["cube2", "peg2"]),
        Action("pick", ["cube1", "peg3"]), Action("place", ["cube1", "cube2"]),
        Action("pick", ["cube3", "peg1"]), Action("place", ["cube3", "peg3"]),
        Action("pick", ["cube1", "cube2"]), Action("place", ["cube1", "peg1"]),
        Action("pick", ["cube2", "peg2"]), Action("place", ["cube2", "cube3"]),
        Action("pick", ["cube1", "peg1"]), Action("place", ["cube1", "cube2"]),
    ]
    ex = SymbolicStateExecutor()
    ex.reset([p.as_tuple() for p in init])
    result = ex.execute_plan(actions)
    assert result.success, result.error
    assert goal_satisfied(result.final_state, goals)
