from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_hanoi_symbolic import build_hanoi_problem


def test_metric_ff_adapter_returns_non_empty_plan(tmp_path):
    disks, pegs, objects, types, init = build_hanoi_problem(3)
    goals = [Predicate("on", [disks[i], disks[i + 1]]) for i in range(2)] + [Predicate("on", [disks[-1], "peg3"])]
    problem = PlanningProblem("hanoi", objects, types, init, goals, "test_hanoi_3")
    problem_path = PDDLGenerator(tmp_path).generate(problem)
    domain_path = PROJECT_ROOT / "domains" / "hanoi" / "domain.pddl"
    result = PlannerAdapter("metric_ff").solve(domain_path, problem_path)
    assert result.success, result.error
    assert result.actions
