"""Option DSL — restricted task description language.

LLM outputs must conform to this schema. No arbitrary Python allowed.

TaskProgram structure:
  {
    "version": "1.0",
    "task_type": "stacking" | "hanoi",
    "goal": [["on", "cube1", "cube2"]],
    "subgoals": [
      [["clear", "cube2"]],
      [["on", "cube1", "cube2"]]
    ],
    "constraints": [
      "pick cube1 before cube2",
      "max 3 retries on grasp"
    ],
    "steps": [
      {"type": "achieve", "goal": [["clear", "cube2"]]},
      {"type": "ensure", "predicate": ["graspable", "cube1"]},
      {"type": "action", "option": "move", "params": {"obj": "cube1", "target": "cube2"}},
      {"type": "retry", "option": "grasp", "params": {"obj": "cube1"}, "max_retries": 2, "strategy": "xy_realign"},
      {"type": "verify", "predicate": ["on", "cube1", "cube2"]}
    ]
  }
"""
@dataclass
class TaskStep:
    type: str  # "achieve" | "ensure" | "action" | "retry" | "verify"
    option: str = ""          # skill name (for action/retry)
    params: dict = field(default_factory=dict)  # {"obj": "cube1", ...}
    goal: list = field(default_factory=list)    # for achieve
    predicate: list = field(default_factory=list)  # for ensure/verify
    max_retries: int = 1
    strategy: str = "reset"

@dataclass 
class TaskProgram:
    version: str = "1.0"
    task_type: str = "stacking"
    goal: list = field(default_factory=list)
    subgoals: list = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    steps: list[TaskStep] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> "TaskProgram":
        """Parse from LLM JSON output."""
        ...

    def to_goal_only(self) -> dict:
        """Fallback: extract just the goal predicates for BFS planner."""
        return {"goal_predicates": self.goal}
