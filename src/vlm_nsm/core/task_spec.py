"""Task Specification — declarative task definition.

Replaces hardcoded task logic with configurable YAML/JSON specs.
New tasks (Hanoi, drawer) require only a config file, not code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskSpec:
    """Complete declarative definition of a manipulation task.

    Example (stack 2-block):
        TaskSpec(
            name="stack_2block",
            env_name="Stack",
            env_kwargs={"cube_names": ["cube1", "cube2"], "robots": "Panda"},
            planner="internal_stacking",
            n_objects=2,
            skills=["reach", "grasp", "lift", "reach_place", "release_place", "retreat"],
            object_map={"cube1": 1, "cube2": 2, "table": 0},
            goal_predicates=[["on", "cube1", "cube2"]],
            target_stack=["cube2", "cube1"],
        )
    """

    name: str
    env_name: str = "Stack"
    env_kwargs: dict = field(default_factory=dict)
    planner: str = "bfs"
    n_objects: int = 2
    skills: list[str] = field(default_factory=lambda: [
        "reach", "grasp", "lift", "reach_place", "release_place", "retreat",
    ])
    object_map: dict[str, int] = field(default_factory=dict)
    goal_predicates: list[list] = field(default_factory=list)
    target_stack: list[str] = field(default_factory=list)
    horizon: int = 1000
    early_stop_on_goal: bool = True
    controller: str = "pid"
    monitor_enabled: bool = False
    max_replans: int = 3
    execution_config: dict = field(default_factory=dict)
    # Multi-task support (2026-06-18)
    action_handler: str = "move"  # "move" (stacking) or "pick_place" (Hanoi)
    planner_kwargs: dict = field(default_factory=dict)  # extra args for planner.solve()
    init_hook: Any = None  # callable (env, task) -> None for custom initial state


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load a TaskSpec from a YAML or JSON file.

    Args:
        path: Path to .yaml or .json config file.

    Returns:
        TaskSpec instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        with open(path) as f:
            data = json.load(f)

    return TaskSpec(**data)


def save_task_spec(task: TaskSpec, path: str | Path):
    """Save a TaskSpec to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in task.__dict__.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Pre-defined Task Specs ───────────────────────────────────────────────────

def get_stacking_2block_spec() -> TaskSpec:
    """Standard 2-block stacking task (red on green)."""
    return TaskSpec(
        name="stack_2block",
        env_name="Stack",
        env_kwargs={"cube_names": ["cube1", "cube2"], "robots": "Panda"},
        planner="internal_stacking",
        n_objects=2,
        skills=["reach", "grasp", "lift", "reach_place", "release_place", "retreat"],
        object_map={"cube1": 1, "cube2": 2, "table": 0},
        goal_predicates=[["on", "cube1", "cube2"]],
        target_stack=["cube2", "cube1"],
        execution_config={
            "kp": 0.05, "max_delta": 0.02,
            "holding_check_enabled": True,
            "lift_early_exit_enabled": True,
            "place_mode": "simple",
        },
    )


def get_hanoi_3disk_spec() -> TaskSpec:
    """3-disk Hanoi tower task."""
    return TaskSpec(
        name="hanoi_3disk",
        env_name="Stack",
        env_kwargs={"cube_names": ["cube1", "cube2", "cube3"], "robots": "Panda"},
        planner="internal_hanoi",
        n_objects=3,
        skills=["reach", "grasp", "lift", "reach_place", "release_place", "retreat"],
        object_map={"cube1": 1, "cube2": 2, "cube3": 3, "table": 0},
        goal_predicates=[
            ["on", "cube1", "cube2"],
            ["on", "cube2", "cube3"],
            ["on", "cube3", "table"],
        ],
        target_stack=["cube3", "cube2", "cube1"],
        monitor_enabled=True,
        max_replans=3,
        execution_config={
            "kp": 0.05, "max_delta": 0.02,
            "holding_check_enabled": True,
            "lift_early_exit_enabled": True,
        },
    )
