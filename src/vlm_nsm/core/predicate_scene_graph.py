"""PredicateSceneGraph — extended symbolic scene graph with robot state,
task context, and serialization for LLM prompts and verification.

Extends SymbolicSceneGraph (predicates.py) with:
  - Robot state: eef_pose, gripper_open, holding
  - Task context: available_skills, known_supports, forbidden_relations
  - Builder from environment/oracle pose
  - JSON and LLM prompt-friendly text serialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlm_nsm.symbolic.predicates import (
    SymbolicSceneGraph,
    Predicate,
    ObjectSymbol,
    on,
    ontable,
    clear,
    holding,
    handempty,
    graspable,
)


# ── Robot State ──────────────────────────────────────────────────────────────

@dataclass
class RobotState:
    """Current state of the robot end-effector."""
    eef_x: float = 0.0
    eef_y: float = 0.0
    eef_z: float = 0.0
    gripper_open: float = 1.0  # 1.0 = open, 0.0 = closed
    holding_object: str | None = None  # object_id or None

    @property
    def gripper_empty(self) -> bool:
        return self.holding_object is None

    @property
    def eef_pose(self) -> dict[str, float]:
        return {"x": self.eef_x, "y": self.eef_y, "z": self.eef_z}

    def to_dict(self) -> dict:
        return {
            "eef_pose": self.eef_pose,
            "gripper_open": self.gripper_open,
            "holding_object": self.holding_object,
        }


# ── Task Context ─────────────────────────────────────────────────────────────

@dataclass
class TaskContext:
    """Task-level metadata for planning and verification."""
    task_name: str = ""
    available_skills: list[str] = field(default_factory=lambda: [
        "reach", "grasp", "lift", "reach_place", "release_place", "retreat",
    ])
    known_supports: list[str] = field(default_factory=lambda: ["table"])
    forbidden_relations: list[str] = field(default_factory=list)
    domain_predicates: list[str] = field(default_factory=lambda: [
        "on", "ontable", "clear", "holding", "handempty", "graspable",
    ])
    max_replans: int = 3
    horizon: int = 1000

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "available_skills": self.available_skills,
            "known_supports": self.known_supports,
            "forbidden_relations": self.forbidden_relations,
            "domain_predicates": self.domain_predicates,
            "max_replans": self.max_replans,
            "horizon": self.horizon,
        }


# ── PredicateSceneGraph ──────────────────────────────────────────────────────

class PredicateSceneGraph(SymbolicSceneGraph):
    """Extended scene graph with robot state and task context.

    Usage:
        psg = PredicateSceneGraph()
        psg.add_object("cube1", ObjectSymbol(...))
        psg.add_predicate(on("cube1", "cube2"))
        psg.set_robot(eef_pose, gripper_open, holding_object)
        psg.set_task_context(TaskContext(...))

        # Serialization
        print(psg.to_prompt_text())  # LLM-friendly text
        print(psg.to_json())         # JSON for logging
    """

    def __init__(self):
        super().__init__()
        self.robot = RobotState()
        self.task_context = TaskContext()

    # ── Robot state setters ──────────────────────────────────────────────

    def set_robot(self, eef_pose: dict | None = None,
                  gripper_open: float | None = None,
                  holding_object: str | None = None):
        """Update robot state from environment observation."""
        if eef_pose is not None:
            self.robot.eef_x = eef_pose["x"]
            self.robot.eef_y = eef_pose["y"]
            self.robot.eef_z = eef_pose["z"]
        if gripper_open is not None:
            self.robot.gripper_open = gripper_open
        if holding_object is not None:
            self.robot.holding_object = holding_object

    def set_task_context(self, ctx: TaskContext):
        self.task_context = ctx

    # ── Convenience builders ─────────────────────────────────────────────

    def add_object(self, object_id: str, obj_type: str = "cube",
                   color: str = "", size: float | None = None):
        """Add an object to the scene graph."""
        self.objects[object_id] = ObjectSymbol(
            object_id=object_id, obj_type=obj_type,
            color=color, size=size,
        )

    def add_predicate(self, p: Predicate):
        """Add a predicate to the scene graph."""
        self.predicates.add(p)

    def add_relation(self, rel_type: str, obj: str, support: str = ""):
        """Add a binary or unary relation predicate."""
        if rel_type == "on":
            self.predicates.add(on(obj, support))
        elif rel_type == "clear":
            self.predicates.add(clear(obj))
        elif rel_type == "ontable":
            self.predicates.add(ontable(obj))
        elif rel_type == "holding":
            self.predicates.add(holding(obj))
        elif rel_type == "handempty":
            self.predicates.add(handempty())
        elif rel_type == "graspable":
            self.predicates.add(graspable(obj))
        else:
            args = (obj, support) if support else (obj,)
            self.predicates.add(Predicate(rel_type, args))

    def add_robot_predicates(self):
        """Synchronize robot state to robot-related predicates."""
        # Clear old robot predicates
        to_remove = set()
        for p in self.predicates:
            if p.name in ("holding", "handempty"):
                to_remove.add(p)
        self.predicates -= to_remove

        if self.robot.holding_object:
            self.predicates.add(holding(self.robot.holding_object))
        else:
            self.predicates.add(handempty())

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Full serialization including robot and task context."""
        base = super().to_dict()
        base["robot"] = self.robot.to_dict()
        base["task_context"] = self.task_context.to_dict()
        return base

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_prompt_text(self) -> str:
        """Generate LLM-friendly prompt description of the scene.

        Format:
            Scene: 2-block stacking
            Objects:
              - cube1 (red cube, 4cm) at (0.05, 0.03, 0.83)
              - cube2 (green cube, 5cm) at (0.05, 0.03, 0.88)
            Relations:
              - cube1 is on cube2
              - cube2 is on table
              - cube1 is clear
            Robot:
              - EEF at (-0.11, 0.02, 1.01)
              - Gripper is empty
            Available skills:
              - reach, grasp, lift, reach_place, release_place, retreat
            Allowed predicates:
              - on, ontable, clear, holding, handempty, graspable
        """
        lines = []

        # Header
        task_name = self.task_context.task_name or "unknown"
        lines.append(f"Scene: {task_name}")

        # Objects
        lines.append("Objects:")
        for obj_id, obj in self.objects.items():
            size_str = f", {obj.size*1000:.0f}mm" if obj.size else ""
            color_str = f"{obj.color} " if obj.color else ""
            lines.append(f"  - {obj_id} ({color_str}{obj.obj_type}{size_str})")

        # Relations
        lines.append("Relations:")
        for p in sorted(self.predicates, key=str):
            if p.name in ("graspable", "reachable", "stable", "perception_available"):
                continue  # skip technical predicates
            args_str = ", ".join(p.args)
            if len(p.args) == 0:
                lines.append(f"  - {p.name}()")
            elif len(p.args) == 1:
                lines.append(f"  - {p.args[0]} is {p.name}")
            elif p.name == "on":
                lines.append(f"  - {p.args[0]} is on {p.args[1]}")
            elif p.name == "clear":
                lines.append(f"  - {p.args[0]} is clear")
            else:
                lines.append(f"  - {p.name}({args_str})")

        # Robot
        lines.append("Robot:")
        lines.append(f"  - EEF at ({self.robot.eef_x:.2f}, {self.robot.eef_y:.2f}, {self.robot.eef_z:.2f})")
        if self.robot.holding_object:
            lines.append(f"  - Holding {self.robot.holding_object}")
        else:
            lines.append(f"  - Gripper is empty")

        # Task context
        lines.append("Available skills:")
        lines.append(f"  - {', '.join(self.task_context.available_skills)}")
        lines.append("Allowed predicates:")
        lines.append(f"  - {', '.join(self.task_context.domain_predicates)}")
        if self.task_context.forbidden_relations:
            lines.append(f"Forbidden: {', '.join(self.task_context.forbidden_relations)}")

        return "\n".join(lines)


# ── Builder from Environment ─────────────────────────────────────────────────

def build_predicate_scene_graph(
    env,
    cubes: list[str],
    task_name: str = "stacking",
    available_skills: list[str] | None = None,
) -> PredicateSceneGraph:
    """Build a PredicateSceneGraph from the current environment state.

    Uses oracle object poses from robosuite env + predicate extraction.

    Args:
        env: RobosuiteEnvWrapper instance.
        cubes: List of cube names (e.g. ["cube1", "cube2"]).
        task_name: Task identifier.
        available_skills: Skills for this task.

    Returns:
        PredicateSceneGraph populated from current env state.
    """
    from vlm_nsm.core.physical_predicate_extractor import (
        physical_state_to_predicates,
    )

    psg = PredicateSceneGraph()

    # Objects from env
    try:
        all_poses = env.get_all_object_poses()
    except Exception:
        all_poses = {}

    for obj_id in cubes:
        pose = all_poses.get(obj_id, {"x": 0, "y": 0, "z": 0})
        # Determine object properties from known registry
        color = "red" if "cube1" in obj_id or "1" in obj_id else \
                "green" if "cube2" in obj_id or "2" in obj_id else \
                "blue" if "cube3" in obj_id or "3" in obj_id else ""
        size = 0.04 if "1" in obj_id else 0.05 if "2" in obj_id else 0.04
        psg.add_object(obj_id, obj_type="cube", color=color, size=size)

    # Predicates from physical state
    try:
        targets = dict(all_poses)
        targets["table"] = env.get_object_pose("table")
        thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
        physical_predicates = physical_state_to_predicates(
            all_poses, targets, thresholds)
        for p in physical_predicates:
            psg.add_predicate(Predicate(p[0], tuple(p[1:])))
    except Exception:
        pass

    # Robot state
    try:
        eef = env.get_eef_pose()
        grip = env.get_gripper_state()
        psg.set_robot(eef, grip.get("open", 1.0))
    except Exception:
        pass

    # Add robot-related predicates
    psg.add_robot_predicates()

    # Basic predicate completion
    for obj_id in cubes:
        psg.add_predicate(graspable(obj_id))
        # If not supporting anything and not supported, it's on table
        has_support = any(
            p.name == "on" and p.args[0] == obj_id for p in psg.predicates)
        if not has_support:
            psg.add_predicate(ontable(obj_id))

    # Task context
    ctx = TaskContext(
        task_name=task_name,
        available_skills=available_skills or [
            "reach", "grasp", "lift", "reach_place", "release_place", "retreat",
        ],
    )
    psg.set_task_context(ctx)

    return psg


# ── Builder from Existing SymbolicSceneGraph ─────────────────────────────────

def from_symbolic_scene_graph(
    sg: SymbolicSceneGraph,
    robot: RobotState | None = None,
    task_context: TaskContext | None = None,
) -> PredicateSceneGraph:
    """Upgrade an existing SymbolicSceneGraph to PredicateSceneGraph.

    Args:
        sg: Existing symbolic scene graph.
        robot: Optional robot state.
        task_context: Optional task context.

    Returns:
        PredicateSceneGraph with all predicates and objects copied.
    """
    psg = PredicateSceneGraph()
    for k, v in sg.objects.items():
        if isinstance(v, ObjectSymbol):
            psg.objects[k] = ObjectSymbol(
                object_id=v.object_id, color=v.color,
                obj_type=v.obj_type, size=v.size, pose_source=v.pose_source)
        else:
            # Handle mock objects from tests
            d = v.__dict__ if hasattr(v, '__dict__') else {}
            psg.objects[k] = ObjectSymbol(
                object_id=d.get('object_id', k),
                color=d.get('color', ''),
                obj_type=d.get('obj_type', 'cube'),
                size=d.get('size', None),
                pose_source=d.get('pose_source', 'unknown'))
    psg.predicates = set(sg.predicates)
    if robot is not None:
        psg.robot = robot
    if task_context is not None:
        psg.task_context = task_context
    return psg
