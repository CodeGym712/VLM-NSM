"""Behavior Tree Executor — structured control flow for skill execution.

Replaces linear option sequences with BT nodes:
  - Sequence: execute children in order, fail if any child fails
  - Fallback: execute children until one succeeds
  - Condition: check a predicate, SUCCESS if true, FAILURE if false
  - Action: wrap a skill execution (via SkillLibrary)
  - Retry: retry a child action up to N times
  - Verify: check expected effects after an action

Node status: SUCCESS, FAILURE, RUNNING (reserved for async, unused currently)

TICK protocol: each node.tick(env, context) → SUCCESS | FAILURE

Usage:
    from vlm_nsm.embodied.behavior_tree import (
        Sequence, Fallback, Condition, Action, Retry, Verify,
        BTStatus, BTTrace,
    )

    tree = Sequence(children=[
        Condition("clear", "cube2"),
        Action("reach", target_object="cube1"),
        Action("grasp", target_object="cube1"),
        Action("lift", target_object="cube1"),
        Action("reach_place", target_object="cube1", target_location="cube2"),
        Action("release_place", target_object="cube1", target_location="cube2"),
        Verify("on", "cube1", "cube2"),
    ])

    trace = BTTrace()
    status = tree.tick(env, pid, skill_library, trace)
    print(trace.summary())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Status ───────────────────────────────────────────────────────────────────

class BTStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"  # reserved


# ── Trace ─────────────────────────────────────────────────────────────────────

@dataclass
class BTNodeTrace:
    """Single node execution record."""
    node_type: str = ""
    node_name: str = ""
    status: BTStatus = BTStatus.FAILURE
    steps: int = 0
    failure_type: str | None = None
    pre_predicates: list = field(default_factory=list)
    post_predicates: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class BTTrace:
    """Full behavior tree execution trace."""
    nodes: list[BTNodeTrace] = field(default_factory=list)
    total_steps: int = 0
    final_status: BTStatus = BTStatus.FAILURE

    def record(self, trace: BTNodeTrace):
        self.nodes.append(trace)
        self.total_steps += trace.steps

    def summary(self) -> str:
        lines = [f"BT Trace: {len(self.nodes)} nodes, {self.total_steps} steps, {self.final_status.value}"]
        for n in self.nodes:
            status = "OK" if n.status == BTStatus.SUCCESS else "FAIL"
            lines.append(f"  [{status}] {n.node_type}:{n.node_name} "
                         f"steps={n.steps} fail={n.failure_type}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "total_steps": self.total_steps,
            "final_status": self.final_status.value,
            "nodes": [
                {
                    "type": n.node_type, "name": n.node_name,
                    "status": n.status.value, "steps": n.steps,
                    "failure_type": n.failure_type,
                }
                for n in self.nodes
            ],
        }


# ── BT Nodes ─────────────────────────────────────────────────────────────────

class BTNode(ABC):
    """Abstract behavior tree node."""

    @abstractmethod
    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        """Execute node. Returns SUCCESS or FAILURE."""


class Sequence(BTNode):
    """Execute children in order. Returns FAILURE if any child fails."""

    def __init__(self, children: list[BTNode], name: str = "Sequence"):
        self.children = children
        self.name = name

    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        for child in self.children:
            status = child.tick(env, pid, skill_library, trace)
            if status == BTStatus.FAILURE:
                return BTStatus.FAILURE
        return BTStatus.SUCCESS


class Fallback(BTNode):
    """Execute children until one succeeds. Returns FAILURE if all fail."""

    def __init__(self, children: list[BTNode], name: str = "Fallback"):
        self.children = children
        self.name = name

    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        for child in self.children:
            status = child.tick(env, pid, skill_library, trace)
            if status == BTStatus.SUCCESS:
                return BTStatus.SUCCESS
        return BTStatus.FAILURE


class Condition(BTNode):
    """Check a predicate in the current environment state.
    SUCCESS if predicate holds, FAILURE otherwise.
    """

    def __init__(self, pred_name: str, *args: str, name: str = ""):
        self.pred_name = pred_name
        self.args = list(args)
        self.name = name or f"Condition({pred_name})"

    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        t = BTNodeTrace(node_type="Condition", node_name=self.name)
        try:
            from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
            from vlm_nsm.core.predicate_monitor import goal_satisfied

            poses = env.get_all_object_poses()
            targets = dict(poses)
            try:
                targets["table"] = env.get_object_pose("table")
            except Exception:
                pass
            physical = physical_state_to_predicates(
                poses, targets, {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05})
            pred = [self.pred_name] + self.args
            ok = bool(goal_satisfied(physical, [pred], mode="symbolic"))
            t.status = BTStatus.SUCCESS if ok else BTStatus.FAILURE
        except Exception as e:
            t.status = BTStatus.FAILURE
            t.failure_type = f"cond_error:{e}"
        trace.record(t)
        return t.status


class Action(BTNode):
    """Execute a single skill via SkillLibrary."""

    def __init__(self, skill_name: str, name: str = "", **params):
        self.skill_name = skill_name
        self.params = params
        self.name = name or f"Action({skill_name})"

    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        t = BTNodeTrace(node_type="Action", node_name=self.name)

        try:
            skill = skill_library.get(self.skill_name)
        except KeyError:
            t.status = BTStatus.FAILURE
            t.failure_type = f"unknown_skill:{self.skill_name}"
            trace.record(t)
            return BTStatus.FAILURE

        from vlm_nsm.embodied.skills.base_skill import SkillParams
        params = SkillParams(
            target_object=self.params.get("target_object", ""),
            target_location=self.params.get("target_location", ""),
            approach_height=self.params.get("approach_height", 0.10),
            gripper_cmd=self.params.get("gripper_cmd", -1.0),
            max_steps=self.params.get("max_steps", 800),
        )

        # Capture pre-state
        try:
            from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
            poses = env.get_all_object_poses()
            targets = dict(poses)
            try: targets["table"] = env.get_object_pose("table")
            except: pass
            physical = physical_state_to_predicates(poses, targets,
                {"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
            t.pre_predicates = [list(p) for p in physical]
        except Exception:
            pass

        result = skill.execute_with_pid(env, pid, params)

        t.steps = result.steps
        t.status = BTStatus.SUCCESS if result.success else BTStatus.FAILURE
        t.failure_type = result.failure_type
        if hasattr(result, 'post_predicates'):
            t.post_predicates = result.post_predicates
        trace.record(t)
        return t.status


class Retry(BTNode):
    """Retry a child action up to max_retries times."""

    def __init__(self, child: BTNode, max_retries: int = 3,
                 strategy: str = "reset", name: str = ""):
        self.child = child
        self.max_retries = max_retries
        self.strategy = strategy
        self.name = name or f"Retry(x{max_retries})"

    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        for attempt in range(self.max_retries):
            status = self.child.tick(env, pid, skill_library, trace)
            if status == BTStatus.SUCCESS:
                return BTStatus.SUCCESS
            if attempt < self.max_retries - 1:
                trace.record(BTNodeTrace(
                    node_type="Retry", node_name=f"{self.name}#{attempt+1}",
                    status=BTStatus.FAILURE,
                    metadata={"attempt": attempt + 1, "strategy": self.strategy},
                ))
        return BTStatus.FAILURE


class Verify(BTNode):
    """Check expected effects after an action. Like Condition but
    with richer failure reporting.
    """

    def __init__(self, pred_name: str, *args: str, name: str = ""):
        self.pred_name = pred_name
        self.args = list(args)
        self.name = name or f"Verify({pred_name})"

    def tick(self, env, pid, skill_library, trace: BTTrace) -> BTStatus:
        t = BTNodeTrace(node_type="Verify", node_name=self.name)
        try:
            from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
            from vlm_nsm.core.predicate_monitor import goal_satisfied

            poses = env.get_all_object_poses()
            targets = dict(poses)
            try:
                targets["table"] = env.get_object_pose("table")
            except Exception:
                pass
            physical = physical_state_to_predicates(
                poses, targets, {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05})
            pred = [self.pred_name] + self.args
            ok = bool(goal_satisfied(physical, [pred], mode="symbolic"))
            t.post_predicates = [list(p) for p in physical]
            t.status = BTStatus.SUCCESS if ok else BTStatus.FAILURE
            if not ok:
                t.failure_type = f"verify_failed:{self.pred_name}"
        except Exception as e:
            t.status = BTStatus.FAILURE
            t.failure_type = f"verify_error:{e}"
        trace.record(t)
        return t.status


# ── Factory: Standard 2-Block Tree ───────────────────────────────────────────

def stacking_2block_tree(pick: str = "cube1", place: str = "cube2") -> BTNode:
    """Build a standard behavior tree for 2-block stacking.

    Structure:
      Sequence(
        Action("reach", obj=pick),
        Action("grasp", obj=pick),
        Action("lift", obj=pick),
        Action("reach_place", obj=pick, target=place),
        Fallback(
          Sequence(
            Action("release_place", obj=pick, target=place),
            Verify("on", pick, place),
          ),
          Sequence(           # on failure: regrasp + replan
            Action("retreat"),
            Action("reach", obj=pick),
            Retry(Action("grasp", obj=pick), max_retries=2),
            Action("lift", obj=pick),
            Action("reach_place", obj=pick, target=place),
            Action("release_place", obj=pick, target=place),
          ),
        ),
      )
    """
    return Sequence(children=[
        Action("reach", target_object=pick, name=f"reach_{pick}"),
        Action("grasp", target_object=pick, name=f"grasp_{pick}"),
        Action("lift", target_object=pick, name=f"lift_{pick}"),
        Action("reach_place", target_object=pick,
               target_location=place, name=f"reach_place_{pick}"),
        Fallback(children=[
            Sequence(children=[
                Action("release_place", target_object=pick,
                       target_location=place, name=f"release_{pick}"),
                Verify("on", pick, place, name=f"verify_on({pick},{place})"),
            ], name="PlaceAndVerify"),
            Sequence(children=[
                Action("retreat", name="retreat_for_retry"),
                Action("reach", target_object=pick, name=f"re_reach_{pick}"),
                Retry(Action("grasp", target_object=pick, name=f"re_grasp_{pick}"),
                      max_retries=2, strategy="continue", name="RetryGrasp"),
                Action("lift", target_object=pick, name=f"re_lift_{pick}"),
                Action("reach_place", target_object=pick,
                       target_location=place, name=f"re_reach_place_{pick}"),
                Action("release_place", target_object=pick,
                       target_location=place, name=f"re_release_{pick}"),
            ], name="RecoverAndRetry"),
        ], name="PlaceWithFallback"),
    ], name="Stack2Block")
