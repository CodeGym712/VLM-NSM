"""AtomicPickSkill — reach + grasp + lift as a single option.

Unlike DecomposedSkillPick (ReachSkill + GraspSkill + LiftSkill),
this skill keeps the PID object alive across all three sub-phases,
matching the proven direct PID path structure.

Returns SkillResult with diagnostics per sub-phase.
"""

from __future__ import annotations

import math
import numpy as np

from vlm_nsm.embodied.skills.base_skill import BaseSkill, SkillParams, SkillResult
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


class AtomicPickSkill(BaseSkill):
    """Pick an object: reach → grasp → lift, as one atomic option.

    Uses the SAME PID object throughout, with continuous integral/velocity state.
    This matches the proven direct PID path structure.
    """

    name = "atomic_pick"

    def __init__(self):
        self._diagnostics = {}

    def get_target(self, env, params: SkillParams) -> dict:
        obj = env.get_object_pose(params.target_object)
        return {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        return False  # never used — execute_with_pid is fully overridden

    def execute_with_pid(self, env, pid_ctrl, params: SkillParams) -> SkillResult:
        pick = params.target_object
        total_steps = 0
        diag = {}

        # ── Phase 1: Reach ──
        obj = env.get_object_pose(pick)
        diag['object_z_before_reach'] = obj['z']
        target = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.10}
        pid_ctrl.set_phase('reach_pick')
        pid_ctrl.set_target(pos=target)
        reach_ok = False
        for _ in range(800):
            eef = env.get_eef_pose()
            pid_ctrl.set_current(eef)
            a = pid_ctrl.act('reach_pick', None)
            a[6] = -1.0
            try:
                env.step(a)
            except ValueError:
                return SkillResult(False, total_steps, "env_error",
                                   diagnostics=diag)
            total_steps += 1
            if _dist(eef, target) < 0.01:  # tighter: 10mm
                reach_ok = True
                break
        if not reach_ok:
            diag['failure_subphase'] = 'reach'
            return SkillResult(False, total_steps, "reach_timeout",
                               diagnostics=diag)

        eef_after_reach = env.get_eef_pose()
        diag['xy_error_before_grasp'] = math.hypot(
            obj['x'] - eef_after_reach['x'],
            obj['y'] - eef_after_reach['y'])

        # ── Phase 2: Grasp ──
        grip_before = env.get_gripper_state()
        diag['gripper_open_before'] = grip_before.get('open', -1.0)
        diag['object_z_before_grasp'] = env.get_object_pose(pick)['z']

        g = execute_grasp(
            env, pick,
            config=GraspConfig(
                kp=0.05, max_delta=0.01, descend_offset=0.005,
                close_steps=70, max_retries=2,
                holding_check_enabled=True,
                micro_align_enabled=True,
                micro_xy_tol=0.005, micro_max_delta=0.006,
                micro_align_max_steps=60,
            ),
            skip_approach=True,
            skip_lift=True,
        )
        total_steps += g.get('total_step_count', 0)
        diag['grasp_steps'] = g.get('total_step_count', 0)
        grip_after = env.get_gripper_state()
        diag['gripper_open_after'] = grip_after.get('open', -1.0)
        diag['contact_detected'] = g['success']

        if not g['success']:
            diag['failure_subphase'] = 'grasp'
            return SkillResult(False, total_steps,
                               g.get('failure_type', 'grasp_failed'),
                               diagnostics=diag)

        # ── Phase 3: Lift ──
        obj = env.get_object_pose(pick)
        diag['object_z_before_lift'] = obj['z']
        target = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.15}
        pid_ctrl.set_phase('lift')
        pid_ctrl.set_target(pos=target)
        lift_ok = False
        for _ in range(800):
            eef = env.get_eef_pose()
            pid_ctrl.set_current(eef)
            a = pid_ctrl.act('lift', None)
            a[6] = 1.0
            try:
                env.step(a)
            except ValueError:
                return SkillResult(False, total_steps, "env_error",
                                   diagnostics=diag)
            total_steps += 1
            if _dist(eef, target) < 0.02:
                lift_ok = True
                break
        if not lift_ok:
            diag['failure_subphase'] = 'lift'
            return SkillResult(False, total_steps, "lift_timeout",
                               diagnostics=diag)

        obj_after_lift = env.get_object_pose(pick)
        diag['object_z_after'] = obj_after_lift['z']
        diag['object_z_delta'] = obj_after_lift['z'] - diag['object_z_before_lift']
        diag['eef_z_lift_delta'] = env.get_eef_pose()['z'] - eef_after_reach['z']

        # Hold check: did the object actually come up?
        holding = grip_after.get('has_object', False) or \
                  (grip_after.get('open', 1.0) > 0.003 and
                   grip_after.get('open', 1.0) < 0.04)
        diag['holding_after_lift'] = holding and diag['object_z_delta'] > 0.01

        if not diag['holding_after_lift']:
            diag['failure_subphase'] = 'lift_slip'
            return SkillResult(False, total_steps, "slip",
                               diagnostics=diag)

        result = SkillResult(True, total_steps, diagnostics=diag)
        result.pre_predicates = []   # populated by task_runner
        result.post_predicates = []  # populated by task_runner
        result.expected_effect = [
            ["holding", pick],
            ["clear", pick],
        ]
        result.effect_satisfied = True
        return result
