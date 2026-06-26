"""AtomicPlaceSkill — reach_place + release_place as a single option."""

from __future__ import annotations

import math
import numpy as np

from vlm_nsm.embodied.skills.base_skill import BaseSkill, SkillParams, SkillResult
from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


class AtomicPlaceSkill(BaseSkill):
    """Place an object: reach_place → release_place, as one atomic option."""

    name = "atomic_place"

    def get_target(self, env, params: SkillParams) -> dict:
        tgt = env.get_target_pose(params.target_location)
        return {"x": tgt["x"], "y": tgt["y"], "z": tgt["z"] + 0.15}

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        return False

    def execute_with_pid(self, env, pid_ctrl, params: SkillParams) -> SkillResult:
        pick = params.target_object
        place = params.target_location
        total_steps = 0
        diag = {}

        # ── Phase 1: Reach Place ──
        tgt = env.get_target_pose(place)
        target = {'x': tgt['x'], 'y': tgt['y'], 'z': tgt['z'] + 0.15}
        pid_ctrl.set_phase('reach_place')
        pid_ctrl.set_target(pos=target)
        for _ in range(800):
            eef = env.get_eef_pose()
            pid_ctrl.set_current(eef)
            a = pid_ctrl.act('reach_place', None)
            a[6] = 1.0
            try: env.step(a)
            except ValueError:
                return SkillResult(False, total_steps, "env_error", diagnostics=diag)
            total_steps += 1
            if _dist(eef, target) < 0.02: break
        diag['reach_place_steps'] = total_steps

        # ── Phase 2: Release ──
        p = execute_place(env, pick, place,
            config=PlaceConfig(kp=0.05, max_delta=0.02, max_retries=2, mode='simple'),
            skip_preplace=False)
        total_steps += p.get('step_count', 0)
        diag['release_steps'] = p.get('step_count', 0)

        if not p.get('success', False):
            diag['failure_subphase'] = 'release'
            return SkillResult(False, total_steps,
                               p.get('failure_type', 'place_failed'),
                               diagnostics=diag)

        return SkillResult(True, total_steps, diagnostics=diag)
