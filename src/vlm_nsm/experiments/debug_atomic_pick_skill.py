"""Atomic Pick Skill — A/B/C comparison experiment.

Compares three pick paths on 2-block pick-only task (cube1):
  A. DirectPick: direct PID reach + execute_grasp + lift (proven)
  B. DecomposedSkillPick: ReachSkill + GraspSkill + LiftSkill
  C. AtomicPickSkill: single option with reach+grasp+lift inline
"""

import os, json, math
import numpy as np

os.environ['MUJOCO_GL'] = 'egl'

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
from vlm_nsm.embodied.skills.base_skill import SkillLibrary, SkillParams
from vlm_nsm.embodied.skills.reach_skill import ReachSkill
from vlm_nsm.embodied.skills.grasp_skill import GraspSkill
from vlm_nsm.embodied.skills.lift_skill import LiftSkill
from vlm_nsm.embodied.skills.atomic_pick_skill import AtomicPickSkill


def _dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


def direct_pick(env, pid, pick):
    """Path A: Direct PID reach + execute_grasp + lift."""
    total_steps = 0
    diag = {}

    # reach
    obj = env.get_object_pose(pick)
    target = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.10}
    pid.set_phase('reach_pick'); pid.set_target(pos=target)
    for _ in range(800):
        e = env.get_eef_pose(); pid.set_current(e)
        a = pid.act('reach_pick', None); a[6] = -1.0
        try: env.step(a)
        except: return {'success': False, 'failure_subphase': 'reach', 'steps': total_steps}
        total_steps += 1
        if _dist(e, target) < 0.02: break
    diag['reach_steps'] = total_steps

    # grasp
    g = execute_grasp(env, pick,
        config=GraspConfig(kp=0.05, max_delta=0.01, close_steps=70, max_retries=2, holding_check_enabled=True),
        skip_approach=True, skip_lift=True)
    total_steps += g.get('total_step_count', 0)
    if not g['success']:
        return {'success': False, 'failure_subphase': 'grasp', 'failure_type': g.get('failure_type'), 'steps': total_steps}

    # lift
    obj = env.get_object_pose(pick)
    target = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.15}
    pid.set_phase('lift'); pid.set_target(pos=target)
    for _ in range(800):
        e = env.get_eef_pose(); pid.set_current(e)
        a = pid.act('lift', None); a[6] = 1.0
        try: env.step(a)
        except: return {'success': False, 'failure_subphase': 'lift', 'steps': total_steps}
        total_steps += 1
        if _dist(e, target) < 0.02: break

    obj_after = env.get_object_pose(pick)
    grip_after = env.get_gripper_state()
    diag['object_z_delta'] = obj_after['z'] - obj['z']
    diag['holding'] = grip_after.get('has_object', False) or (grip_after.get('open', 1) > 0.003)
    diag['steps'] = total_steps
    return {'success': True, 'steps': total_steps, 'diagnostics': diag}


def decomposed_pick(env, pid, lib, pick):
    """Path B: ReachSkill + GraspSkill + LiftSkill."""
    total_steps = 0
    for name in ['reach', 'grasp', 'lift']:
        s = lib.get(name)
        r = s.execute_with_pid(env, pid, SkillParams(target_object=pick))
        total_steps += r.steps
        if not r.success:
            return {'success': False, 'failure_subphase': name,
                    'failure_type': r.failure_type, 'steps': total_steps}
    # Check holding
    obj = env.get_object_pose(pick)
    grip = env.get_gripper_state()
    return {'success': True, 'steps': total_steps,
            'diagnostics': {'object_z_delta': 0, 'holding': grip.get('has_object', False)}}


def atomic_pick(env, pid, lib, pick):
    """Path C: AtomicPickSkill."""
    s = lib.get('atomic_pick')
    r = s.execute_with_pid(env, pid, SkillParams(target_object=pick))
    holding = r.diagnostics.get('holding_after_lift', False) if hasattr(r, 'diagnostics') else False
    return {'success': r.success, 'steps': r.steps,
            'failure_subphase': r.diagnostics.get('failure_subphase') if hasattr(r, 'diagnostics') else None,
            'failure_type': r.failure_type,
            'diagnostics': r.diagnostics if hasattr(r, 'diagnostics') else {'holding': holding}}


def main():
    env = RobosuiteEnvWrapper(env_name='Stack', robots='Panda', cube_names=['cube1', 'cube2'], render_mode='none')
    env.env.horizon = 8000

    lib = SkillLibrary()
    for s in [ReachSkill(), GraspSkill(), LiftSkill(), AtomicPickSkill()]:
        lib.register(s)

    N = 20
    results = {'direct': [], 'decomposed': [], 'atomic': []}

    for ep in range(N):
        # Path A
        env.reset(); pid = OperationalSpacePID()
        results['direct'].append(direct_pick(env, pid, 'cube1'))

        # Path B
        env.reset(); pid = OperationalSpacePID()
        results['decomposed'].append(decomposed_pick(env, pid, lib, 'cube1'))

        # Path C
        env.reset(); pid = OperationalSpacePID()
        results['atomic'].append(atomic_pick(env, pid, lib, 'cube1'))

        if (ep+1) % 5 == 0:
            da = sum(1 for r in results['direct'] if r['success'])/len(results['direct'])
            db = sum(1 for r in results['decomposed'] if r['success'])/len(results['decomposed'])
            dc = sum(1 for r in results['atomic'] if r['success'])/len(results['atomic'])
            print(f"  [{ep+1}/20] direct={da:.0%} decomposed={db:.0%} atomic={dc:.0%}")

    # Summary
    for name, data in [('Direct', results['direct']), ('Decomposed', results['decomposed']), ('Atomic', results['atomic'])]:
        ok = sum(1 for r in data if r['success'])
        failures = {}
        for r in data:
            if not r['success']:
                ft = r.get('failure_subphase', 'unknown')
                failures[ft] = failures.get(ft, 0) + 1
        print(f"\n{name}: {ok}/{N} ({ok/N:.0%}) failures={failures}")

    out = 'generated/results/upper_ns/atomic_pick_debug.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'N': N, 'results': results}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
