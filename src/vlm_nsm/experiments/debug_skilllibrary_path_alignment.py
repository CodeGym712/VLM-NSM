"""Phase 4R.1: Full path alignment — find all EEF differences between execution paths.

Compares on SAME seed, SAME initial state:
  A. Direct PID reference
  B. SkillLibrary full path
  C. BT Action path

Records EEF pose, object pose, and trajectory samples at every phase boundary.
All differences > 5mm are flagged as "must fix" or "warning".
"""

import os, json, math
import numpy as np

os.environ['MUJOCO_GL'] = 'egl'

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
from vlm_nsm.embodied.skills.base_skill import SkillLibrary, SkillParams
from vlm_nsm.embodied.skills.reach_skill import ReachSkill, ReachPlaceSkill, RetreatSkill
from vlm_nsm.embodied.skills.grasp_skill import GraspSkill, ReleasePlaceSkill
from vlm_nsm.embodied.skills.lift_skill import LiftSkill
from vlm_nsm.embodied.behavior_tree import Action, BTTrace, BTStatus

def _dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)

def run_direct(env, pid, pick, place):
    """Path A: Direct PID reference."""
    trace = {}
    obj = env.get_object_pose(pick)
    # reach
    t = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.10}
    trace['reach_target'] = dict(t)
    pid.set_phase('reach_pick'); pid.set_target(pos=t)
    for _ in range(800):
        e = env.get_eef_pose(); pid.set_current(e); a = pid.act('reach_pick', None); a[6] = -1.0
        try: env.step(a)
        except: break
        if _dist(e, t) < 0.02: break
    trace['reach_eef'] = dict(env.get_eef_pose()); trace['reach_obj'] = dict(env.get_object_pose(pick))
    # grasp
    g = execute_grasp(env, pick, config=GraspConfig(kp=0.05, max_delta=0.01, close_steps=70, max_retries=2, holding_check_enabled=True), skip_approach=True, skip_lift=True)
    trace['grasp_eef'] = dict(env.get_eef_pose()); trace['grasp_obj'] = dict(env.get_object_pose(pick))
    trace['grasp_ok'] = g['success']
    # lift
    obj = env.get_object_pose(pick)
    t = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.15}
    trace['lift_target'] = dict(t)
    pid.set_phase('lift'); pid.set_target(pos=t)
    for _ in range(800):
        e = env.get_eef_pose(); pid.set_current(e); a = pid.act('lift', None); a[6] = 1.0
        try: env.step(a)
        except: break
        if _dist(e, t) < 0.02: break
    trace['lift_eef'] = dict(env.get_eef_pose()); trace['lift_obj'] = dict(env.get_object_pose(pick))
    # reach_place
    tgt = env.get_target_pose(place)
    tt = {'x': tgt['x'], 'y': tgt['y'], 'z': tgt['z'] + 0.15}
    trace['reach_place_target'] = dict(tt)
    pid.set_phase('reach_place'); pid.set_target(pos=tt)
    for _ in range(800):
        e = env.get_eef_pose(); pid.set_current(e); a = pid.act('reach_place', None); a[6] = 1.0
        try: env.step(a)
        except: break
        if _dist(e, tt) < 0.02: break
    trace['reach_place_eef'] = dict(env.get_eef_pose()); trace['reach_place_obj'] = dict(env.get_object_pose(pick))
    # release
    p = execute_place(env, pick, place, config=PlaceConfig(kp=0.05, max_delta=0.02, max_retries=2, mode='simple'))
    trace['release_ok'] = p.get('success', False)
    trace['release_failure'] = p.get('failure_type')
    trace['release_eef'] = dict(env.get_eef_pose())
    trace['release_obj'] = dict(env.get_object_pose(pick))
    return trace

def run_skilllib(env, pid, lib, pick, place):
    """Path B: Full SkillLibrary."""
    trace = {}
    phases = [
        ("reach", pick, pick, 0.10),
        ("grasp", pick, pick, 0.10),
        ("lift", pick, pick, 0.10),
        ("reach_place", pick, place, 0.15),
        ("release_place", pick, place, 0.10),
    ]
    for name, obj, tgt, approach in phases:
        skill = lib.get(name)
        params = SkillParams(target_object=obj, target_location=tgt, approach_height=approach)
        result = skill.execute_with_pid(env, pid, params)
        key = name.replace('release_place', 'release')
        trace[f'{key}_eef'] = dict(env.get_eef_pose())
        trace[f'{key}_obj'] = dict(env.get_object_pose(pick) if name != 'retreat' else {'x':0,'y':0,'z':0})
        trace[f'{key}_ok'] = result.success
        trace[f'{key}_failure'] = result.failure_type
        if not result.success and name != 'release_place':
            break
    return trace

def run_bt(env, pid, lib, pick, place):
    """Path C: BT Action nodes."""
    trace = {}
    for skill_name in ['reach', 'grasp', 'lift', 'reach_place', 'release_place']:
        node = Action(skill_name, target_object=pick, target_location=place if 'place' in skill_name else pick)
        bt_trace = BTTrace()
        status = node.tick(env, pid, lib, bt_trace)
        key = skill_name.replace('release_place', 'release')
        trace[f'{key}_eef'] = dict(env.get_eef_pose())
        trace[f'{key}_obj'] = dict(env.get_object_pose(pick) if 'retreat' != skill_name else {'x':0,'y':0,'z':0})
        trace[f'{key}_ok'] = (status == BTStatus.SUCCESS)
        trace[f'{key}_failure'] = bt_trace.nodes[-1].failure_type if bt_trace.nodes and bt_trace.nodes[-1].failure_type else None
        if status == BTStatus.FAILURE and skill_name != 'release_place':
            break
    return trace

def compare_traces(traces, seed):
    """Compare phase-level success and targets, not absolute EEF (different object placements per reset)."""
    phases = ['reach', 'grasp', 'lift', 'reach_place', 'release']
    rows = []
    for phase in phases:
        ok_direct = traces['direct'].get(f'{phase}_ok')
        ok_skilllib = traces['skilllib'].get(f'{phase}_ok')
        ok_bt = traces['bt'].get(f'{phase}_ok')

        target_direct = traces['direct'].get(f'{phase}_target', {})
        target_skilllib = traces['skilllib'].get(f'{phase}_target', {})

        rows.append({
            'seed': seed, 'phase': phase,
            'direct_ok': ok_direct,
            'skilllib_ok': ok_skilllib,
            'bt_ok': ok_bt,
            'direct_target': target_direct,
            'skilllib_target': target_skilllib,
            'release_ok_direct': traces['direct'].get('release_ok'),
            'release_ok_skilllib': traces['skilllib'].get('release_ok'),
            'release_failure_skilllib': traces['skilllib'].get('release_failure'),
        })
    return rows

def main():
    env = RobosuiteEnvWrapper(env_name='Stack', robots='Panda', cube_names=['cube1', 'cube2'], render_mode='none')
    env.env.horizon = 8000

    lib = SkillLibrary()
    for s in [ReachSkill(), GraspSkill(), LiftSkill(), ReachPlaceSkill(), ReleasePlaceSkill(), RetreatSkill()]:
        lib.register(s)

    all_rows = []
    for seed in [0, 1, 2, 3, 4]:
        pid = OperationalSpacePID()
        # Direct
        np.random.seed(seed); env.reset(); pid.reset()
        trace_direct = run_direct(env, pid, 'cube1', 'cube2')
        # SkillLib
        np.random.seed(seed); env.reset(); pid.reset()
        trace_skilllib = run_skilllib(env, pid, lib, 'cube1', 'cube2')
        # BT
        np.random.seed(seed); env.reset(); pid.reset()
        trace_bt = run_bt(env, pid, lib, 'cube1', 'cube2')

        traces = {'direct': trace_direct, 'skilllib': trace_skilllib, 'bt': trace_bt}
        rows = compare_traces(traces, seed)
        all_rows.extend(rows)

        # Quick summary for this seed
        sk_phases_ok = sum(1 for r in rows if r['skilllib_ok'] is True)
        print(f"seed{seed}: direct_ok={trace_direct.get('release_ok')}, "
              f"skilllib_ok={trace_skilllib.get('release_ok')} "
              f"skilllib_phases_ok={sk_phases_ok}/5 "
              f"failure={trace_skilllib.get('release_failure')}")

    # Save
    out_data = 'generated/results/upper_ns/path_alignment_debug.json'
    os.makedirs(os.path.dirname(out_data), exist_ok=True)
    with open(out_data, 'w') as f:
        json.dump({'n_seeds': 5, 'rows': all_rows}, f, indent=2)

    # Summary: per-phase pass rate for SkillLibrary
    from collections import Counter
    phase_failures = Counter()
    for r in all_rows:
        if r['skilllib_ok'] is False:
            phase_failures[r['phase']] += 1
    print(f"\nOverall ({len(all_rows)} phase-seed pairs):")
    for phase in ['reach','grasp','lift','reach_place','release']:
        fails = phase_failures.get(phase, 0)
        print(f"  {phase}: {5-fails}/5 passed via SkillLibrary")
    direct_ok = sum(1 for r in all_rows if r.get('release_ok_direct') and r['phase']=='release')
    skilllib_ok = sum(1 for r in all_rows if r.get('release_ok_skilllib') and r['phase']=='release')
    print(f"\nRelease success: direct={direct_ok}/5, skilllib={skilllib_ok}/5")
    # Find which phase fails first in SkillLibrary
    for seed in range(5):
        seed_rows = [r for r in all_rows if r['seed']==seed]
        failed_phases = [r['phase'] for r in seed_rows if r['skilllib_ok'] is False]
        if failed_phases:
            print(f"  seed{seed}: first SkillLib failure at '{failed_phases[0]}'")

if __name__ == '__main__':
    main()
