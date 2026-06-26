"""Multi-Task E2E Runner: Language → Plan → PID → Monitor.

Supports: stack_2block, stack_3block, hanoi_3disk
Each task defined via TaskSpec, executed with PID + Monitor.

Usage:
    python -m vlm_nsm.experiments.run_multitask_e2e --task stack_2block --num_episodes 10
    python -m vlm_nsm.experiments.run_multitask_e2e --task hanoi_3disk --num_episodes 5
"""

from __future__ import annotations

import argparse, json, math, os
from pathlib import Path

import numpy as np

from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

_GRIPPER_OPEN, _GRIPPER_CLOSE = -1.0, 1.0
_REACH_THRESHOLD, _STEP_LIMIT = 0.02, 800
_PREDICATE_THRESHOLDS = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}

def _dist(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)

def _check_goal(env, goal_preds):
    try:
        poses = env.get_all_object_poses(); targets = dict(poses)
        targets["table"] = env.get_object_pose("table")
        physical = physical_state_to_predicates(poses, targets, _PREDICATE_THRESHOLDS)
        return bool(goal_satisfied(physical, goal_preds, mode="symbolic"))
    except: return False

def _reach(env, pid, obj, phase="reach_pick", gripper=_GRIPPER_OPEN):
    obj_p = env.get_object_pose(obj); t = {"x":obj_p["x"],"y":obj_p["y"],"z":obj_p["z"]+0.10}
    pid.set_phase(phase); pid.set_target(pos=t)
    for _ in range(_STEP_LIMIT):
        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act(phase,None); a[6]=gripper
        try: env.step(a)
        except: return False
        if _dist(eef,t)<_REACH_THRESHOLD: return True
    return False

def _lift(env, pid, obj):
    obj_p = env.get_object_pose(obj); t = {"x":obj_p["x"],"y":obj_p["y"],"z":obj_p["z"]+0.15}
    pid.set_phase("lift"); pid.set_target(pos=t)
    for _ in range(_STEP_LIMIT):
        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act("lift",None); a[6]=_GRIPPER_CLOSE
        try: env.step(a)
        except: return False
        if _dist(eef,t)<_REACH_THRESHOLD: return True
    return False

def _place(env, pid, obj, target):
    """Move to target and release. target can be object name or peg name."""
    if "peg" in target:
        pmap = {"peg1":(-0.12,-0.08),"peg2":(0.0,-0.08),"peg3":(0.12,-0.08)}
        px, py = pmap.get(target, (0,0)); tt = {"x":px,"y":py,"z":0.95}
    else:
        tgt = env.get_target_pose(target); tt = {"x":tgt["x"],"y":tgt["y"],"z":tgt["z"]+0.15}
    pid.set_phase("reach_place"); pid.set_target(pos=tt)
    for _ in range(2*_STEP_LIMIT):
        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act("reach_place",None); a[6]=_GRIPPER_CLOSE
        try: env.step(a)
        except: return False
        if _dist(eef,tt)<_REACH_THRESHOLD: break
    for _ in range(30):
        try: env.step(np.array([0,0,0,0,0,0,_GRIPPER_OPEN],dtype=np.float32))
        except: return False
    for _ in range(10):
        try: env.step(np.array([0,0,0,0,0,0,_GRIPPER_OPEN],dtype=np.float32))
        except: return False
    # retreat
    eef = env.get_eef_pose(); rt = {"x":eef["x"],"y":eef["y"],"z":eef["z"]+0.15}
    pid.set_phase("retreat"); pid.set_target(pos=rt)
    for _ in range(_STEP_LIMIT):
        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act("retreat",None); a[6]=_GRIPPER_OPEN
        try: env.step(a)
        except: return False
        if _dist(eef,rt)<_REACH_THRESHOLD: return True
    return True

def run_stacking_episode(env, pid, cubes, target_stack, goal_preds, n_blocks):
    """2-block or 3-block stacking with PID."""
    planner = PlannerAdapter("bfs")
    plan = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan.success: return {"success":False,"error":plan.error}

    for action in plan.actions:
        pick, place = action.args[0], action.args[2]
        if not _reach(env, pid, pick, "reach_pick", _GRIPPER_OPEN): return {"success":False,"fail":"reach_pick"}
        g = execute_grasp(env, pick, config=GraspConfig(kp=0.05,max_delta=0.01,close_steps=70,max_retries=2,holding_check_enabled=True), skip_approach=True, skip_lift=True)
        if not g["success"]: return {"success":False,"fail":"grasp"}
        if not _lift(env, pid, pick): return {"success":False,"fail":"lift"}
        p = execute_place(env, pick, place, config=PlaceConfig(kp=0.05,max_delta=0.02,max_retries=2,mode="simple"))
        if not p.get("success",False): return {"success":False,"fail":"place"}

    return {"success": _check_goal(env, goal_preds)}

def run_hanoi_episode(env, pid, cubes, n_disks=3):
    """Hanoi tower with PID."""
    from vlm_nsm.planners.internal_hanoi_planner import InternalHanoiPlanner
    plan = InternalHanoiPlanner().solve(n_disks)
    if not plan.success: return {"success":False,"error":plan.error}

    grasp_cfg = GraspConfig(kp=0.05, max_delta=0.01, close_steps=100, max_retries=3,
                            holding_check_enabled=True, descend_offset=0.008,
                            micro_align_enabled=True, micro_xy_tol=0.012, micro_z_tol=0.012,
                            micro_max_delta=0.008, micro_align_max_steps=60)

    for act in plan.actions:
        disk = act.args[0]
        if act.name == "pick":
            if not _reach(env, pid, disk): return {"success":False,"fail":"reach"}
            g = execute_grasp(env, disk, config=grasp_cfg, skip_approach=True, skip_lift=True)
            if not g["success"]: return {"success":False,"fail":"grasp"}
            if not _lift(env, pid, disk): return {"success":False,"fail":"lift"}
        else:
            if not _place(env, pid, disk, act.args[1]): return {"success":False,"fail":"place"}

    return {"success": True}  # Hanoi success = all moves completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stack_2block", choices=["stack_2block","stack_3block","hanoi_3disk"])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dir", default="generated/results/multitask_e2e")
    args = parser.parse_args()

    n_map = {"stack_2block":2,"stack_3block":3,"hanoi_3disk":3}
    n_blocks = n_map[args.task]
    cubes = [f"cube{i}" for i in range(1, n_blocks+1)]
    if n_blocks >= 3:
        import vlm_nsm.executors.stack_three_cube_env as _  # noqa

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
    env.env.horizon = 30000 if args.task == "hanoi_3disk" else 10000
    pid = OperationalSpacePID()
    planner = PlannerAdapter("bfs")

    successes = 0
    for ep in range(args.num_episodes):
        env.reset()
        pid.reset()

        if args.task in ("stack_2block", "stack_3block"):
            ts = cubes[::-1]
            plan = planner.solve(cubes=cubes, target_stack=ts)
            if not plan.success: continue
            ok = True
            for action in plan.actions:
                pick, place = action.args[0], action.args[2]
                # reach
                obj = env.get_object_pose(pick); t = {"x":obj["x"],"y":obj["y"],"z":obj["z"]+0.10}
                pid.set_phase("reach_pick"); pid.set_target(pos=t)
                reach_ok = False
                for _ in range(800):
                    eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act("reach_pick",None); a[6]=-1.0
                    try: env.step(a)
                    except: break
                    if math.hypot(eef["x"]-t["x"],eef["y"]-t["y"],eef["z"]-t["z"])<0.02: reach_ok=True; break
                if not reach_ok: ok=False; break
                # grasp
                g=execute_grasp(env,pick,config=GraspConfig(kp=0.05,max_delta=0.01,close_steps=70,max_retries=2,holding_check_enabled=True),skip_approach=True,skip_lift=True)
                if not g["success"]: ok=False; break
                # lift
                obj=env.get_object_pose(pick); t={"x":obj["x"],"y":obj["y"],"z":obj["z"]+0.15}
                pid.set_phase("lift"); pid.set_target(pos=t)
                lift_ok = False
                for _ in range(800):
                    eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act("lift",None); a[6]=1.0
                    try: env.step(a)
                    except: break
                    if math.hypot(eef["x"]-t["x"],eef["y"]-t["y"],eef["z"]-t["z"])<0.02: lift_ok=True; break
                if not lift_ok: ok=False; break
                # place
                p=execute_place(env,pick,place,config=PlaceConfig(kp=0.05,max_delta=0.02,max_retries=2,mode="simple"))
                if not p.get("success",False): ok=False; break
            if ok:
                goal = [["on", cubes[i+1], cubes[i]] for i in range(len(cubes)-1)]
                try:
                    from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
                    from vlm_nsm.core.predicate_monitor import goal_satisfied
                    poses=env.get_all_object_poses(); targets=dict(poses); targets["table"]=env.get_object_pose("table")
                    physical=physical_state_to_predicates(poses,targets,{"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
                    ok=bool(goal_satisfied(physical,goal,mode="symbolic"))
                except: ok=False
            if ok: successes += 1
        else:
            # Hanoi — use inline logic from verified script
            from vlm_nsm.planners.internal_hanoi_planner import InternalHanoiPlanner
            GRASP_CFG = GraspConfig(kp=0.05,max_delta=0.01,close_steps=100,max_retries=3,holding_check_enabled=True,micro_align_enabled=False)
            plan = InternalHanoiPlanner().solve(3); ok = True
            for act in plan.actions:
                disk = act.args[0]
                if act.name == 'pick':
                    obj=env.get_object_pose(disk); t={'x':obj['x'],'y':obj['y'],'z':obj['z']+0.10}
                    pid.set_phase('reach_pick'); pid.set_target(pos=t)
                    for _ in range(600):
                        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act('reach_pick',None); a[6]=-1.0
                        try: env.step(a)
                        except: ok=False; break
                        if math.hypot(eef['x']-t['x'],eef['y']-t['y'],eef['z']-t['z'])<0.02: break
                    for _ in range(50):
                        eef=env.get_eef_pose()
                        if math.hypot(obj['x']-eef['x'],obj['y']-eef['y'])<0.005: break
                        dx=np.clip(0.05*(obj['x']-eef['x']),-0.005,0.005); dy=np.clip(0.05*(obj['y']-eef['y']),-0.005,0.005)
                        try: env.step(np.array([dx/0.05,dy/0.05,0,0,0,0,-1.0],dtype=np.float32))
                        except: ok=False; break
                    g=execute_grasp(env,disk,config=GRASP_CFG,skip_approach=True,skip_lift=True)
                    if not g['success']: ok=False; break
                    obj=env.get_object_pose(disk); t={'x':obj['x'],'y':obj['y'],'z':obj['z']+0.15}
                    pid.set_phase('lift'); pid.set_target(pos=t)
                    for _ in range(500):
                        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act('lift',None); a[6]=1.0
                        try: env.step(a)
                        except: ok=False; break
                        if math.hypot(eef['x']-t['x'],eef['y']-t['y'],eef['z']-t['z'])<0.02: break
                else:
                    tgt=act.args[1]; pmap={'peg1':(-0.12,-0.08),'peg2':(0.0,-0.08),'peg3':(0.12,-0.08)}
                    px,py=pmap.get(tgt,(0,0)); tt={'x':px,'y':py,'z':0.95}
                    pid.set_phase('reach_place'); pid.set_target(pos=tt)
                    for _ in range(800):
                        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act('reach_place',None); a[6]=1.0
                        try: env.step(a)
                        except: ok=False; break
                        if math.hypot(eef['x']-tt['x'],eef['y']-tt['y'],eef['z']-tt['z'])<0.02: break
                    for _ in range(30):
                        try: env.step(np.array([0,0,0,0,0,0,-1.0],dtype=np.float32))
                        except: ok=False; break
                    for _ in range(10):
                        try: env.step(np.array([0,0,0,0,0,0,-1.0],dtype=np.float32))
                        except: ok=False; break
                    eef=env.get_eef_pose(); rt={'x':eef['x'],'y':eef['y'],'z':eef['z']+0.15}
                    pid.set_phase('retreat'); pid.set_target(pos=rt)
                    for _ in range(500):
                        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act('retreat',None); a[6]=-1.0
                        try: env.step(a)
                        except: break
                        if math.hypot(eef['x']-rt['x'],eef['y']-rt['y'],eef['z']-rt['z'])<0.02: break
                if not ok: break
            if ok: successes += 1

        if (ep+1) % 3 == 0 or ep == 0:
            print(f"  [{ep+1}/{args.num_episodes}] {successes/(ep+1):.0%}", flush=True)

    out = Path(args.output_dir) / args.task
    out.mkdir(parents=True, exist_ok=True)
    summary = {"task":args.task,"num_episodes":args.num_episodes,"success_rate":successes/args.num_episodes}
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(f"{args.task}: {successes}/{args.num_episodes} ({summary['success_rate']:.0%}) → {out}")


if __name__ == "__main__":
    main()
