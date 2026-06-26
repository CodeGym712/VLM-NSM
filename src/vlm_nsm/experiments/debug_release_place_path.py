"""Debug: SkillLibrary ReleasePlaceSkill vs direct execute_place.

Compares three paths on the SAME placement task:
  A. Direct execute_place(max_retries=2)
  B. SkillLibrary ReleasePlaceSkill
  C. BT Action node wrapping ReleasePlaceSkill
"""

import os, json, math, numpy as np
os.environ['MUJOCO_GL']='egl'

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
from vlm_nsm.embodied.skills.base_skill import SkillLibrary, SkillParams
from vlm_nsm.embodied.skills.grasp_skill import ReleasePlaceSkill
from vlm_nsm.embodied.behavior_tree import Action, BTStatus, BTTrace
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied

env = RobosuiteEnvWrapper(env_name='Stack', robots='Panda', cube_names=['cube1','cube2'], render_mode='none')
env.env.horizon = 8000
pid = OperationalSpacePID()

def _dist(a,b): return math.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2+(a['z']-b['z'])**2)

def check_on(env):
    try:
        poses=env.get_all_object_poses();targets=dict(poses);targets["table"]=env.get_object_pose("table")
        physical=physical_state_to_predicates(poses,targets,{"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
        return goal_satisfied(physical,[["on","cube1","cube2"]],mode="symbolic")
    except: return False

def setup_scene():
    """Pick cube1 and move above cube2. Return True if grasp+lift succeeded."""
    env.reset(); pid.reset(); np.random.seed(42)
    pick, place = "cube1", "cube2"
    obj=env.get_object_pose(pick);t={'x':obj['x'],'y':obj['y'],'z':obj['z']+0.10}
    pid.set_phase('reach_pick');pid.set_target(pos=t)
    for _ in range(800):
        e=env.get_eef_pose();pid.set_current(e);a=pid.act('reach_pick',None);a[6]=-1.0
        try:env.step(a)
        except:return False,None,None
        if _dist(e,t)<0.02:break
    g=execute_grasp(env,pick,config=GraspConfig(kp=0.05,max_delta=0.01,close_steps=70,max_retries=2,holding_check_enabled=True),skip_approach=True,skip_lift=True)
    if not g['success']: return False,None,None
    obj=env.get_object_pose(pick);t={'x':obj['x'],'y':obj['y'],'z':obj['z']+0.15}
    pid.set_phase('lift');pid.set_target(pos=t)
    for _ in range(800):
        e=env.get_eef_pose();pid.set_current(e);a=pid.act('lift',None);a[6]=1.0
        try:env.step(a)
        except:return False,None,None
        if _dist(e,t)<0.02:break
    tgt=env.get_target_pose(place);tt={'x':tgt['x'],'y':tgt['y'],'z':tgt['z']+0.15}
    pid.set_phase('reach_place');pid.set_target(pos=tt)
    for _ in range(800):
        e=env.get_eef_pose();pid.set_current(e);a=pid.act('reach_place',None);a[6]=1.0
        try:env.step(a)
        except:return False,None,None
        if _dist(e,tt)<0.02:break
    return True, pick, place

print("=" * 60)
print("Phase 3.1: SkillLibrary vs Direct execute_place Debug")
print("=" * 60)

results = {"A_direct": [], "B_skilllib": [], "C_bt_action": []}

for ep in range(5):
    # --- Path A: Direct execute_place ---
    ok, pick, place = setup_scene()
    if not ok: results["A_direct"].append(False); continue
    obj_z_before = env.get_object_pose(pick)["z"]
    p_result = execute_place(env, pick, place, config=PlaceConfig(kp=0.05,max_delta=0.02,max_retries=2,mode='simple'))
    a_ok = check_on(env)
    results["A_direct"].append({
        "ep":ep, "success":a_ok, "place_ok":p_result.get("success"),
        "failure_type":p_result.get("failure_type"),
        "step_count":p_result.get("step_count"),
        "retries":p_result.get("total_attempts"),
        "obj_z_before":obj_z_before,
    })

    # --- Path B: SkillLibrary ReleasePlaceSkill ---
    ok, pick, place = setup_scene()
    if not ok: results["B_skilllib"].append(False); continue
    lib = SkillLibrary(); lib.register(ReleasePlaceSkill())
    skill = lib.get("release_place")
    obj_z_before = env.get_object_pose(pick)["z"]
    params = SkillParams(target_object=pick, target_location=place)
    sk_result = skill.execute_with_pid(env, pid, params)
    b_ok = check_on(env)
    results["B_skilllib"].append({
        "ep":ep, "success":b_ok, "skill_ok":sk_result.success,
        "failure_type":sk_result.failure_type,
        "steps":sk_result.steps,
        "obj_z_before":obj_z_before,
    })

    # --- Path C: BT Action node ---
    ok, pick, place = setup_scene()
    if not ok: results["C_bt_action"].append(False); continue
    lib2 = SkillLibrary(); lib2.register(ReleasePlaceSkill())
    action_node = Action("release_place", target_object=pick, target_location=place)
    trace = BTTrace()
    obj_z_before = env.get_object_pose(pick)["z"]
    st = action_node.tick(env, pid, lib2, trace)
    c_ok = check_on(env)
    last_trace = trace.nodes[-1] if trace.nodes else None
    results["C_bt_action"].append({
        "ep":ep, "success":c_ok, "bt_status":st.value,
        "failure_type":last_trace.failure_type if last_trace else None,
        "steps":last_trace.steps if last_trace else 0,
        "obj_z_before":obj_z_before,
    })

# --- Report ---
print(f"\nResults (5 episodes, same seed each condition):")
for path, name in [("A_direct","Direct execute_place"),("B_skilllib","SkillLibrary"),("C_bt_action","BT Action")]:
    data = [r for r in results[path] if isinstance(r, dict)]
    ok = sum(1 for r in data if r["success"])
    print(f"\n{name}:")
    print(f"  Success: {ok}/{len(data)}")
    if data:
        failures = [r.get("failure_type","?") for r in data if not r["success"]]
        print(f"  Failure types: {failures}")
        # Compare parameters
        if path == "A_direct":
            print(f"  Calls: execute_place(max_retries=2, mode='simple')")
        elif path == "B_skilllib":
            print(f"  Calls: ReleasePlaceSkill.execute_with_pid → execute_place(max_retries=2, mode='simple')")
        print(f"  Place config: kp=0.05, max_delta=0.02, max_retries=2, mode='simple'")

# Save
out = "generated/results/upper_ns/release_place_path_debug.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"results": {k: [r for r in v if isinstance(r,dict)] for k,v in results.items()}}, f, indent=2)
print(f"\nSaved to {out}")
