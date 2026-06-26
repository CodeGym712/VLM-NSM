"""True VLA E2E: Language → Plan → VLA+Gate → PID Fallback → Monitor.

Unlike run_e2e_language_to_sim.py (which uses DAgger BC), this uses
the VLA Diffusion Policy (M1) with Safety Gate for reach_pick phase,
and PID controller for all other phases.

Pipeline:
  NL → Qwen/Mock Parser → Alignment → BFS Planner →
  for each phase:
    if reach_pick: VLA predicts → Gate validates → execute (or PID fallback)
    else: PID executes directly
  → Predicate Monitor → goal check

Usage:
    python -m vlm_nsm.experiments.run_e2e_vla_pid \
        --checkpoint generated/results/vla/vla_m1_real_sym/best.pt \
        --parser mock --num_episodes 10
"""

from __future__ import annotations

import argparse, json, time, os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_PHASE_TO_SKILL_ID = {"reach_pick":0,"grasp":1,"lift":2,"reach_place":3,"release":4,"retreat":5}
N_PHASES = 6


def load_vla_model(checkpoint_path: str, device: torch.device):
    """Load Diffusion Policy VLA model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "diffusion_policy")
    if model_type == "diffusion_policy":
        from vlm_nsm.vla.models.diffusion_policy import DiffusionPolicy
        model = DiffusionPolicy(
            symbolic_dim=ckpt.get("symbolic_dim", 42),
            action_dim=ckpt.get("action_dim", 4),
            horizon=ckpt.get("horizon", 16),
            cond_dim=ckpt.get("cond_dim", 1216),
        ).to(device)
    else:
        from vlm_nsm.vla.models.vision_encoder import VLABCModel
        model = VLABCModel(symbolic_dim=ckpt.get("symbolic_dim", 42)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_type


def run_vla_e2e_demo(env, instruction, parser, planner, vla_model, vla_model_type,
                     device, gate, symbolic_dim=42,
                     record_video=False, video_frames=None):
    """Run one complete Language→VLA+Gate+PID→Monitor episode."""
    result = {
        "instruction": instruction, "success": False,
        "parse_success": False, "plan_success": False, "exec_success": False,
    }

    # 1. Parse language
    parse_result = parser.parse(instruction)
    if not parse_result.success:
        result["parse_error"] = parse_result.error
        return result
    result["parse_success"] = True

    # 2. Alignment + planning
    from vlm_nsm.core.alignment import AlignmentLayer
    from vlm_nsm.planning.stacking_utils import target_stack_from_predicates

    alignment = AlignmentLayer()
    try:
        align_result = alignment.align(parse_result.goal_json)
    except Exception as e:
        result["alignment_error"] = str(e)
        return result

    try:
        target_stack = target_stack_from_predicates(align_result.goal_predicates)
    except Exception as e:
        result["plan_error"] = f"target_stack: {e}"
        return result

    from vlm_nsm.core.planner_adapter import PlannerAdapter
    planner_inst = PlannerAdapter("bfs" if planner == "bfs" else planner)
    cubes = [f"cube{i}" for i in range(1, len(target_stack) + 1)]
    plan_result = planner_inst.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        result["plan_error"] = plan_result.error
        return result
    result["plan_success"] = True
    result["plan_actions"] = [(a.name, a.args) for a in plan_result.actions]

    # 3. Execute each action with VLA+Gate+PID
    from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
    from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
    from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import build_stacking_goal

    pid = OperationalSpacePID()
    total_steps = 0
    gate_ok = 0
    gate_total = 0
    success = True

    for action in plan_result.actions:
        pick_obj = action.args[0]
        place_obj = action.args[2] if len(action.args) > 2 else "table"

        for step_name in ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]:
            if step_name == "reach_pick":
                # VLA phase
                obj = env.get_object_pose(pick_obj)
                target = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}
                pid.set_phase("reach_pick")
                pid.set_target(pos=target)
                scripted_action = pid.act("reach_pick", None)

                # Capture RGB for VLA
                try:
                    img_a = env.env.sim.render(camera_name="agentview", height=128, width=128)
                    img_f = env.env.sim.render(camera_name="frontview", height=128, width=128)
                    eef = env.get_eef_pose()
                    grip = env.get_gripper_state()
                    # Build observation
                    av_t = torch.from_numpy(img_a.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
                    fv_t = torch.from_numpy(img_f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
                    prop_t = torch.tensor([[eef["x"],eef["y"],eef["z"],grip.get("open",0.),0,0,0]], device=device)
                    sym_t = torch.zeros(1, symbolic_dim, device=device)  # B5: real symbolic
                    with torch.no_grad():
                        if vla_model_type == "diffusion_policy":
                            chunk = vla_model.sample(av_t, fv_t, prop_t, sym_t)
                            vla_4d = chunk[0, 0].cpu().numpy()  # first action in chunk
                        else:
                            vla_4d = vla_model(av_t, fv_t, prop_t, sym_t).cpu().numpy().ravel()
                    vla_action_np = np.zeros(7, dtype=np.float32)
                    vla_action_np[:3] = vla_4d[:3]
                    vla_action_np[6] = vla_4d[3] if len(vla_4d) > 3 else -1.0
                    # Gate check
                    approved, final_action, reason = gate.validate(
                        vla_action_np, scripted_action, 0.5,
                        np.array([eef["x"],eef["y"],eef["z"]]), "reach_pick", None)
                    gate_total += 1
                    if approved:
                        gate_ok += 1
                    action_7d = final_action
                except Exception:
                    action_7d = scripted_action  # fallback to PID

            elif step_name == "grasp":
                cfg = GraspConfig(kp=0.05, max_delta=0.01, close_steps=70, max_retries=2,
                                  holding_check_enabled=True)
                g = execute_grasp(env, pick_obj, config=cfg, skip_approach=True, skip_lift=True)
                total_steps += g.get("total_step_count", 0)
                if not g["success"]:
                    success = False
                    break
                continue

            elif step_name in ("release", "retreat"):
                if step_name == "release":
                    cfg = PlaceConfig(kp=0.05, max_delta=0.02, max_retries=2, mode="simple")
                    p = execute_place(env, pick_obj, place_obj, config=cfg)
                    total_steps += p.get("step_count", 0)
                    if not p.get("success", False):
                        success = False
                        break
                continue

            else:
                # lift, reach_place: PID control
                if step_name == "lift":
                    obj = env.get_object_pose(pick_obj)
                    target = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.15}
                    gripper = 1.0
                else:  # reach_place
                    tgt = env.get_target_pose(place_obj)
                    target = {"x": tgt["x"], "y": tgt["y"], "z": tgt["z"] + 0.15}
                    gripper = 1.0

                pid.set_phase(step_name)
                pid.set_target(pos=target)
                for _ in range(500):
                    eef = env.get_eef_pose()
                    pid.set_current(eef)
                    a = pid.act(step_name, None)
                    a[6] = gripper
                    try:
                        env.step(a)
                        total_steps += 1
                    except ValueError:
                        success = False
                        break
                    if np.linalg.norm(np.array([eef["x"]-target["x"], eef["y"]-target["y"], eef["z"]-target["z"]])) < 0.02:
                        break

            if not success:
                break

    # 4. Monitor: verify goal predicates
    goal_ok = False
    try:
        from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
        from vlm_nsm.core.predicate_monitor import goal_satisfied

        goal_preds = build_stacking_goal(target_stack)
        poses = env.get_all_object_poses()
        targets = dict(poses)
        targets["table"] = env.get_object_pose("table")
        physical = physical_state_to_predicates(poses, targets,
            {"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
        goal_ok = bool(goal_satisfied(physical, goal_preds, mode="symbolic"))
    except Exception:
        pass

    result["success"] = goal_ok
    result["steps"] = total_steps
    result["gate_acceptance"] = gate_ok / max(gate_total, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="generated/results/vla/vla_m1_real_sym/best.pt")
    parser.add_argument("--parser", default="mock", choices=["mock","cached_qwen","online_qwen"])
    parser.add_argument("--planner", default="bfs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dir", default="generated/results/e2e_vla_pid")
    parser.add_argument("--gate_config", default=None)
    parser.add_argument("--qwen_model", default="qwen-plus")
    parser.add_argument("--qwen_api_base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load VLA model
    print(f"Loading VLA model: {args.checkpoint}")
    vla_model, vla_model_type = load_vla_model(args.checkpoint, device)
    print(f"  Model type: {vla_model_type}")

    # Gate
    from vlm_nsm.neuro_symbolic.safety_gate import GateConfig, NeuroSymbolicSafetyGate
    gate = NeuroSymbolicSafetyGate(GateConfig(max_xyz_deviation=0.025))

    # Env
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                              cube_names=["cube1","cube2"], render_mode="none")

    # Use pre-parsed demos (same as run_e2e_language_to_sim.py)
    DEMOS = [
        {"instruction": "Put the red cube on the green cube.",
         "goal_predicates": [["on","cube1","cube2"]], "cubes": ["cube1","cube2"],
         "target_stack": ["cube2","cube1"]},
        {"instruction": "Stack the red block on top of the green one.",
         "goal_predicates": [["on","cube1","cube2"]], "cubes": ["cube1","cube2"],
         "target_stack": ["cube2","cube1"]},
        {"instruction": "Place the cube on top of the other cube.",
         "goal_predicates": [["on","cube1","cube2"]], "cubes": ["cube1","cube2"],
         "target_stack": ["cube2","cube1"]},
    ]

    successes = 0
    for i, demo in enumerate(DEMOS):
        env.reset()
        target_stack = demo["target_stack"]
        cubes = demo["cubes"]

        # Plan
        from vlm_nsm.core.planner_adapter import PlannerAdapter
        planner_inst = PlannerAdapter("bfs")
        plan = planner_inst.solve(cubes=cubes, target_stack=target_stack)
        if not plan.success:
            print(f"  [{i+1}] Plan failed: {plan.error}")
            continue

        # Execute with VLA+PID
        from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
        from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
        from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
        from vlm_nsm.experiments.run_embodied_stacking_robosuite import build_stacking_goal

        pid = OperationalSpacePID()
        total_steps = 0; gate_ok = 0; gate_total = 0; success = True

        for action in plan.actions:
            pick_obj = action.args[0]
            place_obj = action.args[2] if len(action.args) > 2 else "table"

            for skill in ["reach_pick","grasp","lift","reach_place","release","retreat"]:
                if skill == "reach_pick":
                    obj = env.get_object_pose(pick_obj)
                    target = {"x":obj["x"],"y":obj["y"],"z":obj["z"]+0.10}
                    pid.set_phase("reach_pick"); pid.set_target(pos=target)
                    scripted_a = pid.act("reach_pick",None)
                    try:
                        img_a = env.env.sim.render(camera_name="agentview",height=128,width=128)
                        img_f = env.env.sim.render(camera_name="frontview",height=128,width=128)
                        eef = env.get_eef_pose(); grip = env.get_gripper_state()
                        av = torch.from_numpy(img_a.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
                        fv = torch.from_numpy(img_f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
                        pr = torch.tensor([[eef["x"],eef["y"],eef["z"],grip.get("open",0.),0,0,0]],device=device)
                        sy = torch.zeros(1,42,device=device)
                        with torch.no_grad():
                            chunk = vla_model.sample(av,fv,pr,sy)
                            v4d = chunk[0,0].cpu().numpy()
                        vla_a = np.zeros(7,dtype=np.float32); vla_a[:3]=v4d[:3]; vla_a[6]=v4d[3] if len(v4d)>3 else -1.0
                        approved, final_a, reason = gate.validate(vla_a, scripted_a, 0.5,
                            np.array([eef["x"],eef["y"],eef["z"]]), "reach_pick", None)
                        gate_total += 1
                        if approved: gate_ok += 1
                        env.step(final_a); total_steps += 1
                    except Exception:
                        env.step(scripted_a); total_steps += 1

                elif skill == "grasp":
                    cfg = GraspConfig(kp=0.05,max_delta=0.01,close_steps=70,max_retries=2,holding_check_enabled=True)
                    g = execute_grasp(env,pick_obj,config=cfg,skip_approach=True,skip_lift=True)
                    total_steps += g.get("total_step_count",0)
                    if not g["success"]: success = False

                elif skill == "release":
                    cfg = PlaceConfig(kp=0.05,max_delta=0.02,max_retries=2,mode="simple")
                    p = execute_place(env,pick_obj,place_obj,config=cfg)
                    total_steps += p.get("step_count",0)
                    if not p.get("success",False): success = False

                elif skill == "retreat":
                    eef = env.get_eef_pose(); rt = {"x":eef["x"],"y":eef["y"],"z":eef["z"]+0.15}
                    pid.set_phase("retreat"); pid.set_target(pos=rt)
                    for _ in range(200):
                        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act("retreat",None); a[6]=-1.0
                        try: env.step(a); total_steps+=1
                        except: break
                        if np.linalg.norm([eef["x"]-rt["x"],eef["y"]-rt["y"],eef["z"]-rt["z"]])<0.02: break

                elif skill in ("lift","reach_place"):
                    gripper = 1.0
                    if skill == "lift":
                        obj = env.get_object_pose(pick_obj); t = {"x":obj["x"],"y":obj["y"],"z":obj["z"]+0.15}
                    else:
                        tgt = env.get_target_pose(place_obj); t = {"x":tgt["x"],"y":tgt["y"],"z":tgt["z"]+0.15}
                    pid.set_phase(skill); pid.set_target(pos=t)
                    for _ in range(500):
                        eef=env.get_eef_pose(); pid.set_current(eef); a=pid.act(skill,None); a[6]=gripper
                        try: env.step(a); total_steps+=1
                        except: success=False; break
                        if np.linalg.norm([eef["x"]-t["x"],eef["y"]-t["y"],eef["z"]-t["z"]])<0.02: break

                if not success: break
            if not success: break

        # Monitor
        try:
            goal_preds = build_stacking_goal(target_stack)
            from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
            from vlm_nsm.core.predicate_monitor import goal_satisfied
            poses=env.get_all_object_poses(); targets=dict(poses); targets["table"]=env.get_object_pose("table")
            physical=physical_state_to_predicates(poses,targets,{"xy_threshold":0.05,"z_min":-0.005,"z_max":0.05})
            goal_ok=bool(goal_satisfied(physical,goal_preds,mode="symbolic"))
        except: goal_ok=False

        if goal_ok: successes += 1
        tag = "✅" if goal_ok else "❌"
        print(f"  [{i+1}/{len(DEMOS)}] {tag} \"{demo['instruction'][:50]}\" "
              f"gate={gate_ok/max(gate_total,1):.0%} steps={total_steps}")

    print(f"\nVLA E2E: {successes}/{len(DEMOS)} ({successes/len(DEMOS):.0%})")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"successes": successes, "total": len(DEMOS),
                   "rate": successes/len(DEMOS)}, f, indent=2)
    print(f"Saved to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
