"""End-to-end language → plan → Robosuite simulation demo.

Runs natural language instructions through the full pipeline:
  NL instruction → GoalParser → Alignment → Planner → Robosuite execution
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vlm_nsm.core.goal_parser import MockGoalParser, GoalParseResult
from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.experiments.robosuite_episode_runner import run_episode
from vlm_nsm.learning.eval_bc import load_checkpoint
from vlm_nsm.learning.eval_bc_hybrid import _get_bc_action
from vlm_nsm.perception.oracle_perception import perceive_scene
from vlm_nsm.perception.scene_prompt import build_scene_description
from vlm_nsm.skills.skill_plan_from_symbolic import skill_plan_from_symbolic
from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor


class CachedQwenGoalParser:
    """Goal parser that loads pre-computed Qwen2.5 parse results.

    Cached format: JSON dict mapping instruction string → goal_json dict.
    Example:
        {
          "Put the red cube on the green cube.": {
            "task_type": "stacking",
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
            "goal_predicates": [["on", "cube1", "cube2"]]
          }
        }
    """

    def __init__(self, cache_path: str):
        import json
        self._cache = json.loads(Path(cache_path).read_text())

    def parse(self, instruction: str, **kwargs) -> GoalParseResult:
        goal_json = self._cache.get(instruction)
        if goal_json is None:
            return GoalParseResult(
                success=False,
                goal_json=None,
                error=f"Instruction not found in cache: {instruction[:60]}",
            )
        return GoalParseResult(success=True, goal_json=goal_json)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

N_PHASES = 6
_PHASE_TO_SKILL_ID = {
    "reach_pick": 0, "grasp": 1, "lift": 2,
    "reach_place": 3, "release": 4, "retreat": 5,
}
VALID_PREDICATES = ["on", "clear", "free-gripper", "grasped", "smaller"]
TYPE_TABLE = {
    "cube1": "block", "cube2": "block", "table": "table",
    "red_cube": "block", "green_cube": "block", "blue_cube": "block",
}
PREDICATE_SIGNATURE = {
    "on": [["block", "block"], ["block", "table"]],
    "clear": [["block"]],
    "free-gripper": [[]],
    "grasped": [["block"]],
    "smaller": [["block", "block"], ["block", "table"]],
}
OBJECTS = ["cube1", "cube2", "table", "red_cube", "green_cube", "blue_cube"]

# Pre-defined demos: (instruction, task_type, expected_goal, cubes, target_stack)
DEMOS = [
    {
        "instruction": "Put the red cube on the green cube.",
        "task_type": "stacking",
        "goal_predicates": [["on", "cube1", "cube2"]],
        "cubes": ["cube1", "cube2"],
        "target_stack": ["cube2", "cube1"],
    },
    {
        "instruction": "Stack cube A on cube B.",
        "task_type": "stacking",
        "goal_predicates": [["on", "cube1", "cube2"]],
        "cubes": ["cube1", "cube2"],
        "target_stack": ["cube2", "cube1"],
    },
    {
        "instruction": "Place the cube on top of the other cube.",
        "task_type": "stacking",
        "goal_predicates": [["on", "cube1", "cube2"]],
        "cubes": ["cube1", "cube2"],
        "target_stack": ["cube2", "cube1"],
    },
    {
        "instruction": "Move the block onto the target block.",
        "task_type": "stacking",
        "goal_predicates": [["on", "cube1", "cube2"]],
        "cubes": ["cube1", "cube2"],
        "target_stack": ["cube2", "cube1"],
    },
    {
        "instruction": "Stack the red block on top of the green one.",
        "task_type": "stacking",
        "goal_predicates": [["on", "cube1", "cube2"]],
        "cubes": ["cube1", "cube2"],
        "target_stack": ["cube2", "cube1"],
    },
]


def build_stacking_goal(target_stack):
    """Build goal predicates from target stack (bottom-to-top convention).

    Does NOT add on(bottom, table) — physical predicate extraction checks
    table-center xy, and the bottom block drifts during stacking.
    """
    predicates = []
    for i in range(len(target_stack) - 1, 0, -1):
        predicates.append(["on", target_stack[i], target_stack[i - 1]])
    return predicates


def _render_frame_safe(env, camera="agentview", height=512, width=512):
    """Capture a rendered frame via offscreen sim render; returns None on failure.

    Uses sim.render() for offscreen mode (env.render() returns None when
    has_renderer=False). Falls back silently on any error.
    """
    try:
        sim = env.env.sim
        img = sim.render(camera_name=camera, height=height, width=width)
        return img
    except Exception:
        return None


def _write_video_safe(frames, path, fps=20):
    """Write frames to MP4 via OpenCV; returns error string or None."""
    if not frames:
        return "no frames captured"
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return None
    except ImportError:
        return "opencv not available"
    except Exception as e:
        return f"video write error: {e}"


def run_demo(env, demo, planner, bc_model, device, action_mean, action_std,
             exec_config, output_dir, use_bc=True, bc_phases=None,
             parser=None, record_video=False, video_frames=None,
             fallback_parser=None, scene_description=None,
             scene_info=None, control_interface="legacy"):
    """Run one end-to-end demo.

    If the primary parser fails and fallback_parser is provided, retries
    parsing+alignment+planning with the fallback. Does not re-execute the
    episode (env state would be lost).
    """
    instruction = demo["instruction"]
    cubes = demo["cubes"]

    # Determine parser tag
    _parser_tag = type(parser).__name__ if parser else "MockGoalParser"

    # 1. Parse goal
    _parser = parser if parser is not None else MockGoalParser()
    parse_kwargs = {
        "instruction": instruction,
        "task_type": demo["task_type"],
        "objects": cubes,
        "n_blocks": len(cubes),
    }
    if scene_description is not None:
        parse_kwargs["scene_description"] = scene_description
    parse_result = _parser.parse(**parse_kwargs)

    # Fallback logic: if parse failed and we have a fallback, retry
    fallback_used = False
    if not parse_result.success and fallback_parser is not None:
        print(f"  Primary parser failed ({parse_result.error}), trying fallback...")
        parse_result = fallback_parser.parse(**parse_kwargs)
        fallback_used = True
        _parser_tag += f"+{type(fallback_parser).__name__}"

    goal_json = parse_result.goal_json

    # Scene inventory fields for summary
    _scene_obj_list = (
        [{"object_id": o.label, "color": o.color} for o in scene_info.objects]
        if scene_info else None
    )
    _scene_fields = {
        "scene_inventory_source": "oracle" if scene_description else "none",
        "scene_description": scene_description,
        "scene_objects": _scene_obj_list,
        "object_grounding_valid": None,
        "grounding_errors": None,
    }

    # 2. Alignment
    aligner = AlignmentLayer()
    align_result = aligner.align_and_validate(
        goal_json=goal_json if goal_json else demo,
        valid_objects=cubes + ["table"],
        valid_predicates=VALID_PREDICATES,
        type_table={k: v for k, v in TYPE_TABLE.items() if k in cubes or k == "table"},
        predicate_signature=PREDICATE_SIGNATURE,
        task_type=demo["task_type"],
        n_blocks=len(cubes),
    )

    if not align_result.success:
        return {
            "instruction": instruction,
            "success": False,
            "parser": _parser_tag,
            "parse_success": parse_result.success,
            "parse_error": None if parse_result.success else parse_result.error,
            "fallback_used": fallback_used,
            "fallback_parser": type(fallback_parser).__name__ if fallback_used else None,
            "parsed_goal": goal_json,
            "goal_json": goal_json,
            "aligned_goal_predicates": None,
            "parser_target_stack": None,
            "demo_target_stack": demo.get("target_stack", []),
            "target_stack_source": None,
            "target_stack_mismatch": None,
            "raw_qwen_output_path": parse_result.raw_output_path if hasattr(parse_result, "raw_output_path") else None,
            "alignment_status": "failed",
            "alignment_error": align_result.error,
            "symbolic_plan": None,
            "physical_goal_predicates": None,
            "plan_status": None,
            "plan_actions": None,
            "planner_error": None,
            "policy_used": None,
            "execution_success": False,
            "failure_type": None,
            "episode_result": None,
            **_scene_fields,
        }

    # 3. Derive target_stack from parser/alignment output (parser-driven)
    on_predicates = [
        (p.name, p.args[0], p.args[1])
        for p in align_result.goal_predicates
        if p.name == "on"
    ]
    try:
        target_stack = target_stack_from_predicates(on_predicates, cubes)
    except ValueError as e:
        return {
            "instruction": instruction,
            "success": False,
            "parser": _parser_tag,
            "parse_success": True,
            "parse_error": None,
            "fallback_used": fallback_used,
            "fallback_parser": type(fallback_parser).__name__ if fallback_used else None,
            "parsed_goal": goal_json,
            "goal_json": goal_json,
            "aligned_goal_predicates": [[pred.name, *pred.args] for pred in align_result.goal_predicates],
            "parser_target_stack": None,
            "demo_target_stack": demo.get("target_stack", []),
            "target_stack_source": "parser_aligned_goal",
            "target_stack_mismatch": None,
            "raw_qwen_output_path": parse_result.raw_output_path if hasattr(parse_result, "raw_output_path") else None,
            "alignment_status": "passed",
            "symbolic_plan": None,
            "physical_goal_predicates": None,
            "plan_status": "failed",
            "plan_actions": None,
            "planner_error": f"target_stack derivation: {e}",
            "policy_used": None,
            "execution_success": False,
            "failure_type": None,
            "episode_result": None,
            **_scene_fields,
        }
    goal_predicates = build_stacking_goal(target_stack)

    # 4. Plan
    plan_result = planner.solve(
        cubes=cubes,
        target_stack=target_stack,
    )

    if not plan_result.success:
        return {
            "instruction": instruction,
            "success": False,
            "parser": _parser_tag,
            "parse_success": True,
            "parse_error": None,
            "fallback_used": fallback_used,
            "fallback_parser": type(fallback_parser).__name__ if fallback_used else None,
            "parsed_goal": goal_json,
            "goal_json": goal_json,
            "aligned_goal_predicates": [[pred.name, *pred.args] for pred in align_result.goal_predicates],
            "parser_target_stack": target_stack,
            "demo_target_stack": demo.get("target_stack", []),
            "target_stack_source": "parser_aligned_goal",
            "target_stack_mismatch": target_stack != demo.get("target_stack", []),
            "raw_qwen_output_path": parse_result.raw_output_path if hasattr(parse_result, "raw_output_path") else None,
            "alignment_status": "passed",
            "symbolic_plan": None,
            "physical_goal_predicates": None,
            "plan_status": "failed",
            "plan_actions": None,
            "planner_error": plan_result.error,
            "policy_used": None,
            "execution_success": False,
            "failure_type": None,
            "episode_result": None,
            **_scene_fields,
        }

    # 5. Execute in Robosuite
    if use_bc:
        _bc_phases = bc_phases if bc_phases is not None else ["reach_pick"]
        bc_phase_ids = {_PHASE_TO_SKILL_ID[p] for p in _bc_phases}
        condition_on_phase = True

        def policy_fn(obs, phase_id):
            obs_in = obs
            if condition_on_phase:
                one_hot = np.zeros(N_PHASES, dtype=np.float32)
                one_hot[phase_id] = 1.0
                obs_in = np.concatenate([obs, one_hot])
            return _get_bc_action(bc_model, obs_in, device, action_mean, action_std, [3, 4, 5])
    else:
        bc_phase_ids = None
        policy_fn = None

    ep_kwargs = dict(horizon=1000, early_stop_on_goal=True)
    if exec_config:
        for k in ["kp", "max_delta", "use_place_state_machine", "horizon_aware",
                   "place_retry", "grasp_close_steps", "grasp_max_retries",
                   "place_max_retries", "place_mode", "skip_retreat_if_goal_satisfied",
                   "holding_check_enabled", "lift_early_exit_enabled",
                   "close_steps_min", "close_steps_max", "close_stabilize_steps",
                   "release_steps_min", "release_steps_max", "post_release_stable_steps",
                   "close_mode", "close_steps_fast", "fallback_extend_close_if_not_holding"]:
            if k in exec_config:
                ep_kwargs[k] = exec_config[k]

    # Optional frame capture
    _restore_step = None
    if record_video and video_frames is not None:
        frame = _render_frame_safe(env)
        if frame is not None:
            video_frames.append(frame)
        original_step = env.step
        def _step_and_capture(action):
            obs, rew, done, info = original_step(action)
            f = _render_frame_safe(env)
            if f is not None:
                video_frames.append(f)
            return obs, rew, done, info
        env.step = _step_and_capture
        _restore_step = original_step

    try:
        if control_interface == "skill_executor":
            skill_plan = skill_plan_from_symbolic(plan_result.actions)
            skill_executor = ScriptedSkillExecutor(ep_kwargs)
            skill_results = []
            total_steps = 0
            final_failure_type = None
            goal_success = True
            for command in skill_plan.commands:
                skill_result = skill_executor.execute(
                    command,
                    env,
                    context={
                        "cubes": cubes,
                        "policy_override": policy_fn,
                        "override_phases": bc_phase_ids,
                    },
                )
                skill_results.append(skill_result.to_dict())
                total_steps += skill_result.steps
                if not skill_result.success:
                    goal_success = False
                    final_failure_type = skill_result.failure_type
                    break
            result = {
                "goal_success": goal_success,
                "num_steps": total_steps,
                "failure_type": final_failure_type,
                "bc_steps": sum(
                    int((r.get("metrics", {}).get("raw_result", {}) or {}).get("bc_steps", 0))
                    for r in skill_results
                ),
                "scripted_steps": sum(
                    int((r.get("metrics", {}).get("raw_result", {}) or {}).get("scripted_steps", 0))
                    for r in skill_results
                ),
                "skill_results": skill_results,
                "skill_plan": skill_plan.to_dict(),
            }
        else:
            result = run_episode(
                env=env,
                actions=plan_result.actions,
                cubes=cubes,
                goal_predicates=goal_predicates,
                policy_override=policy_fn,
                override_phases=bc_phase_ids,
                **ep_kwargs,
            )
    except Exception as e:
        result = {"goal_success": False, "num_steps": 0, "failure_type": f"exception: {e}",
                   "bc_steps": 0, "scripted_steps": 0}
    finally:
        if _restore_step is not None:
            env.step = _restore_step

    ep_out = {
        "goal_success": result.get("goal_success", False),
        "steps": result.get("num_steps", 0),
        "failure_type": result.get("failure_type"),
        "bc_steps": result.get("bc_steps", 0),
        "scripted_steps": result.get("scripted_steps", 0),
    }

    # Build standardized summary fields
    _aligned_predicates = [
        [pred.name, *pred.args]
        for pred in align_result.goal_predicates
    ]
    _parser_target_stack = target_stack  # derived in step 3
    _demo_target_stack = demo.get("target_stack", [])
    _target_stack_source = "parser_aligned_goal"
    _target_stack_mismatch = _parser_target_stack != _demo_target_stack
    _symbolic_plan = [
        {"name": a.name, "args": a.args}
        for a in plan_result.actions
    ]
    _physical_goal_preds = goal_predicates  # build_stacking_goal already excludes table
    _raw_qwen_path = (
        parse_result.raw_output_path
        if hasattr(parse_result, "raw_output_path")
        else None
    )
    _failure_type = result.get("failure_type") if isinstance(result, dict) else None

    return {
        "instruction": instruction,
        "success": result.get("goal_success", False),
        "parser": _parser_tag,
        "parse_success": True,
        "parse_error": parse_result.error if not parse_result.success else None,
        "fallback_used": fallback_used,
        "fallback_parser": type(fallback_parser).__name__ if fallback_used else None,
        "parsed_goal": goal_json,
        "goal_json": goal_json,
        "aligned_goal_predicates": _aligned_predicates,
        "parser_target_stack": _parser_target_stack,
        "demo_target_stack": _demo_target_stack,
        "target_stack_source": _target_stack_source,
        "target_stack_mismatch": _target_stack_mismatch,
        "raw_qwen_output_path": _raw_qwen_path,
        "alignment_status": "passed",
        "alignment_output": str(align_result.goal_predicates),
        "symbolic_plan": _symbolic_plan,
        "physical_goal_predicates": _physical_goal_preds,
        "plan_status": "success",
        "plan_actions": [(a.name, a.args) for a in plan_result.actions],
        "planner_error": plan_result.error if plan_result.error else None,
        "policy_used": "+".join(_bc_phases) if use_bc else "scripted-only",
        "control_interface": control_interface,
        "skill_plan": result.get("skill_plan") if isinstance(result, dict) else None,
        "skill_results": result.get("skill_results") if isinstance(result, dict) else None,
        "execution_success": result.get("goal_success", False),
        "failure_type": _failure_type,
        "episode_result": ep_out,
        "video_recorded": False,
        "video_path": None,
        "video_error": None,
        **_scene_fields,
    }


def main():
    parser = argparse.ArgumentParser(description="E2E language-to-sim demo")
    parser.add_argument("--checkpoint", default="generated/results/stage_7B-4/dagger_reach_bc/best.pt")
    parser.add_argument("--config", default="generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bc", action="store_true", default=True)
    parser.add_argument("--no_bc", action="store_false", dest="use_bc")
    parser.add_argument("--output_dir", default="generated/results/stage_9/e2e_sim_demo")
    parser.add_argument("--bc_phases", nargs="+", default=["reach_pick"],
                        choices=["reach_pick", "reach_place", "none"],
                        help="BC-controlled phases (default: reach_pick)")
    parser.add_argument("--parser", default="mock", choices=["mock", "cached_qwen", "online_qwen"],
                        help="Goal parser: mock, cached_qwen, or online_qwen (real Qwen API)")
    parser.add_argument("--cached_outputs", default=None,
                        help="Path to cached Qwen outputs JSON (required when --parser cached_qwen)")
    parser.add_argument("--qwen_model", default="qwen2.5-32b-instruct",
                        help="Qwen model name (default: qwen2.5-32b-instruct)")
    parser.add_argument("--qwen_api_base", default=None,
                        help="API base URL for Qwen-compatible endpoint")
    parser.add_argument("--qwen_api_key_env", default="QWEN_API_KEY",
                        help="Env var name for API key (default: QWEN_API_KEY)")
    parser.add_argument("--qwen_temperature", type=float, default=0.0,
                        help="Generation temperature (default: 0.0)")
    parser.add_argument("--qwen_max_tokens", type=int, default=512,
                        help="Max output tokens (default: 512)")
    parser.add_argument("--qwen_timeout", type=float, default=60.0,
                        help="API timeout in seconds (default: 60)")
    parser.add_argument("--qwen_retries", type=int, default=1,
                        help="Retries on parse failure (default: 1)")
    parser.add_argument("--save_raw_qwen_outputs", action="store_true", default=False,
                        help="Save raw Qwen model outputs to disk")
    parser.add_argument("--fallback_parser", default="none",
                        choices=["none", "cached_qwen", "mock"],
                        help="Fallback parser if online_qwen fails (default: none)")
    parser.add_argument("--record_video", action="store_true", default=False,
                        help="Record off-screen video of each demo episode")
    parser.add_argument("--video_dir", default=None,
                        help="Output directory for videos (default: output_dir/videos)")
    parser.add_argument("--video_fps", type=int, default=20,
                        help="Frames per second for recorded video (default: 20)")
    parser.add_argument("--video_max_episodes", type=int, default=None,
                        help="Maximum number of episodes to record (default: all)")
    parser.add_argument("--instructions", default=None,
                        help="Path to JSON file with demo instructions (overrides hardcoded DEMOS)")
    parser.add_argument("--scene_inventory", default="none", choices=["none", "oracle"],
                        help="Scene inventory mode for object grounding (default: none)")
    parser.add_argument("--control_interface", default="legacy",
                        choices=["legacy", "skill_executor"],
                        help="Control interface: legacy runner or opt-in skill executor")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load external instructions if provided
    demos = list(DEMOS)
    if args.instructions:
        import json as _j
        instr_path = Path(args.instructions)
        if instr_path.exists():
            demos = _j.loads(instr_path.read_text())
            print(f"Loaded {len(demos)} instructions from {args.instructions}")
        else:
            print(f"Warning: --instructions file not found: {args.instructions}, using hardcoded DEMOS")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, aux = load_checkpoint(args.checkpoint, device)

    # Load action norm
    norm_path = PROJECT_ROOT / "generated/results/stage_7B-1/action_norm_stats.json"
    if norm_path.exists():
        import json as j
        stats = j.loads(norm_path.read_text())
        action_mean = np.array(stats["action_mean"], dtype=np.float32)
        action_std = np.array(stats["action_std"], dtype=np.float32)
        print(f"Loaded action norm from {norm_path}")
    else:
        action_mean = None
        action_std = None
        print("No action norm stats found — using raw BC output")

    # Load exec config
    exec_config = None
    if args.config:
        import json as j
        config_path = Path(args.config)
        if config_path.exists():
            exec_config = j.loads(config_path.read_text())
            print(f"Loaded config from {config_path}")

    # Planner
    planner = PlannerAdapter("bfs")

    # Parser
    fallback_parser_obj = None
    if args.parser == "cached_qwen":
        if args.cached_outputs is None:
            print("Error: --cached_outputs is required when --parser cached_qwen")
            return
        goal_parser = CachedQwenGoalParser(args.cached_outputs)
        print(f"Using cached Qwen parser: {args.cached_outputs}")
    elif args.parser == "online_qwen":
        import os as _os
        api_key = _os.environ.get(args.qwen_api_key_env)
        if not api_key:
            print(f"Error: {args.qwen_api_key_env} environment variable not set. "
                  f"Set it to your Qwen API key or use --parser mock/cached_qwen.")
            return
        raw_dir = str(output_dir / "raw_qwen_outputs") if args.save_raw_qwen_outputs else None
        goal_parser = OnlineQwenGoalParser(
            model=args.qwen_model,
            api_base=args.qwen_api_base,
            api_key_env=args.qwen_api_key_env,
            temperature=args.qwen_temperature,
            max_tokens=args.qwen_max_tokens,
            timeout=args.qwen_timeout,
            retries=args.qwen_retries,
            save_raw_outputs=args.save_raw_qwen_outputs,
            raw_output_dir=raw_dir,
        )
        print(f"Using online Qwen parser: model={args.qwen_model}")
        # Set up fallback
        if args.fallback_parser == "cached_qwen":
            if args.cached_outputs is None:
                print("Warning: fallback_parser=cached_qwen but --cached_outputs not set; no fallback")
            else:
                fallback_parser_obj = CachedQwenGoalParser(args.cached_outputs)
                print(f"  Fallback: cached_qwen ({args.cached_outputs})")
        elif args.fallback_parser == "mock":
            fallback_parser_obj = MockGoalParser()
            print("  Fallback: mock")
    else:
        goal_parser = MockGoalParser()
        print("Using MockGoalParser")

    # Video recording setup
    record_video = args.record_video
    video_dir = Path(args.video_dir) if args.video_dir else (output_dir / "videos")
    video_fps = args.video_fps
    video_max_episodes = args.video_max_episodes
    if record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        parser_tag = args.parser

    # Process each demo
    all_results = []
    for i, demo in enumerate(demos):
        print(f"\n{'='*60}")
        print(f"Demo {i+1}/{len(demos)}: {demo['instruction']}")
        print(f"{'='*60}")

        # Determine render mode
        render_mode = "offscreen" if (record_video and (video_max_episodes is None or i < video_max_episodes)) else "none"

        # Create fresh env per demo
        cubes_for_env = ["cube1", "cube2"]  # canonical cube names for env
        env = RobosuiteEnvWrapper(
            env_name="Stack",
            robots="Panda",
            cube_names=cubes_for_env,
            render_mode=render_mode,
        )
        env.reset()

        # Scene inventory (oracle perception)
        scene_description = None
        scene_info = None
        if args.scene_inventory == "oracle":
            try:
                scene_info = perceive_scene(env)
                scene_description = build_scene_description(scene_info)
            except Exception as e:
                print(f"  Scene inventory: error ({e})")

        # Video frame collector
        video_frames = [] if render_mode == "offscreen" else None

        bc_phases_for_demo = None if args.bc_phases == ["none"] else args.bc_phases
        result = run_demo(
            env, demo, planner, model, device, action_mean, action_std,
            exec_config, output_dir, use_bc=args.use_bc,
            bc_phases=bc_phases_for_demo,
            parser=goal_parser,
            record_video=(video_frames is not None),
            video_frames=video_frames,
            fallback_parser=fallback_parser_obj if args.parser == "online_qwen" else None,
            scene_description=scene_description,
            scene_info=scene_info,
            control_interface=args.control_interface,
        )
        env.env.close()

        # Write video if frames were collected
        video_path = None
        video_error = None
        if video_frames is not None and len(video_frames) > 1:
            outcome = "success" if result["success"] else "failure"
            video_filename = f"e2e_{parser_tag}_{outcome}_ep{i+1:03d}_v0_1_1.mp4"
            video_path = video_dir / video_filename
            video_error = _write_video_safe(video_frames, video_path, video_fps)
            if video_error is None:
                result["video_recorded"] = True
                result["video_path"] = str(video_path.relative_to(output_dir.parent.parent.parent))
            else:
                video_path = None
                result["video_error"] = video_error
        elif video_frames is not None:
            video_error = "no frames captured (render returned None)"
            result["video_error"] = video_error

        status = "✅" if result["success"] else "❌"
        ep = result.get("episode_result") or {}
        print(f"  Status: {status}")
        print(f"  Alignment: {result.get('alignment_status', 'N/A')}")
        print(f"  Plan: {result.get('plan_status', 'N/A')}")
        print(f"  Success: {result['success']}")
        if ep:
            print(f"  Steps: {ep.get('steps', '?')}, BC steps: {ep.get('bc_steps', '?')}")
            print(f"  Failure: {ep.get('failure_type', '?')}")
        print(f"  Policy: {result.get('policy_used', '?')}")
        if video_path:
            print(f"  Video: {video_path}")
        elif video_error:
            print(f"  Video: error ({video_error})")

        all_results.append(result)

        # Save individual demo result
        demo_out = output_dir / f"demo_{i+1}.json"
        demo_out.write_text(json.dumps(result, indent=2, default=str))

    # Summary
    n_success = sum(1 for r in all_results if r["success"])
    print(f"\n{'='*60}")
    print(f"E2E Demo Summary: {n_success}/{len(demos)} successful ({100*n_success/len(demos):.1f}%)")
    print(f"{'='*60}")
    for i, r in enumerate(all_results):
        ep = r.get("episode_result") or {}
        steps = ep.get("steps", "?")
        fail = ep.get("failure_type", "?")
        print(f"  {i+1}. {'✅' if r['success'] else '❌'} {r['instruction'][:50]:50s} steps={steps} fail={fail}")

    # Write summary
    summary = {
        "num_demos": len(demos),
        "success_count": n_success,
        "success_rate": n_success / len(demos),
        "results": all_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
