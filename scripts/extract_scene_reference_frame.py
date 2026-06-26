#!/usr/bin/env python3
"""Extract scene reference frame (table geometry, robot pose) from Robosuite.

Saves a JSON file usable by eval_online_qwen_parser.py via
--scene_reference_frame, and by the predicate inventory system.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / ".."))

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.perception.scene_context import (
    extract_reference_frame_from_env,
    save_reference_frame,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract scene reference frame from Robosuite",
    )
    parser.add_argument(
        "--output",
        default="generated/configs/scene_reference_frame_stack.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    print("Creating Robosuite Stack environment...")
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        render_mode="offscreen",
        control_freq=20,
    )

    print("Extracting reference frame...")
    ref_frame = extract_reference_frame_from_env(env)

    save_reference_frame(args.output, ref_frame)

    print(f"Reference frame saved to {args.output}")
    print(f"  Table bounds X: [{ref_frame.table_x_min:.3f}, {ref_frame.table_x_max:.3f}]")
    print(f"  Table bounds Y: [{ref_frame.table_y_min:.3f}, {ref_frame.table_y_max:.3f}]")
    print(f"  Table surface Z: {ref_frame.table_z_surface:.3f}")
    print(f"  Robot base: ({ref_frame.robot_base_x:.3f}, {ref_frame.robot_base_y:.3f}, {ref_frame.robot_base_z:.3f})")
    print(f"  Edge threshold: {ref_frame.edge_threshold:.3f}")
    print(f"  Workspace padding: {ref_frame.workspace_padding:.3f}")

    env.env.close()


if __name__ == "__main__":
    main()
