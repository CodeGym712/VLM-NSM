#!/usr/bin/env python3
"""Extract camera calibration from a Robosuite Stack environment and save as JSON.

Output is a JSON file containing K, T_camera_world, width, height, camera_name.
This can be passed to eval_online_qwen_parser.py via --camera_calibration.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / ".."))

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.perception.robosuite_camera_utils import get_camera_info


def main():
    parser = argparse.ArgumentParser(description="Extract camera calibration from Robosuite")
    parser.add_argument("--camera", default="agentview", help="Camera name")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--output", default="generated/configs/camera_calibration_agentview_512.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("Creating Robosuite environment...")
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        render_mode="offscreen",
        control_freq=20,
    )

    print(f"Getting camera info for '{args.camera}' ({args.width}x{args.height})...")
    info = get_camera_info(env, args.camera, args.width, args.height)

    calib = {
        "K": info.K.tolist(),
        "T_camera_world": info.T_camera_world.tolist(),
        "width": info.width,
        "height": info.height,
        "camera_name": info.camera_name,
        "convention": {
            "reshape_mode": "row_major",
            "R_direction": "cam2world",
            "y_flip": True,
            "z_flip": False,
            "name": "row_major_cam2world_yflip",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calib, indent=2))
    print(f"Calibration saved to {output_path}")
    print(f"  K:\n{info.K}")
    print(f"  T_camera_world:\n{info.T_camera_world}")

    env.env.close()


if __name__ == "__main__":
    main()
