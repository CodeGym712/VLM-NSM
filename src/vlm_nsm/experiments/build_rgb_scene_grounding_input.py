"""Build RGB scene grounding eval input with rendered images.

Reads the Stage 17 scene grounding parse50 instructions, renders matching
images via Robosuite with specific cube positions, and outputs an eval input
JSON with image_path fields for the RGB detector pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def build_rgb_scene_grounding_input(
    instructions_path: str,
    output_path: str,
    image_output_dir: str,
    num_instructions: int | None = None,
    seed: int = 42,
) -> dict:
    """Render matching images for each instruction and create eval input.

    Args:
        instructions_path: Path to the base instruction JSON.
        output_path: Output path for the new eval input JSON.
        image_output_dir: Directory to save rendered images.
        num_instructions: Limit to first N instructions (for testing).
        seed: Random seed for env reset.

    Returns:
        Metadata dict.
    """
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.perception.oracle_perception import perceive_scene

    # Load original instructions
    with open(instructions_path) as f:
        instructions = json.load(f)

    if num_instructions:
        instructions = instructions[:num_instructions]

    output_root = Path(output_path).parent
    output_root.mkdir(parents=True, exist_ok=True)
    img_dir = Path(image_output_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Mapping from instruction object_id to sim joint
    CUBE_TO_JOINT = {
        "cube1": "cubeA_joint0",
        "cube2": "cubeB_joint0",
    }

    # Create env
    env = RobosuiteEnvWrapper(
        cube_names=["cube1", "cube2"],
        render_mode="offscreen",
    )

    start_time = time.time()
    new_instructions = []
    n_ok = 0
    n_skip = 0

    try:
        for i, item in enumerate(instructions):
            instruction = item.get("instruction", "")
            scene_info = item.get("scene_info")

            if not scene_info:
                print(f"  [{i + 1}] SKIP (no scene_info): {instruction[:60]}")
                # Still add to output but without image_path
                new_instructions.append(item)
                n_skip += 1
                continue

            # Reset env
            np.random.seed(seed + i)
            env.reset()
            sim = env.env.sim

            # Set cube positions from scene_info
            for obj in scene_info.get("objects", []):
                oid = obj.get("object_id")
                pos = obj.get("position")
                joint_name = CUBE_TO_JOINT.get(oid)
                if joint_name and pos and len(pos) >= 3:
                    # qpos: x, y, z, qw, qx, qy, qz — quat identity for upright
                    sim.data.set_joint_qpos(
                        joint_name,
                        [float(pos[0]), float(pos[1]), float(pos[2]),
                         1.0, 0.0, 0.0, 0.0],
                    )

            sim.forward()

            # Verify positions (optional)
            # for obj in scene_info.get("objects", []):
            #     oid = obj.get("object_id")
            #     body_name = env._cube_to_body.get(oid)
            #     if body_name:
            #         body_id = sim.model.body_name2id(body_name.encode())
            #         actual = sim.data.body_xpos[body_id]
            #         print(f"    {oid}: set to {obj.get('position')}, actual={actual}")

            # Render
            rgb = env.render_rgb(camera="agentview", height=512, width=512)
            if rgb is None:
                print(f"  [{i + 1}] SKIP (render failed): {instruction[:60]}")
                new_instructions.append(item)
                n_skip += 1
                continue

            # Save image
            image_filename = f"scene_{i:04d}.png"
            image_path = img_dir / image_filename
            img = Image.fromarray(rgb.astype("uint8"))
            img.save(str(image_path))

            # Create augmented instruction entry
            new_item = dict(item)
            new_item["image_path"] = str(image_path.absolute())
            new_instructions.append(new_item)
            n_ok += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(instructions)}] ... {n_ok} rendered")

    finally:
        env.env.close()

    elapsed = time.time() - start_time

    # Write new eval input JSON
    with open(output_path, "w") as f:
        json.dump(new_instructions, f, indent=2, ensure_ascii=False)

    print(f"\nRGB scene grounding input built:")
    print(f"  Instructions: {len(new_instructions)} ({n_ok} with images, {n_skip} skipped)")
    print(f"  Images: {image_output_dir}/")
    print(f"  Output: {output_path}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "total": len(new_instructions),
        "with_images": n_ok,
        "skipped": n_skip,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build RGB scene grounding eval input with rendered images",
    )
    parser.add_argument(
        "--instructions",
        default="generated/eval_inputs/stage_17/scene_grounding_parse50.json",
        help="Base instruction JSON path",
    )
    parser.add_argument(
        "--output",
        default="generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json",
        help="Output eval input JSON path",
    )
    parser.add_argument(
        "--image_dir",
        default="generated/eval_inputs/stage_18/images",
        help="Directory for rendered images",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        default=None,
        help="Limit to first N instructions (for testing)",
    )
    args = parser.parse_args()

    build_rgb_scene_grounding_input(
        instructions_path=args.instructions,
        output_path=args.output,
        image_output_dir=args.image_dir,
        num_instructions=args.num_instructions,
    )


if __name__ == "__main__":
    main()
