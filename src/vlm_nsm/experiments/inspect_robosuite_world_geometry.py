#!/usr/bin/env python3
"""Inspect Robosuite world geometry: true table_z, object poses, camera positions.

Determines ground-truth table surface height, object body positions, and camera
extrinsics directly from the MuJoCo simulation state. Used to audit camera
calibration extraction and verify the table-plane projection assumption.

Output: generated/results/stage_18/robosuite_world_geometry_inspection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper


def _make_json_safe(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return obj


def inspect_world_geometry(output_path: str) -> dict:
    """Inspect Robosuite world geometry and write results to JSON.

    Returns:
        Dict with table info, cube positions, camera extrinsics, and metadata.
    """
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        cube_names=["cube1", "cube2"],
        render_mode="offscreen",
    )

    try:
        env.reset()
        sim = env.env.sim

        result: dict = {}

        # --- Table info ---
        table_body_id = sim.model.body_name2id("table")
        table_pos = sim.data.body_xpos[table_body_id].copy()
        result["table"] = {
            "body_name": "table",
            "position": {"x": float(table_pos[0]), "y": float(table_pos[1]), "z": float(table_pos[2])},
            "note": "body_xpos is body centre; table surface may differ by half-height",
        }

        # Also get the table geom to find surface
        table_geom_info = []
        for i in range(sim.model.ngeom):
            body_id = sim.model.geom_bodyid[i]
            if body_id == table_body_id:
                geom_name = sim.model.geom_id2name(i)
                if isinstance(geom_name, bytes):
                    geom_name = geom_name.decode()
                geom_pos = sim.data.geom_xpos[i].copy()
                geom_size = sim.model.geom_size[i].copy()
                table_geom_info.append({
                    "geom_name": str(geom_name) if geom_name else f"geom_{i}",
                    "position": {"x": float(geom_pos[0]), "y": float(geom_pos[1]), "z": float(geom_pos[2])},
                    "size": _make_json_safe(geom_size),
                    "type": int(sim.model.geom_type[i]),
                })
        result["table"]["geoms"] = table_geom_info

        # --- Cube positions ---
        cubes = {}
        for canonical in env.cube_names:
            body_name = env._cube_to_body.get(canonical)
            if body_name is None:
                cubes[canonical] = {"error": "not found"}
                continue
            try:
                body_id = sim.model.body_name2id(body_name)
                pos = sim.data.body_xpos[body_id].copy()
                cubes[canonical] = {
                    "body_name": str(body_name),
                    "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                    "note": "body_xpos is cube centre; bottom = z - half_height",
                }
                # Get cube geom info for size
                cube_geoms = []
                for i in range(sim.model.ngeom):
                    gbody_id = sim.model.geom_bodyid[i]
                    if gbody_id == body_id:
                        gpos = sim.data.geom_xpos[i].copy()
                        gsize = sim.model.geom_size[i].copy()
                        cube_geoms.append({
                            "position": {"x": float(gpos[0]), "y": float(gpos[1]), "z": float(gpos[2])},
                            "size": _make_json_safe(gsize),
                        })
                cubes[canonical]["geoms"] = cube_geoms
            except ValueError:
                cubes[canonical] = {"error": f"body '{body_name}' not found"}
        result["cubes"] = cubes

        # --- All body names (for reference) ---
        result["all_body_names"] = [
            str(sim.model.body_names[i]) for i in range(sim.model.nbody)
        ]

        # --- Camera extrinsics ---
        cameras = {}
        camera_names = ["agentview", "frontview", "birdview", "sideview",
                        "robot0_robotview", "robot0_eye_in_hand"]
        for cam_name in camera_names:
            try:
                cam_id = sim.model.camera_name2id(cam_name)
                xpos = sim.data.cam_xpos[cam_id].copy()
                xmat_flat = sim.data.cam_xmat[cam_id].copy()
                xmat = xmat_flat.reshape(3, 3)
                fovy = float(sim.model.cam_fovy[cam_id])

                # Build T_camera_world: [R | t; 0 0 0 1]
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = xmat
                T[:3, 3] = xpos

                # Compute camera look direction (third column of R = camera Z axis in world)
                look_dir = xmat[:, 2].copy()

                cameras[cam_name] = {
                    "position": {"x": float(xpos[0]), "y": float(xpos[1]), "z": float(xpos[2])},
                    "xmat_row_major": _make_json_safe(xmat_flat),
                    "xmat_3x3": _make_json_safe(xmat),
                    "fov_y_deg": fovy,
                    "look_direction": {
                        "x": float(look_dir[0]), "y": float(look_dir[1]), "z": float(look_dir[2]),
                    },
                    "T_camera_world": _make_json_safe(T),
                }
            except Exception as e:
                cameras[cam_name] = {"error": str(e)}
        result["cameras"] = cameras

        # --- Camera name ID mapping ---
        cam_name_to_id = {}
        for i in range(sim.model.ncam):
            name = sim.model.camera_id2name(i)
            if name:
                cam_name_to_id[str(name)] = i
        result["camera_name_to_id"] = cam_name_to_id

        # --- Summary ---
        result["summary"] = {
            "table_z_from_body": float(table_pos[2]),
            "num_cubes": len(cubes),
            "num_cameras_found": sum(1 for c in cameras.values() if "error" not in c),
            "num_bodies_total": int(sim.model.nbody),
        }

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"World geometry inspection written to {output_path}")

        # Summary to stdout
        print(f"\nTable position: {result['table']['position']}")
        for name, cube in cubes.items():
            if "position" in cube:
                print(f"{name}: {cube['position']}")
        for name, cam in cameras.items():
            if "position" in cam:
                look = cam["look_direction"]
                print(f"{name}: pos={cam['position']}, look_dir=({look['x']:.4f}, {look['y']:.4f}, {look['z']:.4f})")

        return result

    finally:
        env.env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Robosuite world geometry for camera calibration audit",
    )
    parser.add_argument(
        "--output",
        default="generated/results/stage_18/robosuite_world_geometry_inspection.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    inspect_world_geometry(args.output)


if __name__ == "__main__":
    main()
