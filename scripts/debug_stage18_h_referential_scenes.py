"""Debug script: show improved scene descriptions for Stage 18-H referential failure cases.

Simulates the scene info the LLM would see (with the H3 prompt strengthening)
for each of the 5 referential failure cases from Stage 18-G.
"""
from __future__ import annotations

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description


def _make_two_cube_scene() -> SceneInfo:
    """Simulate the standard 2-cube scene from Stage 18 eval (red + green cubes)."""
    return SceneInfo(objects=[
        DetectedObject(
            label="cube1", category="block", color="red",
            pose={"x": 0.101, "y": 0.001, "z": 0.825},
            attributes={
                "on_table": True,
                "near_edges": [],
                "robot_distance": 0.40,
                "robot_direction": ["robot-center-x", "toward-robot"],
                "pose_in_workspace": True,
                "pose_suspicious": False,
                "position_source": "rgb_dual_camera_triangulation",
                "pose_confidence": 0.78,
                "spatial": {
                    "world_relations": {
                        "cube2": ["right_of"],
                    },
                    "ambiguous_y": ["cube2"],
                    "center_norm": (0.50, 0.32),
                    "image_position": "image-center",
                },
            },
        ),
        DetectedObject(
            label="cube2", category="block", color="green",
            pose={"x": -0.098, "y": 0.001, "z": 0.825},
            attributes={
                "on_table": True,
                "near_edges": [],
                "robot_distance": 0.60,
                "robot_direction": ["robot-center-x", "toward-robot"],
                "pose_in_workspace": True,
                "pose_suspicious": False,
                "position_source": "rgb_dual_camera_triangulation",
                "pose_confidence": 0.41,
                "spatial": {
                    "world_relations": {
                        "cube1": ["left_of"],
                    },
                    "ambiguous_y": ["cube1"],
                    "center_norm": (0.50, 0.56),
                    "image_position": "image-center",
                },
            },
        ),
    ])


def main():
    scene = _make_two_cube_scene()
    desc = build_scene_description(scene)

    print("=" * 72)
    print("Simulated Scene Description (with H3 prompt strengthening)")
    print("=" * 72)
    print(desc)
    print()
    print("=" * 72)
    print("Referential Failure Case Analysis")
    print("=" * 72)

    cases = [
        {
            "instruction": "Place the nearer cube onto the farther cube.",
            "audit": "objectively_groundable (robot_relative_near_far)",
            "expected": "top=cube1 (0.40m, nearer), bottom=cube2 (0.60m, farther)",
            "stage18g": "WRONG: swapped to top=cube2, bottom=cube1",
            "improvement": "Pairwise 'nearer_to_robot_than' + 'Robot-relative ordering' "
                           "makes the near/far distinction explicit per-object and globally.",
        },
        {
            "instruction": "Put the cube that is on the table onto the other cube.",
            "audit": "objectively_ambiguous (on_table)",
            "expected": "ungroundable (BOTH cubes are on_table=True)",
            "stage18g": "UNSAFE GUESS: guessed cube2 on cube1",
            "improvement": "Guidance now explicitly says: 'If a referential expression uses "
                           "a property shared equally by multiple objects (e.g. both are "
                           "'on the table'), the reference is ambiguous and you MUST output "
                           "status=\"ungroundable\"'",
        },
        {
            "instruction": "Put the one that is closer to the edge onto the other one.",
            "audit": "objectively_ambiguous (table_edge)",
            "expected": "ungroundable (neither cube is near any edge)",
            "stage18g": "UNSAFE GUESS: guessed cube2 on cube1",
            "improvement": "No cube has near_edges set. The guidance now covers this: "
                           "shared property (no edge proximity) -> ambiguous -> ungroundable.",
        },
        {
            "instruction": "Stack the block that is on the table on top of the other block.",
            "audit": "objectively_ambiguous (on_table)",
            "expected": "ungroundable (BOTH blocks are on_table=True)",
            "stage18g": "UNSAFE GUESS: guessed cube2 on cube1",
            "improvement": "Same as ref_003 — shared 'on table' property -> MUST reject.",
        },
        {
            "instruction": "Take the first cube you see and put it on the second one.",
            "audit": "unsupported_reference_frame (visual_order)",
            "expected": "ungroundable (no visual scanning order defined)",
            "stage18g": "UNSAFE GUESS: guessed cube2 on cube1",
            "improvement": "Guidance now says: 'If a referential expression requires "
                           "a convention not provided (e.g. 'first you see'), output "
                           "status=\"ungroundable\"'",
        },
    ]

    for i, case in enumerate(cases, 1):
        print(f"\n--- Case {i}: {case['instruction']} ---")
        print(f"  Audit:         {case['audit']}")
        print(f"  Expected:      {case['expected']}")
        print(f"  Stage 18-G:    {case['stage18g']}")
        print(f"  H3 fix:        {case['improvement']}")

    print()
    print("=" * 72)
    print("Key Changes (H3 vs Stage 18-G):")
    print("  1. Pairwise robot-relative relations in per-object descriptions")
    print("     (nearer_to_robot_than / farther_from_robot_than)")
    print("  2. Robot-relative ordering in table layout")
    print("     (near to far: cube1 (0.40m) < cube2 (0.60m))")
    print("  3. Explicit ambiguous reference rejection guidance")
    print("     (shared property -> MUST output ungroundable)")
    print("  4. Explicit unsupported convention rejection guidance")
    print("     (convention not provided -> output ungroundable)")
    print("=" * 72)


if __name__ == "__main__":
    main()
