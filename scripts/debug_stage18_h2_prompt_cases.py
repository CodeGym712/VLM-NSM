"""Debug script: show Stage 18-H2 scene descriptions for all targeted cases.

Generates scene descriptions using the updated prompt (shared predicate ambiguity
section, color synonym rules, pairwise robot-relative relations) for the 8
targeted referential/grounding cases.
"""
from __future__ import annotations

from pathlib import Path

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description


def _make_two_cube_scene() -> SceneInfo:
    """Standard 2-cube scene used in Stage 18 eval (red + green cubes on table)."""
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
                    "world_relations": {"cube2": ["right_of"]},
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
                    "world_relations": {"cube1": ["left_of"]},
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

    output_path = Path("generated/results/stage_18/stage18_h2_prompt_cases.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Stage 18-H2 Prompt Debug: Targeted Cases")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Full scene description (with H2-1 + H2-2 improvements):")
    lines.append("-" * 72)
    lines.append(desc)
    lines.append("-" * 72)
    lines.append("")

    cases = [
        {
            "instruction": "Put the cube that is on the table onto the other cube.",
            "audit": "objectively_ambiguous (on_table)",
            "expected": "ungroundable",
            "check": [
                "Shared predicate ambiguity: ✓ present" if "Shared predicate ambiguity:" in desc else "Shared predicate ambiguity: ✗ MISSING",
                "on_table shared note: " + ("✓ present" if "on_table is shared" in desc else "✗ MISSING"),
                "MUST be ungroundable: " + ("✓ present" if "MUST be ungroundable" in desc else "✗ MISSING"),
            ],
        },
        {
            "instruction": "Stack the block that is on the table on top of the other block.",
            "audit": "objectively_ambiguous (on_table)",
            "expected": "ungroundable",
            "check": [
                "Same on_table ambiguity check as above",
            ],
        },
        {
            "instruction": "Put the one that is closer to the edge onto the other one.",
            "audit": "objectively_ambiguous (table_edge)",
            "expected": "ungroundable",
            "check": [
                "No object near edge note: " + ("✓ present" if "No object is near any table edge" in desc else "✗ MISSING"),
            ],
        },
        {
            "instruction": "Take the first cube you see and put it on the second one.",
            "audit": "unsupported_reference_frame (visual_order)",
            "expected": "ungroundable",
            "check": [
                "first you see guidance: " + ("✓ present" if "first you see" in desc else "✗ MISSING"),
            ],
        },
        {
            "instruction": "Stack the reddish block on the greenish block.",
            "audit": "color (should ground: reddish→red, greenish→green)",
            "expected": "grounded (cube1 on cube2)",
            "check": [
                "Color reference rules: " + ("✓ present" if "Color reference rules:" in desc else "✗ MISSING"),
                "reddish→red rule: " + ("✓ present" if "reddish" in desc else "✗ MISSING"),
                "greenish→green rule: " + ("✓ present" if "greenish" in desc else "✗ MISSING"),
            ],
        },
        {
            "instruction": "Put the lighter colored cube on the darker one.",
            "audit": "unsupported_reference_frame (color_attribute)",
            "expected": "ungroundable",
            "check": [
                "Brightness unsupported: " + ("✓ present" if "Do NOT infer brightness" in desc else "✗ MISSING"),
            ],
        },
        {
            "instruction": "Place the bright cube on the dull cube.",
            "audit": "unsupported_reference_frame (color_attribute)",
            "expected": "ungroundable",
            "check": [
                "lighter/darker/bright/dull rule: " + ("✓ present" if "lighter/darker/bright/dull" in desc else "✗ MISSING"),
            ],
        },
        {
            "instruction": "Place the nearer cube onto the farther cube.",
            "audit": "objectively_groundable (robot_relative_near_far)",
            "expected": "grounded (cube1 on cube2)",
            "check": [
                "nearer_to_robot_than: " + ("✓ present" if "nearer_to_robot_than" in desc else "✗ MISSING"),
                "farther_from_robot_than: " + ("✓ present" if "farther_from_robot_than" in desc else "✗ MISSING"),
                "Robot-relative ordering: " + ("✓ present" if "Robot-relative (near to far):" in desc else "✗ MISSING"),
            ],
        },
    ]

    for i, case in enumerate(cases, 1):
        lines.append(f"--- Case {i}: {case['instruction']} ---")
        lines.append(f"  Audit:    {case['audit']}")
        lines.append(f"  Expected: {case['expected']}")
        for check in case["check"]:
            lines.append(f"  {check}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("Summary of key sections in scene description:")
    lines.append("=" * 72)
    key_sections = [
        "Shared predicate ambiguity:",
        "Color reference rules:",
        "nearer_to_robot_than",
        "farther_from_robot_than",
        "Robot-relative (near to far):",
    ]
    for section in key_sections:
        status = "✓" if section in desc else "✗"
        lines.append(f"  {status} {section}")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nOutput written to {output_path}")


if __name__ == "__main__":
    main()
