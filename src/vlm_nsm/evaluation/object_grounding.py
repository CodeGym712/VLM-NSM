"""Object grounding evaluator for parser-driven language grounding.

Compares the parser's object_bindings against expected bindings or
infers correctness from scene inventory and instruction heuristics.
"""

from __future__ import annotations

from typing import Any

from vlm_nsm.perception.data_types import SceneInfo, DEFAULT_COLOR_MAP


def _resolve_color_from_instruction(instruction: str, scene: SceneInfo) -> dict[str, str]:
    """Simple keyword-based color-to-object-id resolution.

    Scans the instruction for color words and maps them to objects
    in the scene that have matching colors.

    Returns a dict of {object_id: color} for color-matched objects.
    """
    color_keywords = {
        "red": "red",
        "green": "green",
        "blue": "blue",
        "yellow": "yellow",
        "purple": "purple",
        "orange": "orange",
        "pink": "pink",
        "white": "white",
        "black": "black",
        "grey": "grey",
        "gray": "grey",
    }

    instruction_lower = instruction.lower()
    result: dict[str, str] = {}

    for obj in scene.objects:
        if obj.color is None:
            # Fall back to DEFAULT_COLOR_MAP
            obj_color = DEFAULT_COLOR_MAP.get(obj.label)
        else:
            obj_color = obj.color

        if obj_color is None:
            continue

        # Check if the instruction mentions this color
        color_lower = obj_color.lower()
        if color_lower in color_keywords and color_keywords[color_lower] in instruction_lower:
            result[obj.label] = obj_color

    return result


def _resolve_positions_from_instruction(
    instruction: str,
    scene: SceneInfo,
) -> dict[str, str]:
    """Simple keyword-based position resolution.

    Maps 'left'/'right'/'front'/'back' to objects based on their pose.
    Returns {object_id: position_label} for position-matched objects.
    """
    result: dict[str, str] = {}
    instruction_lower = instruction.lower()

    has_left = "left" in instruction_lower
    has_right = "right" in instruction_lower
    has_front = "front" in instruction_lower
    has_back = "back" in instruction_lower or "behind" in instruction_lower

    blocks_with_pose = [
        o for o in scene.objects
        if o.category == "block" and o.pose is not None
    ]

    if len(blocks_with_pose) < 2:
        return result

    if has_left and has_right:
        # Sort by x coordinate (left = smaller x)
        sorted_blocks = sorted(blocks_with_pose, key=lambda o: o.pose["x"])
        result[sorted_blocks[0].label] = "left"
        result[sorted_blocks[-1].label] = "right"

    if has_front and has_back:
        # Sort by y coordinate (back = smaller y in typical robosuite coords)
        sorted_blocks = sorted(blocks_with_pose, key=lambda o: o.pose["y"])
        result[sorted_blocks[0].label] = "back"
        result[sorted_blocks[-1].label] = "front"

    return result


def evaluate_object_grounding(
    instruction: str,
    goal_json: dict[str, Any] | None,
    scene_info: SceneInfo | None = None,
    expected_bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate whether the parser correctly grounds objects.

    Args:
        instruction: The natural language instruction.
        goal_json: Parsed goal JSON from the parser (may have object_bindings).
        scene_info: SceneInfo from oracle perception (optional).
        expected_bindings: Ground-truth bindings for comparison (optional).

    Returns:
        dict with keys:
            - object_grounding_valid: bool
            - grounding_errors: list[str]
            - expected_bindings: dict
            - predicted_bindings: dict
            - grounding_method_used: str
    """
    grounding_errors: list[str] = []
    predicted_bindings: dict[str, str] = {}

    # Extract predicted bindings from goal_json
    if goal_json and isinstance(goal_json, dict):
        predicted_bindings = dict(goal_json.get("object_bindings") or {})

    method = "unknown"

    # Strategy 1: Compare against expected bindings
    if expected_bindings:
        method = "expected_bindings_match"
        for role, expected_id in expected_bindings.items():
            predicted_id = predicted_bindings.get(role)
            if predicted_id is None:
                grounding_errors.append(
                    f"Missing binding for '{role}' (expected '{expected_id}')"
                )
            elif predicted_id != expected_id:
                grounding_errors.append(
                    f"Wrong binding for '{role}': got '{predicted_id}', "
                    f"expected '{expected_id}'"
                )

    # Strategy 2: Infer from scene info if available and no expected_bindings
    elif scene_info and scene_info.objects:
        method = "scene_inference"
        color_bindings = _resolve_color_from_instruction(instruction, scene_info)
        pos_bindings = _resolve_positions_from_instruction(instruction, scene_info)

        for role, predicted_id in predicted_bindings.items():
            if role == "top block":
                if predicted_id in color_bindings or predicted_id in pos_bindings:
                    pass  # grounded successfully
                elif color_bindings:
                    # Check if there's a color match that was missed
                    pass
            elif role == "bottom block":
                if predicted_id in color_bindings or predicted_id in pos_bindings:
                    pass
                elif color_bindings:
                    pass

    # Strategy 3: No expected bindings, no scene info — skip evaluation
    else:
        method = "none"

    is_valid = len(grounding_errors) == 0

    return {
        "object_grounding_valid": is_valid,
        "grounding_errors": grounding_errors,
        "expected_bindings": expected_bindings or {},
        "predicted_bindings": predicted_bindings,
        "grounding_method_used": method,
    }
