"""Grounding baselines: ColorOnly and Random."""

import random as _random

from vlm_nsm.grounding.referring_expressions import ReferringExpression, ExpressionType
from vlm_nsm.grounding.visual_symbolic_grounder import GroundingResult


class ColorOnlyGrounder:
    def __init__(self, color_map=None):
        self.color_map = color_map or {"red": "cube1", "green": "cube2"}

    def ground(self, expr: ReferringExpression, scene_state, graph) -> GroundingResult:
        if expr.expr_type == ExpressionType.COLOR:
            color = expr.metadata.get("color", "").lower()
            obj_id = self.color_map.get(color)
            if obj_id:
                return GroundingResult(expression_text=expr.text, expression_type="color",
                    selected_object_id=obj_id, confidence=1.0, status="grounded",
                    reason=f"color match: {color}→{obj_id}")
        return GroundingResult(expression_text=expr.text, expression_type=expr.expr_type.value,
            status="unknown", reason="color-only grounder cannot resolve")


class RandomGrounder:
    def __init__(self, seed=0):
        self.rng = _random.Random(seed)

    def ground(self, expr: ReferringExpression, scene_state, graph) -> GroundingResult:
        obj_ids = list(graph.objects.keys())
        if not obj_ids:
            return GroundingResult(status="impossible", reason="no objects")
        selected = self.rng.choice(obj_ids)
        return GroundingResult(expression_text=expr.text, expression_type="random",
            selected_object_id=selected, confidence=1.0/len(obj_ids),
            status="grounded", reason="random selection")
