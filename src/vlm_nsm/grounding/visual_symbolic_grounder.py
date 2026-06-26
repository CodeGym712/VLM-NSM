"""Visual-symbolic grounder: resolves referring expressions to object IDs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlm_nsm.grounding.referring_expressions import ReferringExpression, ExpressionType
from vlm_nsm.symbolic.predicates import SymbolicSceneGraph


@dataclass
class GroundingResult:
    expression_text: str = ""
    expression_type: str = ""
    selected_object_id: str | None = None
    confidence: float = 0.0
    status: str = "unknown"  # grounded/ambiguous/impossible/multiple
    candidates: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression_text, "type": self.expression_type,
            "selected": self.selected_object_id, "confidence": self.confidence,
            "status": self.status, "candidates": self.candidates,
            "scores": self.scores, "reason": self.reason,
        }


class VisualSymbolicGrounder:
    """Resolves referring expressions using visual and symbolic scene information."""

    def __init__(self, color_map: dict[str, str] | None = None):
        self.color_map = color_map or {"red": "cube1", "green": "cube2", "blue": "cube3"}

    def ground(self, expr: ReferringExpression, scene_state, graph: SymbolicSceneGraph) -> GroundingResult:
        if expr.expr_type == ExpressionType.COLOR:
            return self._ground_color(expr, scene_state, graph)
        elif expr.expr_type == ExpressionType.SPATIAL_ABSOLUTE:
            return self._ground_spatial_absolute(expr, scene_state, graph)
        elif expr.expr_type == ExpressionType.SPATIAL_RELATIVE:
            return self._ground_spatial_relative(expr, scene_state, graph)
        elif expr.expr_type == ExpressionType.RELATIONAL:
            return self._ground_relational(expr, scene_state, graph)
        elif expr.expr_type == ExpressionType.AMBIGUOUS:
            return GroundingResult(expression_text=expr.text, expression_type="ambiguous",
                                  status="ambiguous", reason="explicitly ambiguous expression")
        elif expr.expr_type == ExpressionType.IMPOSSIBLE:
            return GroundingResult(expression_text=expr.text, expression_type="impossible",
                                  status="impossible", reason="explicitly impossible expression")
        return GroundingResult(status="unknown", reason=f"unsupported type: {expr.expr_type}")

    def _ground_color(self, expr, scene_state, graph):
        result = GroundingResult(expression_text=expr.text, expression_type="color")
        color = expr.metadata.get("color", "").lower()
        obj_id = self.color_map.get(color)
        if obj_id:
            result.selected_object_id = obj_id
            result.confidence = 1.0
            result.status = "grounded"
            result.reason = f"color '{color}' → {obj_id}"
        else:
            result.status = "impossible"
            result.reason = f"unknown color '{color}'"
        return result

    def _ground_spatial_absolute(self, expr, scene_state, graph):
        result = GroundingResult(expression_text=expr.text, expression_type="spatial_absolute")
        attr = expr.spatial_attribute or ""
        obj_ids = list(graph.objects.keys())

        positions = {}
        for obj_id in obj_ids:
            obj = scene_state.get_object(obj_id)
            if obj.available and obj.position is not None:
                positions[obj_id] = obj.position

        if not positions:
            result.status = "ambiguous"
            result.reason = "no objects with known positions"
            return result

        if attr == "leftmost":
            selected = min(positions, key=lambda k: positions[k][0])
        elif attr == "rightmost":
            selected = max(positions, key=lambda k: positions[k][0])
        elif attr == "closest_to_robot":
            selected = min(positions, key=lambda k: abs(positions[k][1]))
        elif attr == "farthest_from_robot":
            selected = max(positions, key=lambda k: abs(positions[k][1]))
        elif attr == "frontmost":
            selected = min(positions, key=lambda k: positions[k][1])
        elif attr == "backmost":
            selected = max(positions, key=lambda k: positions[k][1])
        elif attr == "highest":
            selected = max(positions, key=lambda k: positions[k][2])
        elif attr == "lowest":
            selected = min(positions, key=lambda k: positions[k][2])
        else:
            result.status = "unknown"
            result.reason = f"unsupported spatial attribute: {attr}"
            return result

        result.selected_object_id = selected
        result.confidence = 0.9
        result.status = "grounded"
        result.reason = f"{attr} object is {selected}"
        return result

    def _ground_spatial_relative(self, expr, scene_state, graph):
        result = GroundingResult(expression_text=expr.text, expression_type="spatial_relative")
        ref_obj = expr.reference_object_id
        attr = expr.spatial_attribute or "closest"

        ref_pos = None
        if ref_obj:
            ref = scene_state.get_object(ref_obj)
            if ref.available and ref.position is not None:
                ref_pos = ref.position

        if ref_pos is None:
            result.status = "impossible"
            result.reason = f"reference object '{ref_obj}' unavailable"
            return result

        distances = {}
        for obj_id in graph.objects:
            if obj_id == ref_obj:
                continue
            obj = scene_state.get_object(obj_id)
            if obj.available and obj.position is not None:
                distances[obj_id] = float(np.linalg.norm(obj.position - ref_pos))

        if not distances:
            result.status = "ambiguous"
            result.reason = "no other objects with known positions"
            return result

        selected = min(distances, key=distances.get)
        result.selected_object_id = selected
        result.confidence = 0.85
        result.status = "grounded"
        result.scores = {k: 1.0 - v/max(distances.values()) for k, v in distances.items()}
        result.reason = f"{attr} to {ref_obj} is {selected} (dist={distances[selected]:.3f}m)"
        return result

    def _ground_relational(self, expr, scene_state, graph):
        result = GroundingResult(expression_text=expr.text, expression_type="relational")
        relation = expr.relation or ""
        candidates = graph.get_objects_satisfying(relation)
        if len(candidates) == 1:
            result.selected_object_id = candidates[0]
            result.confidence = 1.0
            result.status = "grounded"
            result.reason = f"unique {relation} object: {candidates[0]}"
        elif len(candidates) > 1:
            result.candidates = candidates
            result.status = "multiple"
            result.reason = f"multiple {relation} candidates: {candidates}"
        else:
            result.status = "impossible"
            result.reason = f"no object satisfies {relation}"
        return result
