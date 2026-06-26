"""Referring expression types and parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ExpressionType(Enum):
    COLOR = "color"
    SPATIAL_ABSOLUTE = "spatial_absolute"
    SPATIAL_RELATIVE = "spatial_relative"
    RELATIONAL = "relational"
    AMBIGUOUS = "ambiguous"
    IMPOSSIBLE = "impossible"


REFERRING_EXPRESSION_TYPES = [e.value for e in ExpressionType]


@dataclass
class ReferringExpression:
    text: str
    expr_type: ExpressionType
    target_object_id: str | None = None  # None for ambiguous/impossible
    reference_object_id: str | None = None  # for relative/relational
    spatial_attribute: str | None = None  # leftmost/rightmost/closest/farthest
    relation: str | None = None  # on/clear/ontable/graspable
    expected_grounding_status: str = "grounded"  # grounded/ambiguous/impossible/multiple
    expected_candidates: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.expr_type.value,
            "target_object_id": self.target_object_id,
            "reference_object_id": self.reference_object_id,
            "expected_grounding_status": self.expected_grounding_status,
        }


def parse_color_expr(text: str, target: str, color: str) -> ReferringExpression:
    return ReferringExpression(text=text, expr_type=ExpressionType.COLOR,
                              target_object_id=target, metadata={"color": color},
                              expected_grounding_status="grounded")


def parse_spatial_absolute(text: str, target: str, attr: str) -> ReferringExpression:
    return ReferringExpression(text=text, expr_type=ExpressionType.SPATIAL_ABSOLUTE,
                              target_object_id=target, spatial_attribute=attr,
                              expected_grounding_status="grounded")


def parse_spatial_relative(text: str, target: str, ref_obj: str, attr: str) -> ReferringExpression:
    return ReferringExpression(text=text, expr_type=ExpressionType.SPATIAL_RELATIVE,
                              target_object_id=target, reference_object_id=ref_obj,
                              spatial_attribute=attr, expected_grounding_status="grounded")


def parse_relational(text: str, target: str, relation: str) -> ReferringExpression:
    return ReferringExpression(text=text, expr_type=ExpressionType.RELATIONAL,
                              target_object_id=target, relation=relation,
                              expected_grounding_status="grounded")


def parse_ambiguous(text: str) -> ReferringExpression:
    return ReferringExpression(text=text, expr_type=ExpressionType.AMBIGUOUS,
                              target_object_id=None, expected_grounding_status="ambiguous")


def parse_impossible(text: str) -> ReferringExpression:
    return ReferringExpression(text=text, expr_type=ExpressionType.IMPOSSIBLE,
                              target_object_id=None, expected_grounding_status="impossible")
