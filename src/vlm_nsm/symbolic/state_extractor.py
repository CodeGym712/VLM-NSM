"""SceneState → SymbolicSceneGraph extractor."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vlm_nsm.symbolic.predicates import (
    ObjectSymbol, Predicate, SymbolicSceneGraph,
    on, ontable, clear, holding, handempty, reachable, graspable,
)


class SymbolicStateExtractor:
    """Extract symbolic predicates from SceneState (oracle or perception)."""

    def __init__(
        self,
        on_xy_tolerance: float = 0.05,
        on_z_tolerance: float = 0.02,
        table_height: float = 0.80,
        cube_height: float = 0.05,
        clear_height_margin: float = 0.03,
        workspace_bounds: tuple[float, float, float, float] = (-0.4, -0.4, 0.4, 0.4),
    ):
        self.on_xy_tolerance = on_xy_tolerance
        self.on_z_tolerance = on_z_tolerance
        self.table_height = table_height
        self.cube_height = cube_height
        self.clear_height_margin = clear_height_margin
        self.workspace_bounds = workspace_bounds

    def extract(self, scene_state, holding_object: str | None = None) -> SymbolicSceneGraph:
        """Convert a SceneState to SymbolicSceneGraph."""
        graph = SymbolicSceneGraph(
            source_scene_state_id=getattr(scene_state, "frame_id", "") or str(time.time()),
            metadata={"source": getattr(scene_state, "source", "unknown")},
        )

        # Build object symbols
        for name, obj in scene_state.objects.items():
            meta = getattr(obj, "metadata", {}) or {}
            graph.objects[name] = ObjectSymbol(
                object_id=name,
                color=meta.get("color", ""),
                pose_source=obj.source,
            )

        obj_ids = list(graph.objects.keys())

        # on(x, y): x is stacked on y
        for a in obj_ids:
            for b in obj_ids:
                if a == b:
                    continue
                if self._is_on(scene_state, a, b):
                    graph.add_predicate(on(a, b, src="extracted"))

        # ontable(x)
        for a in obj_ids:
            if self._is_ontable(scene_state, a, graph):
                graph.add_predicate(ontable(a, src="extracted"))

        # clear(x): no object on x
        for a in obj_ids:
            has_top = any(
                p.name == "on" and p.args[1] == a for p in graph.predicates
            )
            if not has_top:
                graph.add_predicate(clear(a, src="extracted"))

        # holding / handempty
        if holding_object:
            graph.add_predicate(holding(holding_object, src="extracted"))
        else:
            graph.add_predicate(handempty(src="extracted"))

        # reachable / graspable
        for a in obj_ids:
            if self._is_reachable(scene_state, a):
                graph.add_predicate(reachable(a, src="extracted"))
            if graph.has("reachable", a) and graph.has("clear", a):
                graph.add_predicate(graspable(a, src="extracted"))

        return graph

    def _is_on(self, scene_state, top: str, bottom: str) -> bool:
        top_obj = scene_state.get_object(top)
        bottom_obj = scene_state.get_object(bottom)
        if not top_obj.available or not bottom_obj.available:
            return False
        if top_obj.position is None or bottom_obj.position is None:
            return False
        t, b = top_obj.position, bottom_obj.position
        xy_dist = np.sqrt((t[0] - b[0])**2 + (t[1] - b[1])**2)
        z_diff = t[2] - b[2]
        return xy_dist < self.on_xy_tolerance and self.cube_height - self.on_z_tolerance < z_diff < self.cube_height + self.on_z_tolerance

    def _is_ontable(self, scene_state, obj_id: str, graph: SymbolicSceneGraph) -> bool:
        obj = scene_state.get_object(obj_id)
        if not obj.available or obj.position is None:
            return False
        z = obj.position[2]
        if not (self.table_height - 0.02 < z < self.table_height + self.cube_height + 0.01):
            return False
        has_support = any(p.name == "on" and p.args[0] == obj_id for p in graph.predicates)
        return not has_support

    def _is_reachable(self, scene_state, obj_id: str) -> bool:
        obj = scene_state.get_object(obj_id)
        if not obj.available or obj.position is None:
            return False
        x, y = obj.position[0], obj.position[1]
        xmin, ymin, xmax, ymax = self.workspace_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax
