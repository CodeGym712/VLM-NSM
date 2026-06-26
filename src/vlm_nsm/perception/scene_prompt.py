"""Build scene-aware prompts from scene info for parser injection."""

from __future__ import annotations

from vlm_nsm.perception.data_types import SceneInfo


def _format_spatial(obj_label: str, attributes: dict) -> str:
    """Append spatial info string from detector attributes if available."""
    parts: list[str] = []
    spatial = attributes.get("spatial")
    if spatial:
        cn = spatial.get("center_norm")
        if cn:
            parts.append(f"image center=({cn[0]:.2f}, {cn[1]:.2f})")
        ipos = spatial.get("image_position")
        if ipos:
            parts.append(f"image-{ipos}")

        # Image-space relations
        relations = spatial.get("relations", {})
        for other_label, rels in sorted(relations.items()):
            for rel in rels:
                parts.append(f"{rel} {other_label}")

        # Image-space ambiguous — suppress when world-space data covers the pair
        ambiguous = spatial.get("ambiguous_relations", [])
        world_rels = spatial.get("world_relations", {})
        ambiguous_x = spatial.get("ambiguous_x", [])
        ambiguous_y = spatial.get("ambiguous_y", [])
        world_known_labels = set(world_rels.keys()) | set(ambiguous_x) | set(ambiguous_y)

        for other_label in ambiguous:
            if other_label not in world_known_labels:
                parts.append(f"x-ambiguous-with {other_label}")

        # World-space relations (from estimated 3D poses)
        world_relations = spatial.get("world_relations", {})
        for other_label, rels in sorted(world_relations.items()):
            for rel in rels:
                parts.append(f"{rel} {other_label}")

        ambiguous_x = spatial.get("ambiguous_x", [])
        for other_label in ambiguous_x:
            parts.append(f"x-ambiguous-with {other_label}")
        ambiguous_y = spatial.get("ambiguous_y", [])
        for other_label in ambiguous_y:
            parts.append(f"y-ambiguous-with {other_label}")
    else:
        cn = attributes.get("center_norm")
        if cn and "x" in cn and "y" in cn:
            parts.append(f"image center=({cn['x']:.2f}, {cn['y']:.2f})")

    return ", ".join(parts) if parts else ""


def _has_spatial_info(attributes: dict) -> bool:
    """Check if object has spatial info (from RGB detector)."""
    return bool(attributes.get("spatial")) or bool(attributes.get("center_norm"))


def _compute_robot_pairwise_relations(scene: SceneInfo) -> dict[str, list[str]]:
    """Compute pairwise robot-relative near/far relations.

    Returns dict mapping object label to list of relation strings like
    "nearer_to_robot_than cube2".
    """
    objects_with_dist = [
        (obj.label, obj.attributes.get("robot_distance"))
        for obj in scene.objects
        if obj.attributes.get("robot_distance") is not None
    ]
    if len(objects_with_dist) < 2:
        return {}

    relations: dict[str, list[str]] = {}
    for i, (label_i, dist_i) in enumerate(objects_with_dist):
        for j, (label_j, dist_j) in enumerate(objects_with_dist):
            if i >= j:
                continue
            if abs(dist_i - dist_j) < 0.01:
                continue
            if dist_i < dist_j:
                relations.setdefault(label_i, []).append(f"nearer_to_robot_than {label_j}")
                relations.setdefault(label_j, []).append(f"farther_from_robot_than {label_i}")
            else:
                relations.setdefault(label_j, []).append(f"nearer_to_robot_than {label_i}")
                relations.setdefault(label_i, []).append(f"farther_from_robot_than {label_j}")

    return relations


def _append_table_layout(lines: list[str], scene: SceneInfo) -> None:
    """Append a table-layout summary grouping objects by region.

    Only emits when at least one object has robot_distance set
    (i.e. predicate inventory is active).
    """
    has_predicates = any(
        obj.attributes.get("robot_distance") is not None
        for obj in scene.objects
    )
    if not has_predicates:
        return

    # Group objects by region
    front_left: list[str] = []
    front_center: list[str] = []
    front_right: list[str] = []
    center_left: list[str] = []
    center: list[str] = []
    center_right: list[str] = []
    back_left: list[str] = []
    back_center: list[str] = []
    back_right: list[str] = []

    for obj in scene.objects:
        near = obj.attributes.get("near_edges", [])
        label = obj.label
        has_left = "left_edge" in near
        has_right = "right_edge" in near
        has_front = "front_edge" in near
        has_back = "back_edge" in near

        # Determine row (Y)
        if has_front:
            row = "front"
        elif has_back:
            row = "back"
        else:
            row = "center"

        # Determine col (X)
        if has_left:
            col = "left"
        elif has_right:
            col = "right"
        else:
            col = "center"

        key = f"{row}_{col}"
        {
            "front_left": front_left,
            "front_center": front_center,
            "front_right": front_right,
            "center_left": center_left,
            "center_center": center,
            "center_right": center_right,
            "back_left": back_left,
            "back_center": back_center,
            "back_right": back_right,
        }[key].append(label)

    lines.append("Table layout:")
    region_labels = [
        ("Front-left", front_left),
        ("Front-center", front_center),
        ("Front-right", front_right),
        ("Center-left", center_left),
        ("Center", center),
        ("Center-right", center_right),
        ("Back-left", back_left),
        ("Back-center", back_center),
        ("Back-right", back_right),
    ]
    for region_name, obj_labels in region_labels:
        if obj_labels:
            lines.append(f"  {region_name}: {', '.join(obj_labels)}")

    # Robot-relative ordering (near to far)
    objs_with_dist = sorted(
        ((obj.label, obj.attributes.get("robot_distance"))
         for obj in scene.objects
         if obj.attributes.get("robot_distance") is not None),
        key=lambda x: x[1],
    )
    if objs_with_dist:
        ordering = " < ".join(f"{label} ({dist:.2f}m)" for label, dist in objs_with_dist)
        lines.append(f"  Robot-relative (near to far): {ordering}")


def _append_shared_predicate_ambiguity(lines: list[str], scene: SceneInfo) -> None:
    """Append explicit shared-predicate ambiguity notes.

    When multiple objects share the same predicate (on_table, same region, no edge
    proximity), render an explicit note that referential expressions relying on
    that predicate are ambiguous and MUST be rejected.

    Only emits when predicate inventory is active (robot_distance present).
    """
    has_predicates = any(
        obj.attributes.get("robot_distance") is not None
        for obj in scene.objects
    )
    if not has_predicates:
        return

    objects = scene.objects
    if len(objects) < 2:
        return

    shared_notes: list[str] = []

    # Check on_table sharing
    on_table_objs = [obj.label for obj in objects if obj.attributes.get("on_table") is True]
    if len(on_table_objs) >= 2:
        shared_notes.append(
            f"on_table is shared by {', '.join(on_table_objs)}. "
            "Phrases like 'the cube on the table' or 'the block that is on the table' "
            "do NOT uniquely identify an object and MUST be ungroundable "
            "unless another distinguishing property is given."
        )

    # Check table edge ambiguity: if no object is near any edge,
    # then "closer to the edge" is ambiguous
    any_edge = any(
        obj.attributes.get("near_edges")
        for obj in objects
    )
    if not any_edge:
        shared_notes.append(
            "No object is near any table edge. "
            "Phrases like 'closer to the edge' or 'near the edge' "
            "do NOT uniquely identify an object and MUST be ungroundable."
        )

    # Check same region sharing
    regions: dict[str, list[str]] = {}
    for obj in objects:
        near = obj.attributes.get("near_edges", [])
        has_front = "front_edge" in near
        has_back = "back_edge" in near
        has_left = "left_edge" in near
        has_right = "right_edge" in near
        row = "front" if has_front else ("back" if has_back else "center")
        col = "left" if has_left else ("right" if has_right else "center")
        region = f"{row}-{col}"
        regions.setdefault(region, []).append(obj.label)

    for region, labels in regions.items():
        if len(labels) >= 2 and region != "center-center":
            shared_notes.append(
                f"Table region '{region}' is shared by {', '.join(labels)}. "
                "Spatial references within this region may be ambiguous."
            )

    if shared_notes:
        lines.append("Shared predicate ambiguity:")
        for note in shared_notes:
            lines.append(f"  - {note}")


def build_scene_description(scene: SceneInfo) -> str:
    """Build a human-readable scene description for prompt injection.

    Includes spatial relations (image-left/image-right) when available
    from RGB detector perception.
    """
    lines = ["Current scene:"]
    robot_pairwise = _compute_robot_pairwise_relations(scene)
    for obj in scene.objects:
        desc = f"  {obj.label} ({obj.category})"
        if obj.color:
            desc += f", color: {obj.color}"
        if obj.pose:
            pos_source = obj.attributes.get("position_source", "")
            if pos_source in ("rgb_bbox_table_plane_projection", "rgb_dual_camera_triangulation"):
                desc += (
                    f" estimated position ({obj.pose['x']:.3f}, {obj.pose['y']:.3f}, "
                    f"{obj.pose['z']:.3f})"
                )
                confidence = obj.attributes.get("pose_confidence")
                if confidence is not None:
                    desc += f", confidence={confidence:.2f}"
            else:
                desc += (
                    f" at ({obj.pose['x']:.2f}, {obj.pose['y']:.2f}, "
                    f"{obj.pose['z']:.2f})"
                )
        spatial_str = _format_spatial(obj.label, obj.attributes)
        if spatial_str:
            desc += f", {spatial_str}"
        has_pose_conf = "pose_confidence" in obj.attributes
        if not has_pose_conf and obj.confidence < 1.0:
            desc += f", confidence={obj.confidence:.2f}"

        # Predicate-level enrichments
        on_table = obj.attributes.get("on_table")
        if on_table is True:
            desc += ", on table"
        elif on_table is False:
            desc += ", stacked/elevated"

        near_edges = obj.attributes.get("near_edges")
        if near_edges:
            desc += f", near {', '.join(near_edges)}"

        if obj.attributes.get("pose_suspicious"):
            desc += ", WARNING: estimated position may be inaccurate"

        robot_dist = obj.attributes.get("robot_distance")
        if robot_dist is not None:
            desc += f", {robot_dist:.2f}m from robot"

        pairwise_rels = robot_pairwise.get(obj.label, [])
        for rel in pairwise_rels:
            desc += f", {rel}"

        lines.append(desc)

    if scene.on_relations:
        lines.append("Stacking relations:")
        for upper, lower in scene.on_relations:
            lines.append(f"  {upper} is on {lower}")

    # Table layout summary (when predicate inventory is active)
    _append_table_layout(lines, scene)

    # Shared predicate ambiguity (when predicate inventory is active)
    _append_shared_predicate_ambiguity(lines, scene)

    if scene.grasped_object:
        lines.append(f"Grasping: {scene.grasped_object}")
    else:
        lines.append("Gripper is empty")

    # Add guidance note based on position source
    if scene.objects:
        has_unknown_3d = any(
            o.attributes.get("position_source") == "unknown_rgb_detector_3d"
            for o in scene.objects
        )
        has_table_projection = any(
            o.attributes.get("position_source") == "rgb_bbox_table_plane_projection"
            for o in scene.objects
        )
        has_stereo = any(
            o.attributes.get("position_source") == "rgb_dual_camera_triangulation"
            for o in scene.objects
        )

        if has_unknown_3d:
            lines.append("")
            lines.append(
                "Note: 3D positions are unknown from RGB detector. "
                "Use image-left/image-right relations from bbox centers "
                "to resolve left/right references under the fixed camera. "
                "If spatial relations are ambiguous or unavailable, "
                'output status="ungroundable" rather than guessing.'
            )

        if has_stereo and has_table_projection:
            lines.append("")
            lines.append(
                "Note: 3D positions from dual-camera stereo triangulation "
                "(calibrated) where both cameras detect the object, with "
                "table-plane projection fallback (approximate) for objects "
                "only detected in one camera. "
                "World relations (left_of/right_of/front_of/behind/near/far) "
                "should be preferred over image-space relations. "
                "If spatial relations are ambiguous (marked x-ambiguous-with or "
                "y-ambiguous-with), output status=\"ungroundable\" "
                "rather than guessing."
            )
        elif has_stereo:
            lines.append("")
            lines.append(
                "Note: 3D positions are estimated from dual-camera stereo "
                "triangulation and are calibrated. Use world left/right/front/behind "
                "relations for spatial references. "
                "World relations (left_of/right_of/front_of/behind/near/far) "
                "should be preferred over image-space relations. "
                "If spatial relations are ambiguous (marked x-ambiguous-with or "
                "y-ambiguous-with), output status=\"ungroundable\" "
                "rather than guessing."
            )
        elif has_table_projection:
            lines.append("")
            lines.append(
                "Note: 3D positions are estimated from camera-to-table projection "
                "and are approximate. Use estimated world left/right/front/behind "
                "relations for spatial references. "
                "If spatial relations are ambiguous (marked x-ambiguous-with or "
                "y-ambiguous-with), output status=\"ungroundable\" "
                "rather than guessing."
            )

        # Predicate inventory guidance
        has_predicates = any(
            o.attributes.get("on_table") is not None
            for o in scene.objects
        )
        if has_predicates:
            lines.append("")
            lines.append(
                "Color reference rules:\n"
                "- Treat 'reddish' as red and 'greenish' as green when "
                "red/green objects are present. Common color variants "
                "(e.g. 'dark red', 'light green') should be matched to "
                "the closest available color.\n"
                "- Do NOT infer brightness attributes such as "
                "lighter/darker/bright/dull unless brightness relations "
                "are explicitly listed in the scene inventory. If "
                "brightness relations are not provided, brightness-based "
                "references MUST be ungroundable."
            )
            lines.append(
                "Note: Each object is annotated with on-table status "
                "(on table vs stacked/elevated), table edge proximity "
                "(near front_edge/back_edge/left_edge/right_edge), "
                "robot-relative distance, and pairwise robot-relative "
                "relations (nearer_to_robot_than / farther_from_robot_than). "
                "A table layout summary groups objects by region with a "
                "robot-relative ordering. "
                "A shared predicate ambiguity section explicitly lists "
                "which predicates are shared by multiple objects and "
                "MUST be treated as ambiguous. "
                "Use these predicates to resolve referential expressions "
                "(e.g. 'the cube on the left', 'the cube closest to you', "
                "'the front cube'). "
                "IMPORTANT: If a referential expression uses a property "
                "shared equally by multiple objects (see shared predicate "
                "ambiguity section above), the reference is ambiguous and you "
                "MUST output status=\"ungroundable\". "
                "If a referential expression requires a convention not provided "
                "(e.g. 'first you see', 'lighter/darker' without brightness data), "
                "output status=\"ungroundable\". "
                "If a position is marked WARNING/suspicious, prefer other "
                "spatial cues for that object."
            )

    return "\n".join(lines)


def inject_scene_into_prompt(prompt_template: str, scene: SceneInfo) -> str:
    """Inject a scene description into a parser prompt template.

    Replaces ``{scene_description}`` if present, otherwise appends.
    """
    scene_desc = build_scene_description(scene)
    if "{scene_description}" in prompt_template:
        return prompt_template.replace("{scene_description}", scene_desc)
    return prompt_template + "\n\n" + scene_desc
