# VLM-NSM perception package — scene understanding for parser grounding.

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo, DEFAULT_COLOR_MAP
from vlm_nsm.perception.oracle_perception import perceive_scene
from vlm_nsm.perception.scene_prompt import build_scene_description, inject_scene_into_prompt
from vlm_nsm.perception.spatial_relations import compute_image_spatial_relations, compute_world_spatial_relations
from vlm_nsm.perception.camera_geometry import (
    intrinsic_from_fov,
    pixel_to_camera_ray,
    transform_ray_to_world,
    ray_plane_intersection,
    pixel_to_table_point,
    triangulate_rays,
    world_to_camera_point,
    camera_point_to_pixel,
    project_world_point_to_pixel,
)
from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene
from vlm_nsm.perception.scene_context import SceneReferenceFrame, load_reference_frame
from vlm_nsm.perception.scene_predicates import compute_scene_predicates

__all__ = [
    "DetectedObject",
    "SceneInfo",
    "DEFAULT_COLOR_MAP",
    "perceive_scene",
    "build_scene_description",
    "inject_scene_into_prompt",
    "compute_image_spatial_relations",
    "compute_world_spatial_relations",
    "intrinsic_from_fov",
    "pixel_to_camera_ray",
    "transform_ray_to_world",
    "ray_plane_intersection",
    "pixel_to_table_point",
    "triangulate_rays",
    "world_to_camera_point",
    "camera_point_to_pixel",
    "project_world_point_to_pixel",
    "triangulate_scene",
    "SceneReferenceFrame",
    "load_reference_frame",
    "compute_scene_predicates",
]
