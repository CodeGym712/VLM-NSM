"""Pure-geometry camera projection utilities.

No Robosuite dependency. All functions operate on numpy arrays and
standard camera models (pinhole + OpenGL-style conventions).

Coordinate conventions
----------------------
- World frame: right-hand, Z-up (robosuite default).
- Camera frame: +X right, +Y up, **looks along -Z** (OpenGL convention).
- Image frame: u right, v down, origin at top-left.
- K (intrinsic): 3x3 matrix where K[0,0]=fx, K[1,1]=fy, K[0,2]=cx, K[1,2]=cy.
- T_camera_world: 4x4 homogeneous transform **from camera to world**
  (i.e. ``p_world = T_camera_world @ p_cam_hom``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def intrinsic_from_fov(
    fov_y_deg: float,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    """Build a 3x3 intrinsic matrix from vertical FOV and image size.

    Assumes square pixels and principal point at image centre (MuJoCo default).

    Args:
        fov_y_deg: Vertical field of view in degrees.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        3x3 intrinsic matrix K.
    """
    fov_y_rad = np.deg2rad(fov_y_deg)
    fy = height / (2.0 * np.tan(fov_y_rad / 2.0))
    fx = fy  # square-pixel assumption
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def pixel_to_camera_ray(
    u: float,
    v: float,
    K: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unproject a pixel to a ray in the camera frame.

    Args:
        u: Horizontal pixel coordinate (0-based, left-to-right).
        v: Vertical pixel coordinate (0-based, top-to-bottom).
        K: 3x3 intrinsic matrix.

    Returns:
        (origin_cam, direction_cam):
            origin_cam — 3-vector, always zero (camera centre).
            direction_cam — unit 3-vector pointing from the camera through the
                pixel (camera-frame convention: +X right, +Y up, -Z forward).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Unproject to camera coordinates (z = -1 for look-direction convention).
    # Image v goes down, camera y goes up -> negate the y offset.
    dx = (u - cx) / fx
    dy = (cy - v) / fy
    dz = -1.0

    direction = np.array([dx, dy, dz], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, -1.0])

    return np.zeros(3, dtype=np.float64), direction / norm


def transform_ray_to_world(
    ray_origin_cam: NDArray[np.float64],
    ray_dir_cam: NDArray[np.float64],
    T_camera_world: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform a camera-frame ray into world frame.

    Args:
        ray_origin_cam: 3-vector origin in camera frame.
        ray_dir_cam: Unit 3-vector direction in camera frame.
        T_camera_world: 4x4 homogeneous camera-to-world transform.

    Returns:
        (origin_world, dir_world).
    """
    R = T_camera_world[:3, :3]
    t = T_camera_world[:3, 3]

    origin_world = R @ ray_origin_cam + t
    dir_world = R @ ray_dir_cam

    # Re-normalise to guard against near-singular R
    norm = np.linalg.norm(dir_world)
    if norm > 1e-12:
        dir_world = dir_world / norm

    return origin_world, dir_world


def ray_plane_intersection(
    ray_origin: NDArray[np.float64],
    ray_dir: NDArray[np.float64],
    plane_z: float,
    eps: float = 1e-12,
) -> NDArray[np.float64] | None:
    """Intersect a ray with the plane ``z = plane_z`` in world coordinates.

    Args:
        ray_origin: 3-vector ray origin (world frame).
        ray_dir: Unit 3-vector ray direction (world frame).
        plane_z: Z-coordinate of the horizontal plane.
        eps: Tolerance for detecting a parallel ray.

    Returns:
        3-vector intersection point, or **None** if the ray is parallel to
        the plane or points away from it.
    """
    dz = ray_dir[2]
    if abs(dz) < eps:
        return None  # parallel

    t = (plane_z - ray_origin[2]) / dz
    if t < 0.0:
        return None  # behind camera

    return ray_origin + t * ray_dir


def pixel_to_table_point(
    u: float,
    v: float,
    K: NDArray[np.float64],
    T_camera_world: NDArray[np.float64],
    table_z: float,
) -> NDArray[np.float64] | None:
    """Project a pixel onto the table plane (z = table_z) in world coordinates.

    This is a convenience wrapper combining the full ray-cast pipeline:

        pixel -> camera ray -> world ray -> plane intersection

    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        K: 3x3 intrinsic matrix.
        T_camera_world: 4x4 camera-to-world transform.
        table_z: Table-plane Z coordinate in world frame.

    Returns:
        3-vector (x, y, table_z) or **None** if the ray misses the plane.
    """
    origin_cam, dir_cam = pixel_to_camera_ray(u, v, K)
    origin_world, dir_world = transform_ray_to_world(origin_cam, dir_cam, T_camera_world)
    return ray_plane_intersection(origin_world, dir_world, table_z)


def world_to_camera_point(
    point_world: NDArray[np.float64],
    T_camera_world: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform a world-frame point to camera-frame coordinates.

    Uses ``R^T @ (p_world - t)`` (inverse of T_camera_world).

    Args:
        point_world: 3-vector in world coordinates.
        T_camera_world: 4x4 camera-to-world homogeneous transform.

    Returns:
        3-vector in camera coordinates (X right, Y up, -Z forward).
    """
    R = T_camera_world[:3, :3]
    t = T_camera_world[:3, 3]
    return R.T @ (point_world - t)


def camera_point_to_pixel(
    camera_point: NDArray[np.float64],
    K: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Project a camera-frame 3D point to pixel coordinates.

    The camera looks along -Z (OpenGL convention). Points at or behind the
    camera plane (z >= 0) return ``None``.

    Args:
        camera_point: 3-vector in camera coordinates.
        K: 3x3 intrinsic matrix.

    Returns:
        ``(u, v)`` pixel coordinates or ``None`` if the point is not in front
        of the camera.
    """
    x, y, z = camera_point[0], camera_point[1], camera_point[2]
    if z >= 0.0:
        return None  # behind or on the camera plane

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x / (-z) + cx
    v = cy - fy * y / (-z)
    return (float(u), float(v))


def project_world_point_to_pixel(
    point_world: NDArray[np.float64],
    K: NDArray[np.float64],
    T_camera_world: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Forward-project a world point onto the camera image plane.

    Convenience combining ``world_to_camera_point`` + ``camera_point_to_pixel``.

    Args:
        point_world: 3-vector in world coordinates.
        K: 3x3 intrinsic matrix.
        T_camera_world: 4x4 camera-to-world homogeneous transform.

    Returns:
        ``(u, v)`` pixel coordinates or ``None`` if the point is behind the
        camera.
    """
    p_cam = world_to_camera_point(point_world, T_camera_world)
    return camera_point_to_pixel(p_cam, K)


def triangulate_rays(
    origin1: NDArray[np.float64],
    dir1: NDArray[np.float64],
    origin2: NDArray[np.float64],
    dir2: NDArray[np.float64],
    eps: float = 1e-12,
) -> NDArray[np.float64] | None:
    """Triangulate the 3D midpoint of two world-frame rays.

    Finds the midpoint of the shortest segment connecting two skew lines
    (rays). This is the maximum-likelihood 3D point under the assumption
    that both rays pass near the same world point and the only errors are
    small angular deviations.

    Args:
        origin1: 3-vector, ray 1 origin in world frame.
        dir1: Unit 3-vector, ray 1 direction in world frame.
        origin2: 3-vector, ray 2 origin in world frame.
        dir2: Unit 3-vector, ray 2 direction in world frame.
        eps: Tolerance for detecting near-parallel rays.

    Returns:
        3-vector midpoint, or **None** if the rays are near-parallel or
        the closest-approach segment lies behind either camera.
    """
    w = origin1 - origin2
    b = np.dot(dir1, dir2)
    d = np.dot(dir1, w)
    e = np.dot(dir2, w)

    denom = 1.0 - b * b
    if abs(denom) < eps:
        return None  # near-parallel rays

    t1 = (b * e - d) / denom
    t2 = (e - b * d) / denom

    if t1 < 0.0 or t2 < 0.0:
        return None  # closest approach behind a camera

    p1 = origin1 + t1 * dir1
    p2 = origin2 + t2 * dir2
    return (p1 + p2) / 2.0
