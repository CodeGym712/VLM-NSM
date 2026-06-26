"""Robosuite camera-information utilities.

Provides adapters to extract intrinsic (K) and extrinsic (T_camera_world)
matrices from a running Robosuite environment.

Usage
-----
>>> from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
>>> env = RobosuiteEnvWrapper(...)
>>> info = get_camera_info(env, "agentview", width=512, height=512)
>>> info.K               # 3x3 intrinsic matrix
>>> info.T_camera_world  # 4x4 camera-to-world
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vlm_nsm.perception.camera_geometry import intrinsic_from_fov


@dataclass
class CameraInfo:
    """Camera calibration data."""

    K: NDArray[np.float64]  # 3x3 intrinsic matrix
    T_camera_world: NDArray[np.float64]  # 4x4 camera-to-world transform
    width: int
    height: int
    camera_name: str


def _reshape_cam_xmat(xmat_flat: NDArray[np.float64],
                      y_flip: bool = True) -> NDArray[np.float64]:
    """Reshape MuJoCo's 9-element row-major orientation to 3x3.

    MuJoCo ``data.cam_xmat`` stores a 3x3 rotation matrix as a flat
    9-element array in **row-major** order.

    If *y_flip* is True (the default), the camera Y axis (column 1) is
    negated to convert from MuJoCo's convention to the OpenGL convention
    expected by the projection code.
    """
    R = xmat_flat.reshape(3, 3).copy()
    if y_flip:
        R[:, 1] *= -1.0
    return R


def get_camera_intrinsics(
    env,
    camera_name: str = "agentview",
    width: int = 512,
    height: int = 512,
) -> NDArray[np.float64]:
    """Get the 3x3 intrinsic matrix K for a Robosuite camera.

    Reads the vertical FOV from MuJoCo and builds K assuming square pixels
    and principal point at image centre.

    Args:
        env: A ``RobosuiteEnvWrapper`` (or any object with ``.env.sim``).
        camera_name: Name of the camera in the MuJoCo model.
        width: Image width in pixels (must match rendering resolution).
        height: Image height in pixels.

    Returns:
        3x3 intrinsic matrix K.
    """
    sim = env.env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    fov_y_deg = float(sim.model.cam_fovy[cam_id])
    return intrinsic_from_fov(fov_y_deg, width, height)


def get_camera_extrinsics(
    env,
    camera_name: str = "agentview",
    y_flip: bool = True,
) -> NDArray[np.float64]:
    """Get the 4x4 camera-to-world transform.

    Builds ``T_camera_world`` from MuJoCo's ``cam_xpos`` (position) and
    ``cam_xmat`` (orientation) arrays.

    Args:
        env: A ``RobosuiteEnvWrapper``.
        camera_name: Camera name in the MuJoCo model.
        y_flip: If True, negate camera Y axis to convert from MuJoCo to
            OpenGL convention.

    Returns:
        4x4 homogeneous camera-to-world matrix.
    """
    sim = env.env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    pos = sim.data.cam_xpos[cam_id].copy()  # (3,)
    R = _reshape_cam_xmat(sim.data.cam_xmat[cam_id].copy(), y_flip=y_flip)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def get_table_height(env, default: float | None = 0.0) -> float | None:
    """Extract the table-surface Z coordinate from the Robosuite simulation.

    Searches for a body named ``"table"`` or similar and returns its Z position.

    Args:
        env: A ``RobosuiteEnvWrapper``.
        default: Value returned if no table body is found.

    Returns:
        Z coordinate of the table surface, or *default*.
    """
    sim = env.env.sim
    for i in range(sim.model.nbody):
        name = str(sim.model.body_names[i])
        if "table" in name.lower():
            z = float(sim.data.body_xpos[i][2])
            # Table surface is roughly the body centre Z plus half the table
            # thickness. For Robosuite's default table this is usually
            # sufficient as-is.
            return z
    return default


def get_camera_info(
    env,
    camera_name: str = "agentview",
    width: int = 512,
    height: int = 512,
    y_flip: bool = True,
) -> CameraInfo:
    """Convenience: return both intrinsic and extrinsic camera data.

    Args:
        env: A ``RobosuiteEnvWrapper``.
        camera_name: Camera name in the MuJoCo model.
        width: Image width.
        height: Image height.
        y_flip: If True, negate camera Y axis in extrinsics.

    Returns:
        A ``CameraInfo`` dataclass.
    """
    return CameraInfo(
        K=get_camera_intrinsics(env, camera_name, width, height),
        T_camera_world=get_camera_extrinsics(env, camera_name, y_flip=y_flip),
        width=width,
        height=height,
        camera_name=camera_name,
    )
