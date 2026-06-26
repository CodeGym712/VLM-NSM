"""RobosuiteEnvWrapper — wraps robosuite Stack environment (v1.5.2).

Provides cube pose extraction, EEF pose, gripper state, and step/reset.

In robosuite 1.5.2 the API changed:
  - Composite controller config (BASIC) replaces OSC_POSE config
  - Actions are normalized to [-1, 1]; scaling to physical deltas is internal
  - eef_site_id is a dict {'right': <int>}
  - robot.gripper is a dict of gripper objects
  - Body names are native strings (no .decode() needed)
  - Stack env has cubeA, cubeB (2 cubes) with body names cubeA_main, cubeB_main

If robosuite is not installed, import raises a clear error.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from vlm_nsm.executors.env_wrappers import BaseEnvWrapper

# Default cube name mapping for Stack env (robosuite 1.5.2)
CUBE_NAME_MAP = {
    "cube1": "cubeA_main",
    "cube2": "cubeB_main",
    "cube3": "cubeC_main",
    "cube4": "cubeD_main",
    "handle": "handle_main",
}


class RobosuiteEnvWrapper(BaseEnvWrapper):
    """Wrapper around robosuite Stack environment.

    Args:
        env_name: Robosuite environment name ("Stack").
        robots: Robot type ("Panda").
        cube_names: List of canonical cube names (e.g. ["cube1", "cube2"]).
        render_mode: "none", "human", or "offscreen".
        control_freq: Control frequency for the environment.
    """

    def __init__(
        self,
        env_name: str = "Stack",
        robots: str = "Panda",
        cube_names: list[str] | None = None,
        render_mode: str = "none",
        control_freq: int = 20,
    ):
        try:
            import robosuite
            from robosuite.controllers import load_composite_controller_config
        except ImportError as exc:
            raise ImportError(
                "Robosuite is not installed. "
                "Install with: pip install robosuite\n"
                "Use MockEnvWrapper for tests without Robosuite."
            ) from exc

        self._robosuite = robosuite
        # Suppress noisy controller-config warnings for single-arm robots
        import logging
        _rl = logging.getLogger("robosuite_logs")
        if _rl.handlers:
            _rl.handlers[0].setLevel(logging.ERROR)

        self._env_name = env_name
        self._robots = robots
        self.cube_names = cube_names or ["cube1", "cube2"]
        self.render_mode = render_mode
        self.control_freq = control_freq

        # Build body name mapping from canonical names
        self._cube_to_body: dict[str, str] = {}
        self._cube_bodies: list[str] = []
        self._body_id_cache: dict[str, int] = {}

        controller = load_composite_controller_config(controller="BASIC")
        has_renderer = render_mode == "human"
        has_offscreen_renderer = render_mode == "offscreen"

        self.env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=controller,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=False,
            control_freq=control_freq,
        )

        # Phase 2.0: Hanoi peg flag
        self._has_pegs = False

        # Discover body names after env creation
        self._discover_body_names()

        # Cache EEF site ID (dict in 1.5.2: {'right': <int>})
        self._eef_site_id: int | None = None
        if hasattr(self.env, "robots") and self.env.robots:
            raw_id = self.env.robots[0].eef_site_id
            if isinstance(raw_id, dict):
                self._eef_site_id = raw_id.get("right", list(raw_id.values())[0])
            else:
                self._eef_site_id = int(raw_id)

    def _setup_hanoi_pegs(self):
        """Auto-position Hanoi pegs if present (StackThreeCube env)."""
        try:
            model = self.env.sim.model
            table_z = 0.80
            positions = [(-0.12,-0.12), (0.0,-0.12), (0.12,-0.12)]
            for i, (x, y) in enumerate(positions):
                name = f'peg{i+1}_main'
                try:
                    bid = model.body_name2id(name)
                    import numpy as np
                    model.body_pos[bid] = np.array([x, y, table_z + 0.05])
                    self._has_pegs = True
                except ValueError:
                    break
            if self._has_pegs:
                self.env.sim.forward()
        except Exception:
            pass

    def _discover_body_names(self):
        """Map canonical cube names to robosuite body names by inspecting sim state."""
        sim = self.env.sim
        body_names = [str(sim.model.body_names[i]) for i in range(sim.model.nbody)]

        # Try to get object names from env
        known_objects = []
        for attr_name in ("cubes", "cubeA", "objects"):
            attr = getattr(self.env, attr_name, None)
            if attr is not None:
                if isinstance(attr, (list, tuple)):
                    for c in attr:
                        if hasattr(c, "root_body"):
                            known_objects.append(c.root_body)
                        elif hasattr(c, "name"):
                            known_objects.append(c.name)
                elif hasattr(attr, "root_body"):
                    known_objects.append(attr.root_body)
                    # Check for cubeB as well
                    if hasattr(self.env, "cubeB") and hasattr(self.env.cubeB, "root_body"):
                        known_objects.append(self.env.cubeB.root_body)

        # Build mapping: cube1 -> first known object, cube2 -> second, etc.
        for i, canonical in enumerate(self.cube_names):
            if i < len(known_objects):
                self._cube_to_body[canonical] = known_objects[i]
            else:
                # Fallback: use CUBE_NAME_MAP
                body_name = CUBE_NAME_MAP.get(canonical, canonical)
                matches = [b for b in body_names if b.lower().replace(" ", "") == body_name.lower().replace(" ", "")]
                if not matches:
                    matches = [b for b in body_names if canonical.lower().replace(" ", "") in b.lower().replace(" ", "")]
                self._cube_to_body[canonical] = matches[0] if matches else body_name

        self._cube_bodies = list(self._cube_to_body.values())

        # Cache body IDs
        for body_name in self._cube_bodies:
            try:
                self._body_id_cache[body_name] = sim.model.body_name2id(body_name)
            except ValueError:
                pass

    def get_available_bodies(self) -> list[str]:
        """Return all body names in the simulation (for debugging)."""
        sim = self.env.sim
        return [str(sim.model.body_names[i]) for i in range(sim.model.nbody)]

    def get_object_pose(self, name: str) -> dict[str, float]:
        """Get the 3D position of a named object.

        Args:
            name: Canonical cube name (e.g. "cube1") or "table".

        Returns:
            Dict with keys "x", "y", "z".
        """
        if name == "table":
            table_id = self.env.sim.model.body_name2id("table")
            pos = self.env.sim.data.body_xpos[table_id]
            return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}

        if name.startswith("cube"):
            body_name = self._cube_to_body.get(name)
        else:
            # Try name+_main first (BoxObject), then name as-is (XML-defined)
            body_name = name + "_main"
            try:
                self.env.sim.model.body_name2id(body_name)
            except ValueError:
                body_name = name  # fallback: XML-defined bodies use exact name

        if body_name is None:
            raise ValueError(
                f"Unknown object '{name}'. Known cube objects: "
                f"{list(self._cube_to_body.keys())}")

        body_id = self._body_id_cache.get(body_name)
        if body_id is None:
            try:
                body_id = self.env.sim.model.body_name2id(body_name)
                self._body_id_cache[body_name] = body_id
            except ValueError as exc:
                available_bodies = self.get_available_bodies()
                raise ValueError(
                    f"Body '{body_name}' (mapped from '{name}') not found. "
                    f"Available bodies: {available_bodies}"
                ) from exc

        pos = self.env.sim.data.body_xpos[body_id]
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}

    def get_object_body_name(self, name: str) -> str | None:
        """Return the robosuite body name for a canonical object name."""
        if name == "table":
            return "table"
        return self._cube_to_body.get(name)

    def get_target_pose(self, name: str) -> dict[str, float]:
        """Get target placement pose.

        For table, returns table pose. For cubes, returns the cube's current pose.
        """
        return self.get_object_pose(name)

    def get_all_object_poses(self) -> dict[str, dict[str, float]]:
        """Get poses of all tracked objects."""
        poses = {}
        for canonical in self.cube_names:
            try:
                poses[canonical] = self.get_object_pose(canonical)
            except ValueError:
                poses[canonical] = {"x": 0.0, "y": 0.0, "z": 0.0}
        poses["table"] = self.get_object_pose("table")
        return poses

    def get_eef_pose(self) -> dict[str, float]:
        """Get end-effector position.

        Uses robosuite site_xpos for accurate tracking.
        """
        if self._eef_site_id is None:
            raise ValueError("EEF site ID not available. Check robot setup.")
        pos = self.env.sim.data.site_xpos[self._eef_site_id]
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}

    def get_gripper_state(self) -> dict[str, Any]:
        """Get gripper state.

        Returns:
            Dict with "pos" (EEF position), "open" (average finger separation),
            "has_object" (heuristic: closed gripper = likely grasping).
        """
        eef_pos = self.get_eef_pose()

        # Gripper openness from finger joint positions
        open_val = 0.0
        if hasattr(self.env, "robots") and self.env.robots:
            robot = self.env.robots[0]
            try:
                jpos = robot.get_gripper_joint_positions()
                # Average absolute finger position — larger = more open
                open_val = float(np.mean(np.abs(jpos)))
            except (TypeError, ValueError, AttributeError):
                open_val = 0.0

        return {
            "pos": eef_pos,
            "open": open_val,
            "has_object": 0.005 < open_val < 0.035,  # partially closed = object between fingers
        }

    def get_objects_on(self, target_name: str) -> list[str]:
        """Get objects supported by target_name.

        Uses PhysicalPredicateExtractor logic to check support relationships.
        """
        object_poses = self.get_all_object_poses()
        targets = {}
        for cn in self.cube_names:
            pose = self.get_object_pose(cn)
            targets[cn] = pose
        targets["table"] = self.get_object_pose("table")

        from vlm_nsm.core.physical_predicate_extractor import extract_on_predicates

        thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
        on_preds = extract_on_predicates(object_poses, targets, thresholds)
        return [pred[1] for pred in on_preds if len(pred) == 3 and pred[2] == target_name]

    def render_rgb(
        self, camera: str = "agentview", height: int = 512, width: int = 512
    ) -> np.ndarray | None:
        """Render an RGB image from a simulation camera.

        Requires ``has_offscreen_renderer=True`` (render_mode="offscreen").

        Args:
            camera: Camera name in the sim (default: "agentview").
            height: Output image height.
            width: Output image width.

        Returns:
            RGB array ``(H, W, 3)`` or ``None`` on failure.
        """
        try:
            return self.env.sim.render(
                camera_name=camera, height=height, width=width
            )
        except Exception:
            return None

    def reset(self) -> dict[str, Any]:
        """Reset the environment.

        Returns:
            Observation dict.
        """
        obs = self.env.reset()
        return obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict]:
        """Execute an action.

        Args:
            action: Action vector (7-dim normalized [-1, 1] for BASIC controller).

        Returns:
            (obs, reward, done, info) tuple.
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
