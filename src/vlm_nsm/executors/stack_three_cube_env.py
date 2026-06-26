"""3-Cube Stack environment for multi-step symbolic learning validation.

Patches robosuite Stack._load_model to include a third cube (blue, 2cm).
The patch is applied at import time via monkey-patching.
"""

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.objects import BoxObject
from robosuite.utils.placement_samplers import UniformRandomSampler

# Store original _load_model
_original_load_model = Stack._load_model


def _patched_load_model(self):
    """Patched _load_model: add cubeC before placement initializer processes cubes."""
    # Create cubeA, cubeB (from original), then add cubeC
    _original_load_model(self)

    # Add third cube (blue, 2cm)
    self.cubeC = BoxObject(
        name="cubeC",
        size_min=[0.04, 0.04, 0.04],
        size_max=[0.04, 0.04, 0.04],
        rgba=[0, 0, 1, 1],
    )

    # Add 3 Hanoi pegs (thin tall cylinders) as static objects
    # Peg positions on table: left, center, right
    peg_radius = 0.006
    peg_height = 0.02  # short pegs: visual markers only, avoid gripper collision
    self.peg1 = BoxObject(name='peg1',
        size_min=[peg_radius, peg_radius, peg_height/2],
        size_max=[peg_radius, peg_radius, peg_height/2],
        rgba=[0.5, 0.3, 0.1, 1],  # brown
        joints=None,  # static, no free joint
    )
    self.peg2 = BoxObject(name='peg2',
        size_min=[peg_radius, peg_radius, peg_height/2],
        size_max=[peg_radius, peg_radius, peg_height/2],
        rgba=[0.5, 0.3, 0.1, 1],
        joints=None,
    )
    self.peg3 = BoxObject(name='peg3',
        size_min=[peg_radius, peg_radius, peg_height/2],
        size_max=[peg_radius, peg_radius, peg_height/2],
        rgba=[0.5, 0.3, 0.1, 1],
        joints=None,
    )

    # Merge all new objects into the model
    new_objects = [self.cubeC, self.peg1, self.peg2, self.peg3]
    if hasattr(self, 'model') and hasattr(self.model, 'merge_objects'):
        self.model.merge_objects(new_objects)

    # Only add cubeC to placement initializer (pegs are static)
    self.placement_initializer.add_objects([self.cubeC])

    # Manually position pegs at fixed table locations
    self._peg_positions = {
        'peg1': (-0.12, -0.08),
        'peg2': (0.00, -0.08),
        'peg3': (0.12, -0.08),
    }


# Apply the patch (default: 3-cube stacking)
Stack._load_model = _patched_load_model


class StackThreeCube(Stack):
    """Stack environment with 3 cubes.

    The actual modification is done by the monkey-patch above.
    This class exists as a named entry point for robosuite registration.
    """

    @property
    def cubeC_body_id(self):
        """Body ID for cubeC."""
        return self.sim.model.body_name2id("cubeC_main")
