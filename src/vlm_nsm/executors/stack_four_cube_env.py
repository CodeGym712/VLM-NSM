"""4-Cube Stack environment — adds cubeC + cubeD to robosuite Stack.

cubeA (4cm) and cubeB (5cm) are kept as-is from the original env.
cubeC (blue, 3cm) and cubeD (yellow, 3cm) are added.
cubeB at 5cm exceeds Panda gripper max opening (~4cm) — handled by
not placing cubeB as a grasp target in the tower config (cubeB stays at base).
"""

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.objects import BoxObject

_original_load_model = Stack._load_model


def _patched_load_model(self):
    _original_load_model(self)

    # Add cubeC (blue, 3cm)
    self.cubeC = BoxObject(
        name="cubeC",
        size_min=[0.03, 0.03, 0.03],
        size_max=[0.03, 0.03, 0.03],
        rgba=[0, 0, 1, 1],
    )

    # Add cubeD (yellow, 3cm)
    self.cubeD = BoxObject(
        name="cubeD",
        size_min=[0.03, 0.03, 0.03],
        size_max=[0.03, 0.03, 0.03],
        rgba=[1, 1, 0, 1],
    )

    new_objects = [self.cubeC, self.cubeD]
    if hasattr(self, 'model') and hasattr(self.model, 'merge_objects'):
        self.model.merge_objects(new_objects)

    self.placement_initializer.add_objects([self.cubeC, self.cubeD])


Stack._load_model = _patched_load_model
