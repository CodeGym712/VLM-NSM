"""Real Drawer with slide joint — drawer body on prismatic joint + attached handle.

Key: high joint damping (200) prevents drawer from sliding away during grasp approach.
The slide joint constrains motion to Y axis (±15cm range).
"""

from xml.etree import ElementTree as ET
from robosuite.environments.manipulation.stack import Stack

_original_load_model = Stack._load_model

_DRAWER_XML = """
<body name="drawer_housing" pos="0.15 0.0 0.808">
  <geom type="box" size="0.10 0.08 0.05" rgba="0.5 0.30 0.10 1"/>
  <body name="drawer_body" pos="0 0 0">
    <joint name="drawer_slide" type="slide" axis="0 1 0" range="0.0 0.15" damping="200" frictionloss="50"/>
    <geom type="box" size="0.08 0.06 0.04" rgba="0.7 0.50 0.30 1"/>
    <body name="drawer_handle" pos="0 0.05 0.02">
      <geom name="handle_geom" type="box" size="0.015 0.015 0.02" rgba="0.2 0.2 0.2 1"/>
    </body>
  </body>
</body>
"""


def _patched_load_model(self):
    _original_load_model(self)
    drawer_elem = ET.fromstring(_DRAWER_XML)
    self.model.worldbody.append(drawer_elem)


Stack._load_model = _patched_load_model
