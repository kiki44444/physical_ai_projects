from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka import Franka

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
    )
)

world.reset()

joint_positions = franka.get_joint_positions()
print(joint_positions)

import numpy as np

q = np.array([0.0, -0.5, 0.0, -2.0 , 0.0, 1.5, 0.7, 0.04, 0.04])

# setting the joints location
# franka.set_joint_positions(q)


# move the joints
from omni.isaac.core.utils.types import ArticulationAction

action = ArticulationAction(joint_positions=q)

franka.apply_action(action)

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
