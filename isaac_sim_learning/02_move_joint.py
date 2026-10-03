from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.franka import Franka

# 1 unit = 1 m
world = World(stage_units_in_meters=1.0)

franka_url = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.0/Isaac/Robots/Franka/franka.usd"
)

franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        usd_path=franka_url,
    )
)

world.reset()

# move the joints
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

# radian
q = np.array([0.0, -0.5, 0.0, -2.0 , 0.0, 1.5, 0.7, 0.04, 0.04])
action = ArticulationAction(joint_positions=q)
franka.apply_action(action)

for _ in range(200):
    world.step(render=False)

print(f"Target: {q}")
print(f"Actual: {franka.get_joint_positions()}")

simulation_app.close()