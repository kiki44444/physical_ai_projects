from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"Headless": True})

from omni.isaac.core import World
from omni.isaac.franka import Franka

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

from omni.isaac.franka.controllers import RMPFlowController
import numpy as np

controller = RMPFlowController(name="franka_rmpflow_controller",
    robot_articulation=franka)

# target gripper coordinate (x, y, z)
target = np.array([0.4, 0.2, 0.5])

for _ in range(500):
    action = controller.forward(target_end_effector_position=target)
    franka.apply_action(action)
    world.step(render=False)

print(f"joint pos: {franka.get_joint_positions()}")
print(f"gripper pos: {franka.end_effector.get_world_pose()[0]}")

simulation_app.close()