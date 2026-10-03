from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"Headless": True})

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
from omni.isaac.franka.controllers import RMPFlowController

# world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()


# franka
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

# cube (center)
cube_pos = np.array([0.5, 0.0, 0.025])

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=cube_pos,
        scale=np.array([0.05, 0.05, 0.05]),
        mass=0.05,
    )
)

world.reset()

# cartesian
controller = RMPFlowController(
    name = "franka_rmpflow_controller",
    robot_articulation=franka
)

# functions
def move_to(target, steps=300):
    for _ in range(steps):
        action = controller.forward(target_end_effector_position=target)
        franka.apply_action(action)
        world.step(render=False)

    actual_pos = franka.end_effector.get_world_pose()[0]
    print(f"Target: {target}")
    print(f"Actual: {actual_pos}")

def close_gripper(steps=100):
    franka.gripper.close()
    for _ in range(steps):
        world.step(render=False)


def open_gripper(steps=100):
    franka.gripper.open()
    for _ in range(steps):
        world.step(render=False)

# pick and place positions
target = np.array([0.4, 0.3, 0.025])

pre_grap = cube_pos + np.array([0.0, 0.0, 0.2])
grasp = cube_pos + np.array([0.0, 0.0, 0.05])
lift = cube_pos + np.array([0.0, 0.0, 0.3])
pre_place = target + np.array([0.0, 0.0, 0.30])
place = target + np.array([0.0, 0.0, 0.05])

# state sequence
print("pre grap")
move_to(pre_grap)

print("descend")
move_to(grasp)

print("close gripper")
close_gripper()

print("lift")
move_to(lift)

print(f"Cube position after lift {cube.get_world_pose()[0]}")

print(" move to target")
move_to(pre_place)

print("lower")
move_to(place)

print("open gripper")
open_gripper()

print("retreat")
move_to(pre_place)

print(f"final cube position: {cube.get_world_pose()[0]}")

simulation_app.close()