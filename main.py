from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.franka import Franka

import numpy as np

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

# 1. Joint space
# q = np.array([0.0, -0.5, 0.0, -2.0 , 0.0, 1.5, 0.7, 0.04, 0.04])

# setting the joints location
# franka.set_joint_positions(q)

# move the joints
# from omni.isaac.core.utils.types import ArticulationAction

# action = ArticulationAction(joint_positions=q)

# franka.apply_action(action)

# while simulation_app.is_running():
#     world.step(render=True)


# 2. Cartesian space
from omni.isaac.franka import KinematicsSolver # for isaac sim 4.0

# IK solver
ik_controller = KinematicsSolver(franka)


A = np.array([0.5, 0.0, 0.3])
B = np.array([0.5,  0.2, 0.3])
C = np.array([0.5, -0.2, 0.3])

tgts = [A, B, C]

# IK (inverse kinematics): cartesian space -> joint space
for target_position in tgts:
    actions, success = ik_controller.compute_inverse_kinematics(target_position=target_position)
    if success:
        print("IK success")
        franka.apply_action(actions)

        # give the sim time to move the rogot
        for _ in range(120):
            world.step(render=True)

    else:
        print("IK falied")

# Keep window open
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()