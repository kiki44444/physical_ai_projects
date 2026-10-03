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

from omni.isaac.franka import KinematicsSolver
import numpy as np

ik_solver = KinematicsSolver(franka)

target = np.array([0.5, 0.4, 0.1])

action, success = ik_solver.compute_inverse_kinematics(target_position=target)

print(f"IK success : {success}")
print(f"IK action: {action}")

if success:
    franka.apply_action(action)
    for _ in range(500):
        world.step(render=False)

    actual_pos, actual_ori = (
        franka.end_effector.get_world_pose()
    )

    print(f"Target:", target)
    print(f"Actual:", actual_pos)

simulation_app.close()