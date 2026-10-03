from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

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

print("DOF names:")
print(franka.dof_names)

print("Joint positions:")
print(franka.get_joint_positions())

simulation_app.close()