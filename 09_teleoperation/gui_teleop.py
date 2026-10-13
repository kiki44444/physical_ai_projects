import os
import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
franka_url = ("http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
              "Assets/Isaac/4.0/Isaac/Robots/Franka/franka.usd")
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka", usd_path=franka_url))
cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube",
                                    position=np.array([0.50, 0.00, 0.025]),
                                    scale=np.array([0.05, 0.05, 0.05]), mass=0.05))
world.reset()
set_camera_view(eye=np.array([1.5, 1.5, 1.2]), target=np.array([0.45, 0.15, 0.1]),
                camera_prim_path="/OmniverseKit_Persp")
controller = RMPFlowController(name="teleop_rmpflow", robot_articulation=franka)
orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))
# The target is the RMPFlow end-effector target, not the fingertip center.
target = np.array([0.50, 0.00, 0.20], dtype=np.float64)
limits_min = np.array([0.25, -0.35, 0.035])
limits_max = np.array([0.75, 0.45, 0.50])

rclpy.init()
node = rclpy.create_node("franka_gui_teleop")

def on_delta(msg):
    global target
    delta = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64)
    target = np.clip(target + np.clip(delta, -0.025, 0.025), limits_min, limits_max)

def on_gripper(msg):
    if msg.data:
        franka.gripper.open()
    else:
        franka.gripper.close()

node.create_subscription(Twist, "/ee_delta", on_delta, 10)
node.create_subscription(Bool, "/gripper_command", on_gripper, 10)
print("GUI ready. Control from a second terminal. Ctrl+C here to exit.", flush=True)
try:
    while simulation_app.is_running():
        rclpy.spin_once(node, timeout_sec=0.0)
        action = controller.forward(target_end_effector_position=target,
                                    target_end_effector_orientation=orientation)
        franka.apply_action(action)
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
