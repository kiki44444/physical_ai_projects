from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
print("1. SimulationApp started", flush=True)

import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

rclpy.init()
node = Node("franka_joint_state_publisher")

# Publisher:
# Isaac Sim -> ROS2
publisher = node.create_publisher(
    JointState,
    "/joint_states",
    10,
)

print("2. ROS publisher created", flush=True)


# latest command received from ROS2
target_joint_positions = None


def joint_command_callback(msg):
    global target_joint_positions

    if len(msg.position) != 9:
        print(f"Invalid joint command: expected 9 joints, " f"got {len(msg.position)}",flush=True)
        return

    target_joint_positions = np.array(msg.position, dtype=np.float32)
    print("Received joint command:",target_joint_positions,flush=True,)

# Subscriber:
# ROS2 -> Isaac Sim
joint_command_subscriber = node.create_subscription(
    JointState,
    "/joint_command",
    joint_command_callback,
    10
)
print("2. ROS2 publisher/subscriber created", flush=True)

# ISAAC
from omni.isaac.core import World
from omni.isaac.franka import Franka

print("3. Creating world", flush=True)
world = World(stage_units_in_meters=1.0)

print("4. Loading Franka...", flush=True)
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
print("5. Franka loaded", flush=True)

world.reset()
print("6. World reset", flush=True)

print("Initial joint positions:",franka.get_joint_positions(),flush=True)

joint_names = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


# smulation + ROS2 loop
print("7. ROS2 control loop started", flush=True)

count = 0
while simulation_app.is_running():
    # get ROS2 messages
    rclpy.spin_once(node, timeout_sec=0)
    # apply command
    if target_joint_positions is not None:
        action = ArticulationAction(joint_positions=target_joint_positions)
        franka.apply_action(action)
    # Step
    world.step(render=False)
    # read actual Franka joint positions
    positions = franka.get_joint_positions()
    # Publish /joint_states
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = joint_names
    msg.position = positions.tolist()

    publisher.publish(msg)
    # Print
    if count % 500 == 0:
        print("Current joints:", np.round(positions, 4),flush=True,)
    count += 1

# shutdown
print("Shutting down...", flush=True)
node.destroy_node()
rclpy.shutdown()

simulation_app.close()