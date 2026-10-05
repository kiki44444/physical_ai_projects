from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

print("1. SimulationApp started", flush=True)

# ROS2 FIRST
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

rclpy.init()

node = Node("franka_joint_state_publisher")
publisher = node.create_publisher(
    JointState,
    "/joint_states",
    10,
)

print("2. ROS publisher created", flush=True)


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


# LOOP
count = 0

while simulation_app.is_running():

    world.step(render=False)

    positions = franka.get_joint_positions()

    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()

    msg.name = [
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

    msg.position = positions.tolist()

    publisher.publish(msg)
    rclpy.spin_once(node, timeout_sec=0)

    if count % 100 == 0:
        print(
            "7. PUBLISHED:",
            positions,
            flush=True
        )

    count += 1