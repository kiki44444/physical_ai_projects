import os
from pathlib import Path
import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

DATA_DIR = Path(os.environ.get("DEMO_DIR", "datasets/demonstrations"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CUBE_SIZE = 0.05
GOAL = np.array([0.40, 0.30, CUBE_SIZE / 2])
SUCCESS_TOLERANCE = 0.05
RANDOM_X = (0.44, 0.56)
RANDOM_Y = (-0.08, 0.08)
rng = np.random.default_rng()
START_CUBE = np.array([0.50, 0.0, CUBE_SIZE / 2])
START_TARGET = np.array([0.50, 0.0, 0.20])
MIN_TARGET = np.array([0.25, -0.35, 0.035])
MAX_TARGET = np.array([0.75, 0.45, 0.50])
SAMPLE_PERIOD = 0.1  # simulation seconds = 10 Hz

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
url = ("http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
       "Assets/Isaac/4.0/Isaac/Robots/Franka/franka.usd")
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka", usd_path=url))
cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube",
                                    position=START_CUBE.copy(),
                                    scale=np.array([0.05, 0.05, 0.05]), mass=0.05))
goal_marker = world.scene.add(VisualCuboid(
    prim_path="/World/GoalMarker", name="goal_marker",
    position=np.array([GOAL[0], GOAL[1], 0.002]),
    scale=np.array([0.065, 0.065, 0.004]),
    color=np.array([0.0, 0.9, 0.2]),
))
world.reset()
set_camera_view(eye=np.array([1.5, 1.5, 1.2]), target=np.array([0.45, 0.15, 0.1]),
                camera_prim_path="/OmniverseKit_Persp")
controller = RMPFlowController(name="teleop_rmpflow", robot_articulation=franka)
orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

rclpy.init()
node = rclpy.create_node("franka_gui_recorder")
target = START_TARGET.copy()
last_delta = np.zeros(3, dtype=np.float64)
gripper_open = True
recording = False
rows = []
last_sample_time = None
pending = []
current_start = START_CUBE.copy()
accumulated_delta = np.zeros(3, dtype=np.float64)


def on_delta(msg):
    global target, last_delta, accumulated_delta
    delta = np.clip(np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=float),
                    -0.025, 0.025)
    new_target = np.clip(target + delta, MIN_TARGET, MAX_TARGET)
    last_delta = new_target - target
    accumulated_delta += last_delta
    target = new_target


def on_gripper(msg):
    global gripper_open
    gripper_open = bool(msg.data)
    if gripper_open:
        franka.gripper.open()
    else:
        franka.gripper.close()


def on_record(msg):
    if msg.data in ("r", "p", "n"):
        pending.append(msg.data)


def save_episode():
    global recording, rows
    recording = False
    if not rows:
        print("No samples to save", flush=True)
        return
    index = 0
    while (DATA_DIR / f"episode_{index:03d}.npz").exists():
        index += 1
    path = DATA_DIR / f"episode_{index:03d}.npz"
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    final_pos, _ = cube.get_world_pose()
    xy_error = float(np.linalg.norm(final_pos[:2] - GOAL[:2]))
    z_error = float(abs(final_pos[2] - GOAL[2]))
    success = bool(xy_error <= SUCCESS_TOLERANCE and z_error <= 0.03)
    arrays.update(
        success=np.array(success),
        final_xy_error=np.array(xy_error),
        final_z_error=np.array(z_error),
        goal_position=GOAL.copy(),
        cube_start_position=current_start.copy(),
    )
    np.savez_compressed(path, **arrays)
    print(f"Result: success={success}, XY error={xy_error:.3f}m, Z error={z_error:.3f}m", flush=True)
    print(f"Saved {len(rows)} samples to {path}", flush=True)
    rows = []


def reset_episode():
    global target, last_delta, gripper_open, last_sample_time, rows, current_start, accumulated_delta
    if recording:
        print("Stop recording with P before resetting", flush=True)
        return
    rows = []
    world.reset()
    current_start = np.array([rng.uniform(*RANDOM_X), rng.uniform(*RANDOM_Y), CUBE_SIZE / 2])
    cube.set_world_pose(position=current_start.copy(),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))
    controller.reset()
    target = START_TARGET.copy()
    last_delta = np.zeros(3)
    accumulated_delta[:] = 0.0
    gripper_open = True
    franka.gripper.open()
    last_sample_time = None
    print(f"New cube start: {np.round(current_start, 3)}; goal: {GOAL}. Press R.", flush=True)


def capture(t):
    ee_pos, ee_quat = franka.end_effector.get_world_pose()
    cube_pos, cube_quat = cube.get_world_pose()
    return {
        "time": np.array(t, dtype=np.float64),
        "joint_positions": np.asarray(franka.get_joint_positions(), dtype=np.float64).copy(),
        "joint_velocities": np.asarray(franka.get_joint_velocities(), dtype=np.float64).copy(),
        "ee_position": np.asarray(ee_pos, dtype=np.float64).copy(),
        "ee_orientation": np.asarray(ee_quat, dtype=np.float64).copy(),
        "cube_position": np.asarray(cube_pos, dtype=np.float64).copy(),
        "cube_orientation": np.asarray(cube_quat, dtype=np.float64).copy(),
        "action": np.array([*accumulated_delta, float(gripper_open)], dtype=np.float64),
        "target_position": target.copy(),
        "goal_position_step": GOAL.copy(),
    }


node.create_subscription(Twist, "/ee_delta", on_delta, 10)
node.create_subscription(Bool, "/gripper_command", on_gripper, 10)
node.create_subscription(String, "/demo_command", on_record, 10)
print("Pick & Place: N randomizes cube; R starts; P saves and evaluates", flush=True)

try:
    while simulation_app.is_running():
        rclpy.spin_once(node, timeout_sec=0.0)
        while pending:
            cmd = pending.pop(0)
            if cmd == "r":
                if not recording:
                    rows = []
                    last_sample_time = None
                    accumulated_delta[:] = 0.0
                    recording = True
                    print("Recording started", flush=True)
            elif cmd == "p":
                if recording:
                    save_episode()
            elif cmd == "n":
                reset_episode()
        if recording:
            now = world.current_time
            if last_sample_time is None or now - last_sample_time >= SAMPLE_PERIOD - 1e-5:
                rows.append(capture(now))
                accumulated_delta[:] = 0.0
                last_sample_time = now
        action = controller.forward(target_end_effector_position=target,
                                    target_end_effector_orientation=orientation)
        franka.apply_action(action)
        world.step(render=True)
        # Delta is an event, not a persistent velocity command.
        last_delta[:] = 0.0
except KeyboardInterrupt:
    pass
finally:
    if recording:
        save_episode()
    node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
