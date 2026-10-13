
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np

from scipy.spatial.transform import Rotation as R

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController


# CONFIG
BLOCK_SIZE = np.array([0.06, 0.04, 0.04])
BLOCK_START = np.array([0.50, 0.00, 0.02])
TARGET_POSITION = np.array([0.40, 0.30, 0.02])
TARGET_YAW_DEG = 90.0
GRASP_Z = 0.020
SAFE_Z = 0.25
POSITION_TOLERANCE = 0.015
ORIENTATION_TOLERANCE_DEG = 5.0

# Rotation interpolation
ROTATION_STEPS = 20

# Prevent excessive lowering if placement calibration fails
MIN_PLACE_Z = -0.03
MAX_PLACE_Z = 0.12


# QUATERNION HELPERS
# Isaac Sim: [w, x, y, z]
# SciPy:     [x, y, z, w]

def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])

def quat_multiply(q1, q2):
    r1 = R.from_quat(wxyz_to_xyzw(q1))
    r2 = R.from_quat(wxyz_to_xyzw(q2))
    return xyzw_to_wxyz((r1 * r2).as_quat())

def quat_error_deg(q1, q2):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)

    return np.degrees(2.0 * np.arccos(dot))

def yaw_quaternion(degrees):
    return xyzw_to_wxyz(
        R.from_euler("z", degrees, degrees=True).as_quat()
    )


# WORLD
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

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

block = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Block",
        name="block",
        position=BLOCK_START,
        scale=BLOCK_SIZE,
        mass=0.05,
    )
)

world.reset()
from omni.isaac.core.utils.viewports import set_camera_view

set_camera_view(
    eye=np.array([1.5, 1.5, 1.2]),
    target=np.array([0.45, 0.15, 0.1]),
    camera_prim_path="/OmniverseKit_Persp",
)

for _ in range(100):
    world.step(render=True)


# ROBOT REFERENCES

left_finger = XFormPrim(
    prim_path="/World/Franka/panda_leftfinger"
)

right_finger = XFormPrim(
    prim_path="/World/Franka/panda_rightfinger"
)

hand = XFormPrim(
    prim_path="/World/Franka/panda_hand"
)

controller = RMPFlowController(
    name="franka_rmpflow_controller",
    robot_articulation=franka,
)

# ORIENTATION

initial_orientation = euler_angles_to_quat(
    np.array([0.0, np.pi, 0.0])
)

rotation_90 = yaw_quaternion(TARGET_YAW_DEG)

# Rotate the commanded hand orientation about world Z.
final_orientation = quat_multiply(
    rotation_90,
    initial_orientation,
)

# Initial block orientation is identity.
target_block_orientation = rotation_90.copy()


# HELPERS

def get_gripper_center():
    left, _ = left_finger.get_world_pose()
    right, _ = right_finger.get_world_pose()
    return (left + right) / 2.0


def get_finger_separation():
    left, _ = left_finger.get_world_pose()
    right, _ = right_finger.get_world_pose()
    return float(np.linalg.norm(left - right))


def get_block_pose():
    p, q = block.get_world_pose()
    return p.copy(), q.copy()


def move_to(position, orientation, steps=350):
    position = np.asarray(position, dtype=float)

    for _ in range(steps):
        action = controller.forward(
            target_end_effector_position=position,
            target_end_effector_orientation=orientation,
        )
        franka.apply_action(action)
        world.step(render=True)

    center = get_gripper_center()
    block_pos, _ = get_block_pose()

    print("  Target:", np.round(position, 4))
    print("  Finger center:", np.round(center, 4))
    print("  Block:", np.round(block_pos, 4))


def open_gripper(steps=200):
    franka.gripper.open()
    for _ in range(steps):
        world.step(render=True)


def close_gripper(steps=300):
    franka.gripper.close()
    for _ in range(steps):
        world.step(render=True)


def print_state(label):
    block_pos, block_quat = get_block_pose()
    hand_pos, hand_quat = hand.get_world_pose()

    print("\n" + "=" * 50)
    print(label)
    print("=" * 50)
    print("Block position:", np.round(block_pos, 4))
    print("Block quaternion:", np.round(block_quat, 4))
    print("Hand position:", np.round(hand_pos, 4))
    print("Hand quaternion:", np.round(hand_quat, 4))
    print("Finger center:", np.round(get_gripper_center(), 4))
    print("Finger separation:", round(get_finger_separation(), 4))


def rotate_gripper_at_height(position, q_start, q_end):
    """Interpolate the desired hand orientation using SLERP."""

    r_start = R.from_quat(wxyz_to_xyzw(q_start))
    r_end = R.from_quat(wxyz_to_xyzw(q_end))

    relative = r_start.inv() * r_end
    rotation_vector = relative.as_rotvec()

    for i in range(1, ROTATION_STEPS + 1):
        alpha = i / ROTATION_STEPS

        interpolated = (
            r_start
            * R.from_rotvec(alpha * rotation_vector)
        )

        q = xyzw_to_wxyz(interpolated.as_quat())

        move_to(position, q, steps=80)


# TASK START
print("\n" + "=" * 50)
print("TASK B — PRECISE PLACEMENT")
print("=" * 50)
print("Block size:", BLOCK_SIZE)
print("Start:", BLOCK_START)
print("Target:", TARGET_POSITION)
print("Target yaw:", TARGET_YAW_DEG)
print("Grasp Z:", GRASP_Z)


# 1. PRE-GRASP
print("\n[1] PRE-GRASP")

open_gripper()

move_to(
    np.array([BLOCK_START[0], BLOCK_START[1], SAFE_Z]),
    initial_orientation,
    steps=450,
)


# 2. DESCEND
print("\n[2] DESCEND")

move_to(
    np.array([BLOCK_START[0], BLOCK_START[1], GRASP_Z]),
    initial_orientation,
    steps=450,
)

print_state("BEFORE GRASP")


# 3. GRASP
print("\n[3] GRASP")

close_gripper()

print_state("AFTER GRASP")


# 4. LIFT
print("\n[4] LIFT")

block_before_lift, _ = get_block_pose()

lift_position = np.array([
    BLOCK_START[0],
    BLOCK_START[1],
    SAFE_Z,
])

move_to(
    lift_position,
    initial_orientation,
    steps=550,
)

block_after_lift, _ = get_block_pose()

lift_amount = block_after_lift[2] - block_before_lift[2]

print("Lift amount:", round(lift_amount, 4))

if lift_amount < 0.03:
    print("GRASP FAILED — STOPPING TASK")
    simulation_app.close()
    raise SystemExit(1)


# 5. ROTATE BLOCK
print("\n[5] ROTATE 90 DEGREES")

# Measure actual block and hand orientations before rotating.
block_before_rotation, block_q_before = get_block_pose()
_, hand_q_before = hand.get_world_pose()

rotate_gripper_at_height(
    lift_position,
    initial_orientation,
    final_orientation,
)

block_after_rotation, block_q_after = get_block_pose()
_, hand_q_after = hand.get_world_pose()

actual_rotation = (
    R.from_quat(wxyz_to_xyzw(block_q_after))
    * R.from_quat(wxyz_to_xyzw(block_q_before)).inv()
)

actual_rotation_angle = np.degrees(
    actual_rotation.magnitude()
)

print_state("AFTER ROTATION")
print("Actual block rotation:", round(actual_rotation_angle, 3), "deg")


# 6. MOVE ABOVE TARGET
print("\n[6] MOVE ABOVE TARGET")

pre_place = np.array([
    TARGET_POSITION[0],
    TARGET_POSITION[1],
    SAFE_Z,
])

move_to(
    pre_place,
    final_orientation,
    steps=650,
)

print_state("ABOVE TARGET")


# 7. MEASURE ACTUAL GRASP OFFSET

print("\n[7] CALCULATE PLACE HEIGHT")

block_pos, block_q = get_block_pose()
center = get_gripper_center()

# Actual world-frame vertical offset while holding block.
block_relative_z = block_pos[2] - center[2]

# Empirical RMPFlow target -> finger center offset.
controller_center_offset_z = center[2] - pre_place[2]

desired_center_z = (
    TARGET_POSITION[2] - block_relative_z
)

place_z = (
    desired_center_z - controller_center_offset_z
)

print("Block relative Z:", round(block_relative_z, 4))
print("Controller center offset Z:", round(controller_center_offset_z, 4))
print("Calculated place Z:", round(place_z, 4))

if not (MIN_PLACE_Z <= place_z <= MAX_PLACE_Z):
    print("PLACE HEIGHT OUT OF SAFE CALIBRATION RANGE")
    simulation_app.close()
    raise SystemExit(1)


# 8. PRECISE LOWERING
print("\n[8] PRECISE LOWERING")

place_position = np.array([
    TARGET_POSITION[0],
    TARGET_POSITION[1],
    place_z,
])

move_to(
    place_position,
    final_orientation,
    steps=500,
)

print_state("BEFORE RELEASE")


# 9. RELEASE
print("\n[9] RELEASE")

open_gripper(steps=300)

for _ in range(250):
    world.step(render=True)

print_state("AFTER RELEASE")


# 10. RETREAT
print("\n[10] RETREAT")

move_to(
    pre_place,
    final_orientation,
    steps=450,
)

for _ in range(250):
    world.step(render=True)


# 11. EVALUATION
final_pos, final_quat = get_block_pose()

position_error = np.linalg.norm(
    final_pos - TARGET_POSITION
)

xy_error = np.linalg.norm(
    final_pos[:2] - TARGET_POSITION[:2]
)

orientation_error = quat_error_deg(
    final_quat,
    target_block_orientation,
)

position_success = (
    position_error < POSITION_TOLERANCE
)

orientation_success = (
    orientation_error < ORIENTATION_TOLERANCE_DEG
)

success = position_success and orientation_success


print("\n" + "=" * 55)
print("TASK B — FINAL RESULT")
print("=" * 55)

print("Target position:", np.round(TARGET_POSITION, 4))
print("Final position: ", np.round(final_pos, 4))
print("Position error:", round(position_error, 5), "m")
print("XY error:", round(xy_error, 5), "m")

print()

print(
    "Target quaternion:",
    np.round(target_block_orientation, 4),
)

print(
    "Final quaternion: ",
    np.round(final_quat, 4),
)

print(
    "Orientation error:",
    round(orientation_error, 3),
    "degrees",
)

print()

print("Position success:", position_success)
print("Orientation success:", orientation_success)
print("TASK B SUCCESS:", success)

print("=" * 55)

print("Task finished. GUI will remain open.")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()