from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController


# ============================================================
# CONFIG
# ============================================================

CUBE_SIZE = 0.05

CUBE_START = np.array([
    0.50,
    0.00,
    CUBE_SIZE / 2.0,
])

TARGET_POSITION = np.array([
    0.40,
    0.30,
    CUBE_SIZE / 2.0,
])

# Calibrated successfully in previous test
GRASP_TARGET_Z = 0.025

PRE_GRASP_Z = 0.20
LIFT_TARGET_Z = 0.25
PRE_PLACE_Z = 0.25

SUCCESS_TOLERANCE = 0.08


# ============================================================
# WORLD
# ============================================================

world = World(
    stage_units_in_meters=1.0
)

world.scene.add_default_ground_plane()


# ============================================================
# FRANKA
# ============================================================

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


# ============================================================
# CUBE
# ============================================================

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=CUBE_START,
        scale=np.array([
            CUBE_SIZE,
            CUBE_SIZE,
            CUBE_SIZE,
        ]),
        mass=0.05,
    )
)


# ============================================================
# RESET
# ============================================================

world.reset()

for _ in range(100):
    world.step(render=False)


# ============================================================
# HAND / FINGERS
# ============================================================

hand = XFormPrim(
    prim_path="/World/Franka/panda_hand"
)

left_finger = XFormPrim(
    prim_path="/World/Franka/panda_leftfinger"
)

right_finger = XFormPrim(
    prim_path="/World/Franka/panda_rightfinger"
)


# ============================================================
# RMPFLOW
# ============================================================

controller = RMPFlowController(
    name="franka_rmpflow_controller",
    robot_articulation=franka,
)


# ============================================================
# ORIENTATION
# ============================================================

top_down_orientation = euler_angles_to_quat(
    np.array([
        0.0,
        np.pi,
        0.0,
    ])
)


print()
print("==========================================")
print("TASK A — PICK & PLACE")
print("==========================================")

print(
    "Requested orientation:",
    np.round(top_down_orientation, 4)
)

print(
    "EE prim:",
    franka.end_effector.prim_path
)

print(
    "Cube start:",
    CUBE_START
)

print(
    "Target:",
    TARGET_POSITION
)

print(
    "Calibrated grasp Z:",
    GRASP_TARGET_Z
)

print("==========================================")
print()


# ============================================================
# HELPERS
# ============================================================

def get_finger_positions():

    left_pos, _ = (
        left_finger.get_world_pose()
    )

    right_pos, _ = (
        right_finger.get_world_pose()
    )

    return (
        left_pos.copy(),
        right_pos.copy(),
    )


def get_gripper_center():

    left_pos, right_pos = (
        get_finger_positions()
    )

    return (
        left_pos + right_pos
    ) / 2.0


def get_finger_separation():

    left_pos, right_pos = (
        get_finger_positions()
    )

    return np.linalg.norm(
        left_pos - right_pos
    )


# ============================================================
# MOVE
# ============================================================

def move_to(
    target,
    steps=300,
):

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    for _ in range(steps):

        action = controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=top_down_orientation,
        )

        franka.apply_action(action)

        world.step(
            render=False
        )

    center = get_gripper_center()

    error = np.linalg.norm(
        center[:2] - target[:2]
    )

    print(
        "  RMPFlow target:",
        np.round(target, 4)
    )

    print(
        "  Gripper center:",
        np.round(center, 4)
    )

    print(
        "  XY error:",
        f"{error:.4f} m"
    )


# ============================================================
# GRIPPER
# ============================================================

def open_gripper(
    steps=200,
):

    franka.gripper.open()

    for _ in range(steps):
        world.step(render=False)


def close_gripper(
    steps=300,
):

    franka.gripper.close()

    for _ in range(steps):
        world.step(render=False)


# ============================================================
# STATE PRINT
# ============================================================

def print_state(title):

    cube_pos, _ = (
        cube.get_world_pose()
    )

    center = (
        get_gripper_center()
    )

    separation = (
        get_finger_separation()
    )

    print()
    print(
        "------------------------------------------"
    )

    print(title)

    print(
        "Cube:",
        np.round(cube_pos, 4)
    )

    print(
        "Gripper center:",
        np.round(center, 4)
    )

    print(
        "Finger separation:",
        f"{separation:.4f} m"
    )

    print(
        "Center - cube:",
        np.round(
            center - cube_pos,
            4
        )
    )

    print(
        "------------------------------------------"
    )

    print()


# ============================================================
# WAYPOINTS
# ============================================================

pre_grasp = np.array([
    CUBE_START[0],
    CUBE_START[1],
    PRE_GRASP_Z,
])


grasp_position = np.array([
    CUBE_START[0],
    CUBE_START[1],
    GRASP_TARGET_Z,
])


lift_position = np.array([
    CUBE_START[0],
    CUBE_START[1],
    LIFT_TARGET_Z,
])


pre_place = np.array([
    TARGET_POSITION[0],
    TARGET_POSITION[1],
    PRE_PLACE_Z,
])


# ============================================================
# STEP 0 — OPEN
# ============================================================

print("[0] OPEN GRIPPER")

open_gripper(
    steps=200
)

print(
    "Finger separation:",
    f"{get_finger_separation():.4f} m"
)


# ============================================================
# STEP 1 — PRE-GRASP
# ============================================================

print()
print("[1] PRE-GRASP")

move_to(
    pre_grasp,
    steps=400,
)


# ============================================================
# STEP 2 — DESCEND
# ============================================================

print()
print("[2] DESCEND")

move_to(
    grasp_position,
    steps=400,
)

print_state(
    "BEFORE GRASP"
)


# ============================================================
# STEP 3 — GRASP
# ============================================================

print("[3] CLOSE GRIPPER")

close_gripper(
    steps=300
)

print_state(
    "AFTER GRASP"
)


# ============================================================
# VERIFY GRIPPER CONTACT
# ============================================================

closed_separation = (
    get_finger_separation()
)

print(
    "Closed finger separation:",
    f"{closed_separation:.4f} m"
)


# ============================================================
# STEP 4 — LIFT
# ============================================================

print()
print("[4] LIFT")

cube_before_lift, _ = (
    cube.get_world_pose()
)

cube_before_lift = (
    cube_before_lift.copy()
)


move_to(
    lift_position,
    steps=500,
)


for _ in range(100):
    world.step(render=False)


cube_after_lift, _ = (
    cube.get_world_pose()
)

cube_after_lift = (
    cube_after_lift.copy()
)


vertical_lift = (
    cube_after_lift[2]
    - cube_before_lift[2]
)


print(
    "Cube before lift:",
    np.round(
        cube_before_lift,
        4
    )
)

print(
    "Cube after lift:",
    np.round(
        cube_after_lift,
        4
    )
)

print(
    "Vertical lift:",
    f"{vertical_lift:.4f} m"
)


# ============================================================
# ABORT IF GRASP FAILED
# ============================================================

if vertical_lift < 0.03:

    print()
    print(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )

    print(
        "GRASP FAILED — ABORTING PICK & PLACE"
    )

    print(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )

    simulation_app.close()

    raise SystemExit


print()
print(">>> GRASP CONFIRMED <<<")
print()


# ============================================================
# MEASURE GRASP TRANSFORM
# ============================================================

#
# We now know exactly where the cube is relative
# to the gripper center while being held.
#
# This lets us calculate the place height instead
# of guessing it.
#

gripper_center = (
    get_gripper_center()
)

cube_position, _ = (
    cube.get_world_pose()
)


cube_relative_to_gripper = (
    cube_position
    - gripper_center
)


print(
    "=========================================="
)

print(
    "MEASURED GRASP OFFSET"
)

print(
    "=========================================="
)

print(
    "Gripper center:",
    np.round(
        gripper_center,
        4
    )
)

print(
    "Cube:",
    np.round(
        cube_position,
        4
    )
)

print(
    "Cube - gripper:",
    np.round(
        cube_relative_to_gripper,
        4
    )
)

print(
    "=========================================="
)

print()


# ============================================================
# STEP 5 — MOVE ABOVE TARGET
# ============================================================

print("[5] MOVE ABOVE TARGET")

move_to(
    pre_place,
    steps=600,
)


for _ in range(100):
    world.step(render=False)


print_state(
    "ABOVE TARGET"
)


# ============================================================
# CALCULATE PLACE TARGET Z
# ============================================================

#
# Empirical relation from calibration:
#
# actual gripper center Z
#     ~= RMPFlow target Z + offset
#
# Measure that offset at the current safe pose rather
# than hardcoding 0.0417.
#

current_center = (
    get_gripper_center()
)


controller_center_z_offset = (
    current_center[2]
    - pre_place[2]
)


#
# We want:
#
# cube center Z = TARGET_POSITION[2]
#
# cube_z =
#     gripper_center_z
#     + cube_relative_to_gripper_z
#
# Therefore:
#
# desired gripper center z =
#     target cube z
#     - cube relative z
#

desired_gripper_center_z = (
    TARGET_POSITION[2]
    - cube_relative_to_gripper[2]
)


#
# Convert desired actual gripper-center height
# back into RMPFlow target height.
#

place_target_z = (
    desired_gripper_center_z
    - controller_center_z_offset
)


print()
print(
    "=========================================="
)

print(
    "PLACE HEIGHT CALCULATION"
)

print(
    "=========================================="
)

print(
    "Controller → center Z offset:",
    f"{controller_center_z_offset:.4f} m"
)

print(
    "Cube → gripper relative Z:",
    f"{cube_relative_to_gripper[2]:.4f} m"
)

print(
    "Desired cube center Z:",
    f"{TARGET_POSITION[2]:.4f} m"
)

print(
    "Desired gripper center Z:",
    f"{desired_gripper_center_z:.4f} m"
)

print(
    "Calculated RMPFlow place Z:",
    f"{place_target_z:.4f} m"
)

print(
    "=========================================="
)

print()


# ============================================================
# STEP 6 — LOWER
# ============================================================

print("[6] LOWER TO PLACE")

place_position = np.array([
    TARGET_POSITION[0],
    TARGET_POSITION[1],
    place_target_z,
])


move_to(
    place_position,
    steps=450,
)


for _ in range(100):
    world.step(render=False)


print_state(
    "BEFORE RELEASE"
)


# ============================================================
# STEP 7 — RELEASE
# ============================================================

print("[7] RELEASE")

open_gripper(
    steps=300
)


# Let cube settle
for _ in range(300):
    world.step(render=False)


cube_after_release, _ = (
    cube.get_world_pose()
)


print(
    "Cube after release:",
    np.round(
        cube_after_release,
        4
    )
)


# ============================================================
# STEP 8 — RETREAT
# ============================================================

print()
print("[8] RETREAT")

retreat = np.array([
    TARGET_POSITION[0],
    TARGET_POSITION[1],
    PRE_PLACE_Z,
])


move_to(
    retreat,
    steps=400,
)


for _ in range(200):
    world.step(render=False)


# ============================================================
# FINAL EVALUATION
# ============================================================

cube_final_position, cube_final_orientation = (
    cube.get_world_pose()
)


xy_error = np.linalg.norm(
    cube_final_position[:2]
    - TARGET_POSITION[:2]
)


xyz_error = np.linalg.norm(
    cube_final_position
    - TARGET_POSITION
)


success = (
    xy_error
    < SUCCESS_TOLERANCE
)


print()
print()
print(
    "=========================================="
)

print(
    "TASK A — FINAL RESULT"
)

print(
    "=========================================="
)


print(
    "Cube start:",
    np.round(
        CUBE_START,
        4
    )
)


print(
    "Cube final:",
    np.round(
        cube_final_position,
        4
    )
)


print(
    "Target:",
    np.round(
        TARGET_POSITION,
        4
    )
)


print()


print(
    "XY error:",
    f"{xy_error:.4f} m"
)


print(
    "XYZ error:",
    f"{xyz_error:.4f} m"
)


print(
    "Tolerance:",
    f"{SUCCESS_TOLERANCE:.4f} m"
)


print()


print(
    "SUCCESS:",
    success
)


print(
    "=========================================="
)

print()


simulation_app.close()