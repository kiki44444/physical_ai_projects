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

SAFE_Z = 0.25

LIFT_TARGET_Z = 0.25

LIFT_SUCCESS_THRESHOLD = 0.03

# Test several grasp heights independently
GRASP_CANDIDATES = [
    0.025,
    0.020,
    0.015,
    0.010,
]


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
# HAND / FINGER LINKS
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
print("ORIENTATION")
print("==========================================")

print(
    "Requested quaternion:",
    np.round(top_down_orientation, 4)
)

print(
    "EE prim:",
    franka.end_effector.prim_path
)

print("==========================================")
print()


# ============================================================
# HELPER: GET FINGER POSITIONS
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


# ============================================================
# HELPER: GET GRIPPER CENTER
# ============================================================

def get_gripper_center():

    left_pos, right_pos = (
        get_finger_positions()
    )

    return (
        left_pos + right_pos
    ) / 2.0


# ============================================================
# HELPER: FINGER SEPARATION
# ============================================================

def get_finger_separation():

    left_pos, right_pos = (
        get_finger_positions()
    )

    return np.linalg.norm(
        left_pos - right_pos
    )


# ============================================================
# HELPER: MOVE
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

    center = (
        get_gripper_center()
    )

    print(
        "RMPFlow target:",
        np.round(target, 4)
    )

    print(
        "Gripper center:",
        np.round(center, 4)
    )

    print(
        "Target -> center:",
        np.round(
            center - target,
            4,
        )
    )


# ============================================================
# HELPER: OPEN
# ============================================================

def open_gripper(
    steps=200,
):

    franka.gripper.open()

    for _ in range(steps):
        world.step(render=False)


# ============================================================
# HELPER: CLOSE
# ============================================================

def close_gripper(
    steps=250,
):

    franka.gripper.close()

    for _ in range(steps):
        world.step(render=False)


# ============================================================
# HELPER: PRINT STATE
# ============================================================

def print_state(
    title,
):

    cube_pos, _ = (
        cube.get_world_pose()
    )

    hand_pos, _ = (
        hand.get_world_pose()
    )

    left_pos, right_pos = (
        get_finger_positions()
    )

    center = (
        get_gripper_center()
    )

    separation = (
        get_finger_separation()
    )

    print()
    print(
        "=========================================="
    )

    print(title)

    print(
        "=========================================="
    )

    print(
        "Cube:",
        np.round(
            cube_pos,
            4,
        )
    )

    print(
        "Hand:",
        np.round(
            hand_pos,
            4,
        )
    )

    print(
        "Left finger:",
        np.round(
            left_pos,
            4,
        )
    )

    print(
        "Right finger:",
        np.round(
            right_pos,
            4,
        )
    )

    print(
        "Gripper center:",
        np.round(
            center,
            4,
        )
    )

    print(
        "Center - cube:",
        np.round(
            center - cube_pos,
            4,
        )
    )

    print(
        "Finger separation:",
        f"{separation:.4f} m"
    )

    print(
        "=========================================="
    )

    print()


# ============================================================
# HELPER: RESET CUBE
# ============================================================

def reset_cube():

    cube.set_world_pose(
        position=CUBE_START
    )

    # Remove any velocity from previous test
    try:
        cube.set_linear_velocity(
            np.zeros(3)
        )

        cube.set_angular_velocity(
            np.zeros(3)
        )

    except Exception:
        pass

    for _ in range(100):
        world.step(render=False)


# ============================================================
# START
# ============================================================

print()
print(
    "=========================================="
)
print(
    "MULTI-HEIGHT GRASP CALIBRATION"
)
print(
    "=========================================="
)

print(
    "Cube size:",
    CUBE_SIZE,
)

print(
    "Cube start:",
    CUBE_START,
)

print(
    "Candidate grasp heights:",
    GRASP_CANDIDATES,
)

print(
    "=========================================="
)
print()


# ============================================================
# SAFE POSITION
# ============================================================

safe_position = np.array([
    CUBE_START[0],
    CUBE_START[1],
    SAFE_Z,
])


# ============================================================
# RESULTS
# ============================================================

results = []

working_grasp_z = None


# ============================================================
# TEST EACH GRASP HEIGHT
# ============================================================

for test_index, grasp_z in enumerate(
    GRASP_CANDIDATES,
    start=1,
):

    print()
    print()
    print(
        "##################################################"
    )

    print(
        f"TEST {test_index}/{len(GRASP_CANDIDATES)}"
    )

    print(
        f"GRASP TARGET Z = {grasp_z:.3f} m"
    )

    print(
        "##################################################"
    )

    print()


    # ========================================================
    # STEP 1 — MOVE SAFE
    # ========================================================

    print("[1] MOVE TO SAFE POSITION")

    move_to(
        safe_position,
        steps=400,
    )


    # ========================================================
    # STEP 2 — OPEN GRIPPER
    # ========================================================

    print()
    print("[2] OPEN GRIPPER")

    open_gripper(
        steps=200
    )


    open_separation = (
        get_finger_separation()
    )

    print(
        "Open finger separation:",
        f"{open_separation:.4f} m"
    )


    # ========================================================
    # STEP 3 — RESET CUBE
    # ========================================================

    print()
    print("[3] RESET CUBE")

    reset_cube()

    cube_reset_pos, _ = (
        cube.get_world_pose()
    )

    print(
        "Cube:",
        np.round(
            cube_reset_pos,
            4,
        )
    )


    # ========================================================
    # STEP 4 — DESCEND
    # ========================================================

    print()
    print(
        f"[4] DESCEND TO Z={grasp_z:.3f}"
    )

    grasp_position = np.array([
        CUBE_START[0],
        CUBE_START[1],
        grasp_z,
    ])

    move_to(
        grasp_position,
        steps=350,
    )


    # Let physics settle
    for _ in range(50):
        world.step(render=False)


    cube_before_close, _ = (
        cube.get_world_pose()
    )

    cube_before_close = (
        cube_before_close.copy()
    )


    print_state(
        "BEFORE CLOSE"
    )


    # ========================================================
    # SAFETY / CONTACT CHECK
    # ========================================================

    descent_displacement = np.linalg.norm(
        cube_before_close
        - CUBE_START
    )

    print(
        "Cube movement during descent:",
        f"{descent_displacement:.5f} m"
    )


    # ========================================================
    # STEP 5 — CLOSE
    # ========================================================

    print()
    print("[5] CLOSE GRIPPER")

    close_gripper(
        steps=300
    )


    closed_separation = (
        get_finger_separation()
    )


    cube_after_close, _ = (
        cube.get_world_pose()
    )

    cube_after_close = (
        cube_after_close.copy()
    )


    print_state(
        "AFTER CLOSE"
    )


    print(
        "Open separation:",
        f"{open_separation:.4f} m"
    )

    print(
        "Closed separation:",
        f"{closed_separation:.4f} m"
    )


    closing_amount = (
        open_separation
        - closed_separation
    )


    print(
        "Closing amount:",
        f"{closing_amount:.4f} m"
    )


    cube_close_motion = np.linalg.norm(
        cube_after_close
        - cube_before_close
    )


    print(
        "Cube movement while closing:",
        f"{cube_close_motion:.5f} m"
    )


    # ========================================================
    # STEP 6 — LIFT
    # ========================================================

    print()
    print("[6] LIFT")


    cube_before_lift = (
        cube_after_close.copy()
    )


    lift_position = np.array([
        CUBE_START[0],
        CUBE_START[1],
        LIFT_TARGET_Z,
    ])


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


    # ========================================================
    # RESULT
    # ========================================================

    vertical_lift = (
        cube_after_lift[2]
        - cube_before_lift[2]
    )


    total_cube_motion = np.linalg.norm(
        cube_after_lift
        - cube_before_lift
    )


    success = (
        vertical_lift
        > LIFT_SUCCESS_THRESHOLD
    )


    result = {
        "grasp_z": grasp_z,
        "open_separation": open_separation,
        "closed_separation": closed_separation,
        "closing_amount": closing_amount,
        "descent_displacement": descent_displacement,
        "cube_close_motion": cube_close_motion,
        "vertical_lift": vertical_lift,
        "total_cube_motion": total_cube_motion,
        "success": success,
    }


    results.append(
        result
    )


    print()
    print(
        "=========================================="
    )

    print(
        f"RESULT — GRASP Z={grasp_z:.3f}"
    )

    print(
        "=========================================="
    )


    print(
        "Open separation:",
        f"{open_separation:.4f} m"
    )

    print(
        "Closed separation:",
        f"{closed_separation:.4f} m"
    )

    print(
        "Closing amount:",
        f"{closing_amount:.4f} m"
    )


    print()


    print(
        "Cube before lift:",
        np.round(
            cube_before_lift,
            4,
        )
    )

    print(
        "Cube after lift:",
        np.round(
            cube_after_lift,
            4,
        )
    )


    print()


    print(
        "Vertical lift:",
        f"{vertical_lift:.4f} m"
    )

    print(
        "Total cube motion:",
        f"{total_cube_motion:.4f} m"
    )


    print()


    print(
        "GRASP SUCCESS:",
        success
    )


    print(
        "=========================================="
    )


    # ========================================================
    # STOP IF SUCCESS
    # ========================================================

    if success:

        working_grasp_z = (
            grasp_z
        )

        print()
        print(
            "******************************************"
        )

        print(
            "WORKING GRASP HEIGHT FOUND"
        )

        print(
            f"Z = {working_grasp_z:.4f} m"
        )

        print(
            "******************************************"
        )

        break


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print(
    "============================================================"
)

print(
    "FINAL CALIBRATION SUMMARY"
)

print(
    "============================================================"
)


for result in results:

    print()

    print(
        f"Grasp Z = "
        f"{result['grasp_z']:.3f}"
    )

    print(
        f"  Open separation:   "
        f"{result['open_separation']:.4f} m"
    )

    print(
        f"  Closed separation: "
        f"{result['closed_separation']:.4f} m"
    )

    print(
        f"  Closing amount:    "
        f"{result['closing_amount']:.4f} m"
    )

    print(
        f"  Descent cube move: "
        f"{result['descent_displacement']:.4f} m"
    )

    print(
        f"  Close cube move:   "
        f"{result['cube_close_motion']:.4f} m"
    )

    print(
        f"  Vertical lift:     "
        f"{result['vertical_lift']:.4f} m"
    )

    print(
        f"  SUCCESS:           "
        f"{result['success']}"
    )


print()
print(
    "============================================================"
)


if working_grasp_z is not None:

    print(
        "CALIBRATION SUCCESS"
    )

    print(
        f"Working grasp target Z: "
        f"{working_grasp_z:.4f} m"
    )

else:

    print(
        "NO WORKING GRASP HEIGHT FOUND"
    )

    print()

    print(
        "The next diagnostic is the CLOSED finger separation."
    )

    print(
        "If it becomes approximately 0 m, "
        "the fingers are closing without the cube between them."
    )

    print(
        "If it remains near the cube width (~0.05 m), "
        "the fingers are contacting the cube but the grasp "
        "is not strong/stable enough."
    )


print(
    "============================================================"
)
print()


# ============================================================
# RETURN SAFE
# ============================================================

move_to(
    safe_position,
    steps=400,
)


# ============================================================
# SHUTDOWN
# ============================================================

simulation_app.close()