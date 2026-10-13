"""Isaac Sim 4.0 GUI closed-loop evaluation of state-based BC.

BC controls XYZ movement and gripper commands.
A safety guard prevents descent when XY is misaligned and
limits approach speed near the cube.

Run:
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 \
/isaac-sim/python.sh evaluate.py --episodes 1 --duration 60
"""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bc_best.pt")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--duration", type=float, default=60.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

import torch

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view

from model import BehaviorCloning


# ============================================================
# 1. CONFIGURATION
# ============================================================

CUBE_SIZE = 0.05

GOAL = np.array(
    [0.40, 0.30, CUBE_SIZE / 2],
    dtype=np.float64,
)

START_TARGET = np.array(
    [0.50, 0.00, 0.20],
    dtype=np.float64,
)

MIN_TARGET = np.array(
    [0.25, -0.35, 0.035],
    dtype=np.float64,
)

MAX_TARGET = np.array(
    [0.75, 0.45, 0.50],
    dtype=np.float64,
)

CONTROL_PERIOD = 0.1

SUCCESS_XY_TOL = 0.05
SUCCESS_Z_TOL = 0.03


# ============================================================
# 2. COLLISION GUARD
# ============================================================

# Maximum XY alignment error before descent is allowed.
ALIGNMENT_XY_TOL = 0.01

# Maximum XY correction per policy step.
MAX_XY_CORRECTION = 0.005

# Maximum downward target change per policy step.
MAX_DOWNWARD_DELTA = 0.002

# Slow down when the finger midpoint approaches the cube.
SLOWDOWN_HEIGHT = 0.12
SLOW_DOWNWARD_DELTA = 0.0005

# Experimental lower approach boundary measured from
# finger-frame midpoint to cube center.
#
# IMPORTANT:
# Finger-frame origins are NOT physical fingertip surfaces.
# This is a conservative test boundary, not a calibrated
# collision-free grasp height.
MIN_FINGER_HEIGHT_ABOVE_CUBE = 0.055

# Maximum allowed cube displacement during approach.
# Stop if the cube is unexpectedly pushed.
MAX_CUBE_DISPLACEMENT = 0.015


# ============================================================
# 3. LOAD MODEL
# ============================================================

rng = np.random.default_rng(args.seed)

model_path = Path(args.model).resolve()

if not model_path.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path}"
    )

checkpoint = torch.load(
    model_path,
    map_location="cpu",
)

if checkpoint.get("obs_dim", 18) != 18:
    raise ValueError(
        "Expected 18-dimensional observations"
    )

position_scale = float(
    checkpoint.get("position_scale", 100.0)
)

model = BehaviorCloning(obs_dim=18)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Loaded BC model: {model_path}",
    flush=True,
)


# ============================================================
# 4. CREATE WORLD
# ============================================================

world = World(stage_units_in_meters=1.0)

world.scene.add_default_ground_plane()

franka_usd = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.0/Isaac/Robots/Franka/franka.usd"
)

franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        usd_path=franka_usd,
    )
)

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0.5, 0.0, 0.025]),
        scale=np.array([CUBE_SIZE] * 3),
        mass=0.05,
    )
)

world.scene.add(
    VisualCuboid(
        prim_path="/World/GoalMarker",
        name="goal_marker",
        position=np.array([
            GOAL[0],
            GOAL[1],
            0.002,
        ]),
        scale=np.array([0.065, 0.065, 0.004]),
        color=np.array([0.0, 0.9, 0.2]),
    )
)

world.reset()

left_finger = XFormPrim(
    prim_path="/World/Franka/panda_leftfinger",
    name="left_finger_debug",
)

if not args.headless:
    set_camera_view(
        eye=np.array([1.5, 1.5, 1.2]),
        target=np.array([0.45, 0.15, 0.1]),
        camera_prim_path="/OmniverseKit_Persp",
    )

controller = RMPFlowController(
    name="bc_rmpflow",
    robot_articulation=franka,
)

orientation = euler_angles_to_quat(
    np.array([0.0, np.pi, 0.0])
)


# ============================================================
# 5. OBSERVATION
# ============================================================

def observe():

    ee_position, _ = franka.end_effector.get_world_pose()
    cube_position, _ = cube.get_world_pose()

    obs = np.concatenate([
        np.asarray(
            franka.get_joint_positions(),
            dtype=np.float64,
        ),
        np.asarray(
            ee_position,
            dtype=np.float64,
        ),
        np.asarray(
            cube_position,
            dtype=np.float64,
        ),
        GOAL,
    ])

    return obs.astype(np.float32)


# ============================================================
# 6. FINGER ALIGNMENT
# ============================================================

def finger_alignment():

    right_pos, _ = franka.end_effector.get_world_pose()
    left_pos, _ = left_finger.get_world_pose()
    cube_pos, _ = cube.get_world_pose()

    right_pos = np.asarray(right_pos, dtype=np.float64)
    left_pos = np.asarray(left_pos, dtype=np.float64)
    cube_pos = np.asarray(cube_pos, dtype=np.float64)

    midpoint = (left_pos + right_pos) / 2.0

    xy_vector = cube_pos[:2] - midpoint[:2]

    xy_error = float(
        np.linalg.norm(xy_vector)
    )

    height_above_cube = float(
        midpoint[2] - cube_pos[2]
    )

    return (
        midpoint,
        cube_pos,
        xy_vector,
        xy_error,
        height_above_cube,
    )


# ============================================================
# 7. RESET EPISODE
# ============================================================

def reset_episode():

    world.reset()

    cube_start = np.array([
        rng.uniform(0.44, 0.56),
        rng.uniform(-0.08, 0.08),
        CUBE_SIZE / 2,
    ])

    cube.set_world_pose(
        position=cube_start,
        orientation=np.array([
            1.0, 0.0, 0.0, 0.0
        ]),
    )

    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))

    controller.reset()

    franka.gripper.open()

    for _ in range(60):
        world.step(render=not args.headless)

    return cube_start


# ============================================================
# 8. EVALUATION
# ============================================================

success_count = 0

try:

    for episode in range(args.episodes):

        start = reset_episode()
        
        target = START_TARGET.copy()

        gripper_open = True

        next_control_time = world.current_time

        end_time = (
            world.current_time + args.duration
        )

        action_count = 0
        stopped_for_safety = False

        print(
            f"\nEpisode {episode + 1}/{args.episodes} "
            f"start={start.round(3)}",
            flush=True,
        )

        while (
            simulation_app.is_running()
            and world.current_time < end_time
        ):

            # ------------------------------------------------
            # POLICY UPDATE
            # ------------------------------------------------

            if (
                world.current_time
                >= next_control_time - 1e-5
            ):

                obs = observe()

                if (
                    obs.shape != (18,)
                    or not np.isfinite(obs).all()
                ):
                    raise ValueError(
                        f"Invalid observation: {obs}"
                    )

                with torch.no_grad():

                    position_pred, gripper_logits = model(
                        torch.from_numpy(obs).unsqueeze(0)
                    )

                bc_delta = (
                    position_pred.squeeze(0).numpy()
                    / position_scale
                )

                delta = np.clip(
                    bc_delta.copy(),
                    -0.02,
                    0.02,
                )

                (
                    midpoint,
                    cube_pos,
                    xy_vector,
                    xy_error,
                    height_above_cube,
                ) = finger_alignment()

                # --------------------------------------------
                # SAFETY: CUBE DISPLACEMENT
                # --------------------------------------------

                cube_displacement = float(
                    np.linalg.norm(
                        cube_pos[:2] - start[:2]
                    )
                )

                if (
                    cube_displacement
                    > MAX_CUBE_DISPLACEMENT
                ):

                    print(
                        "\nSAFETY STOP: "
                        "Cube moved unexpectedly. "
                        f"Displacement="
                        f"{cube_displacement:.3f}m",
                        flush=True,
                    )

                    stopped_for_safety = True
                    break

                # --------------------------------------------
                # SAFETY: XY ALIGNMENT
                # --------------------------------------------

                aligned = (
                    xy_error <= ALIGNMENT_XY_TOL
                )

                if not aligned:

                    # Correct XY to align finger midpoint
                    # with cube center.
                    delta[:2] = np.clip(
                        0.5 * xy_vector,
                        -MAX_XY_CORRECTION,
                        MAX_XY_CORRECTION,
                    )

                    # Block downward movement when
                    # not aligned.
                    delta[2] = max(
                        delta[2],
                        0.0,
                    )

                # --------------------------------------------
                # SAFETY: DESCENT SPEED
                # --------------------------------------------

                delta[2] = max(
                    delta[2],
                    -MAX_DOWNWARD_DELTA,
                )

                if (
                    height_above_cube
                    <= SLOWDOWN_HEIGHT
                ):

                    delta[2] = max(
                        delta[2],
                        -SLOW_DOWNWARD_DELTA,
                    )

                # --------------------------------------------
                # SAFETY: MINIMUM APPROACH HEIGHT
                # --------------------------------------------

                # Estimate the desired target Z from the
                # measured midpoint-to-target offset.
                midpoint_target_offset_z = (
                    midpoint[2] - target[2]
                )

                min_safe_target_z = (
                    cube_pos[2]
                    + MIN_FINGER_HEIGHT_ABOVE_CUBE
                    - midpoint_target_offset_z
                )

                proposed_target = np.clip(
                    target + delta,
                    MIN_TARGET,
                    MAX_TARGET,
                )

                # Prevent the target from descending below
                # the estimated approach boundary.
                proposed_target[2] = max(
                    proposed_target[2],
                    min_safe_target_z,
                )

                # IMPORTANT: update target only once.
                target = proposed_target

                # --------------------------------------------
                # GRIPPER CONTROL
                # --------------------------------------------

                open_probability = float(
                    torch.sigmoid(
                        gripper_logits
                    ).item()
                )

                if (
                    gripper_open
                    and open_probability < 0.35
                ):

                    gripper_open = False
                    franka.gripper.close()

                elif (
                    not gripper_open
                    and open_probability > 0.65
                ):

                    gripper_open = True
                    franka.gripper.open()

                # --------------------------------------------
                # LOGGING
                # --------------------------------------------

                action_count += 1

                if action_count % 50 == 0:
                    right_ee_pos, _ = franka.end_effector.get_world_pose()

                    right_ee_pos = np.asarray(
                        right_ee_pos,
                        dtype=np.float64
                    )

                    ee_cube_relative = right_ee_pos - cube_pos

                    print(
                        f"EE-Cube={np.round(ee_cube_relative, 4)} "
                        f"EE-Cube-XY={np.linalg.norm(ee_cube_relative[:2]):.4f}m "
                        f"EE-Cube-Z={ee_cube_relative[2]:.4f}m"
                    )
                    print(
                        f"t={world.current_time:.1f}s "
                        f"cube={np.round(cube_pos, 3)} "
                        f"midpoint={np.round(midpoint, 3)} "
                        f"xy_error={xy_error:.3f}m "
                        f"height={height_above_cube:.3f}m "
                        f"target={np.round(target, 3)} "
                        f"BC_delta={np.round(bc_delta, 4)} "
                        f"gripper_open={gripper_open} "
                        f"p_open={open_probability:.3f}",
                        flush=True,
                    )
                    

                # CRITICAL: advance control time.
                next_control_time += CONTROL_PERIOD

            # ------------------------------------------------
            # APPLY ROBOT ACTION
            # ------------------------------------------------

            robot_action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation,
            )

            franka.apply_action(robot_action)

            world.step(
                render=not args.headless
            )

        # ----------------------------------------------------
        # EPISODE RESULT
        # ----------------------------------------------------

        cube_final, _ = cube.get_world_pose()

        xy_error = float(
            np.linalg.norm(
                cube_final[:2] - GOAL[:2]
            )
        )

        z_error = float(
            abs(cube_final[2] - GOAL[2])
        )

        success = (
            xy_error <= SUCCESS_XY_TOL
            and z_error <= SUCCESS_Z_TOL
        )

        success_count += int(success)

        print(
            f"Episode {episode + 1}: "
            f"success={success} "
            f"XY_error={xy_error:.3f}m "
            f"Z_error={z_error:.3f}m "
            f"final_cube={np.round(cube_final, 3)} "
            f"safety_stop={stopped_for_safety}",
            flush=True,
        )

    if args.episodes:

        print(
            f"\nPosition success rate: "
            f"{success_count}/{args.episodes} "
            f"({100 * success_count / args.episodes:.1f}%)",
            flush=True,
        )

finally:
    simulation_app.close()