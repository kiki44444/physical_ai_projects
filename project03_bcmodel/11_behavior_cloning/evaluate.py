
"""
Evaluate Behavior Cloning on Franka Panda
Isaac Sim 6.1

Example:
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 \
python evaluate.py --episodes 5 --duration 60
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--duration", type=float, default=60.0)
parser.add_argument("--checkpoint", type=str, default="bc_best.pt")
parser.add_argument("--headless", action="store_true")

args = parser.parse_args()

# --------------------------------------------------
# 1. Start Isaac Sim
# --------------------------------------------------

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import (
        DynamicCuboid,
        VisualCuboid,
    )
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.viewports import set_camera_view

    import omni.kit.app

    # Enable manipulator examples extension
    manager = omni.kit.app.get_app().get_extension_manager()

    manager.set_extension_enabled_immediate(
        "isaacsim.robot.manipulators.examples",
        True,
    )

    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController,
    )

    from model import BehaviorCloning

    # --------------------------------------------------
    # 2. Configuration
    # --------------------------------------------------

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    CUBE_SIZE = 0.05

    CUBE_START = np.array(
        [0.50, 0.00, CUBE_SIZE / 2],
        dtype=np.float64,
    )

    GOAL = np.array(
        [0.50, 0.25, CUBE_SIZE / 2],
        dtype=np.float64,
    )

    INITIAL_TARGET = np.array(
        [0.50, 0.00, 0.20],
        dtype=np.float64,
    )

    LIMITS_MIN = np.array(
        [0.25, -0.35, 0.035],
        dtype=np.float64,
    )

    LIMITS_MAX = np.array(
        [0.75, 0.45, 0.50],
        dtype=np.float64,
    )

    SUCCESS_THRESHOLD = 0.05

    # Match recorder action clipping
    MAX_DELTA = 0.025

    # Evaluation control frequency
    CONTROL_HZ = 10.0
    CONTROL_DT = 1.0 / CONTROL_HZ

    # --------------------------------------------------
    # 3. Load trained model
    # --------------------------------------------------

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    obs_dim = int(checkpoint.get("obs_dim", 18))
    position_scale = float(
        checkpoint.get("position_scale", 100.0)
    )

    model = BehaviorCloning(
        obs_dim=obs_dim,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("[OK] Model loaded:", checkpoint_path)
    print("[INFO] Device:", DEVICE)
    print("[INFO] Obs dim:", obs_dim)
    print("[INFO] Position scale:", position_scale)

    # --------------------------------------------------
    # 4. Create simulation environment
    # --------------------------------------------------

    world = World(
        stage_units_in_meters=1.0
    )

    world.scene.add_default_ground_plane()

    franka = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka",
        )
    )

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=CUBE_START.copy(),
            scale=np.array([CUBE_SIZE] * 3),
            color=np.array([1.0, 0.65, 0.0]),
            mass=0.05,
        )
    )

    goal_marker = world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            position=np.array([
                GOAL[0],
                GOAL[1],
                0.002,
            ]),
            scale=np.array([0.08, 0.08, 0.004]),
            color=np.array([0.0, 1.0, 0.0]),
        )
    )

    world.reset()

    if not args.headless:
        set_camera_view(
            eye=np.array([1.5, 1.5, 1.2]),
            target=np.array([0.45, 0.15, 0.1]),
            camera_prim_path="/OmniverseKit_Persp",
        )

    controller = RMPFlowController(
        name="evaluation_rmpflow",
        robot_articulation=franka,
    )

    orientation = euler_angles_to_quat(
        np.array([0.0, np.pi, 0.0])
    )

    # --------------------------------------------------
    # 5. Observation
    # --------------------------------------------------

    def get_observation():
        joints = np.asarray(
            franka.get_joint_positions(),
            dtype=np.float32,
        )

        ee_pos, _ = franka.end_effector.get_world_pose()
        cube_pos, _ = cube.get_world_pose()

        obs = np.concatenate([
            joints,
            np.asarray(ee_pos, dtype=np.float32),
            np.asarray(cube_pos, dtype=np.float32),
            GOAL.astype(np.float32),
        ])

        if obs.shape != (obs_dim,):
            raise ValueError(
                f"Observation mismatch: "
                f"expected ({obs_dim},), got {obs.shape}"
            )

        return obs

    # --------------------------------------------------
    # 6. Reset
    # --------------------------------------------------

    def reset_episode():
        world.reset()
        controller.reset()

        cube.set_world_pose(
            position=CUBE_START.copy()
        )

        cube.set_linear_velocity(
            np.zeros(3)
        )

        cube.set_angular_velocity(
            np.zeros(3)
        )

        franka.gripper.open()

        for _ in range(30):
            world.step(render=not args.headless)

        return INITIAL_TARGET.copy(), True

    # --------------------------------------------------
    # 7. Evaluation loop
    # --------------------------------------------------

    results = []

    physics_dt = world.get_physics_dt()

    control_interval = max(
        1,
        round(CONTROL_DT / physics_dt),
    )

    print(
        f"[INFO] Physics dt: {physics_dt:.5f}, "
        f"Control interval: {control_interval} steps"
    )

    for episode_idx in range(args.episodes):

        if not simulation_app.is_running():
            break

        print(
            f"\n========== Episode "
            f"{episode_idx + 1}/{args.episodes} ==========",
            flush=True,
        )

        target, gripper_open = reset_episode()

        max_steps = int(
            args.duration / physics_dt
        )

        last_delta = np.zeros(3)

        episode_start = time.time()

        for step in range(max_steps):

            if not simulation_app.is_running():
                break

            if not world.is_playing():
                world.step(render=not args.headless)
                continue

            # Predict at control frequency
            if step % control_interval == 0:

                obs = get_observation()

                obs_tensor = torch.from_numpy(
                    obs
                ).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    position_pred, gripper_logits = model(
                        obs_tensor
                    )

                delta = (
                    position_pred[0]
                    .cpu()
                    .numpy()
                    / position_scale
                )

                delta = np.clip(
                    delta,
                    -MAX_DELTA,
                    MAX_DELTA,
                )

                last_delta = delta.copy()

                target = np.clip(
                    target + delta,
                    LIMITS_MIN,
                    LIMITS_MAX,
                )

                open_probability = torch.sigmoid(
                    gripper_logits
                ).item()

                desired_gripper_open = (
                    open_probability >= 0.5
                )

                if desired_gripper_open != gripper_open:
                    gripper_open = desired_gripper_open

                    if gripper_open:
                        franka.gripper.open()
                    else:
                        franka.gripper.close()

            # Apply controller every physics step
            action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation,
            )

            franka.apply_action(action)

            world.step(render=not args.headless)

            if step % (control_interval * 10) == 0:
                cube_pos, _ = cube.get_world_pose()

                distance = np.linalg.norm(
                    np.asarray(cube_pos[:2]) - GOAL[:2]
                )

                print(
                    f"[STEP {step}] "
                    f"delta={np.round(last_delta, 4)} "
                    f"target={np.round(target, 3)} "
                    f"cube_distance={distance:.3f}",
                    flush=True,
                )

        cube_pos, _ = cube.get_world_pose()

        final_distance = float(
            np.linalg.norm(
                np.asarray(cube_pos[:2]) - GOAL[:2]
            )
        )

        success = final_distance < SUCCESS_THRESHOLD

        results.append({
            "episode": episode_idx + 1,
            "success": success,
            "final_distance": final_distance,
        })

        print(
            f"[RESULT] Episode {episode_idx + 1} | "
            f"Success: {success} | "
            f"Distance: {final_distance:.3f} m | "
            f"Wall time: {time.time() - episode_start:.1f}s",
            flush=True,
        )

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    print("\n========== EVALUATION SUMMARY ==========")

    if results:
        success_rate = np.mean([
            r["success"] for r in results
        ])

        mean_distance = np.mean([
            r["final_distance"] for r in results
        ])

        print(f"Episodes: {len(results)}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Mean final distance: {mean_distance:.3f} m")
    else:
        print("No episodes completed")

finally:
    simulation_app.close()
