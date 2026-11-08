
"""
Closed-loop evaluation of Vision Behavior Cloning in Isaac Sim 6.1.

Run:
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 \
python -u evaluate_vision.py \
    --checkpoint vision_bc_12d_place_v2_ep05.pt \
    --episodes 1 \
    --duration 30 \
    --log-every 5
"""

import argparse
import numpy as np
import torch

from torchvision.models import ResNet18_Weights
from vbc_model import VisionBehaviorCloning


# ============================================================
# 1. Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    default="vision_bc_12d_place_v2_ep05.pt"
)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--log-every", type=int, default=10)

args = parser.parse_args()


# ============================================================
# 2. Isaac Sim
# ============================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

try:
    import omni.kit.app

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera

    manager = omni.kit.app.get_app().get_extension_manager()

    manager.set_extension_enabled_immediate(
        "isaacsim.robot.manipulators.examples",
        True
    )

    if not manager.is_extension_enabled(
        "isaacsim.robot.manipulators.examples"
    ):
        raise RuntimeError("Could not enable Franka extension")

    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController
    )

    # ========================================================
    # 3. Configuration
    # ========================================================

    CUBE_SIZE = 0.05

    CUBE_START = np.array(
        [0.50, 0.00, CUBE_SIZE / 2],
        dtype=np.float64
    )

    GOAL = np.array(
        [0.50, 0.25, CUBE_SIZE / 2],
        dtype=np.float64
    )

    INITIAL_TARGET = np.array(
        [0.50, 0.00, 0.20],
        dtype=np.float64
    )

    LIMITS_MIN = np.array(
        [0.25, -0.35, 0.035],
        dtype=np.float64
    )

    LIMITS_MAX = np.array(
        [0.75, 0.45, 0.50],
        dtype=np.float64
    )

    SUCCESS_THRESHOLD = 0.05
    CONTROL_HZ = 10.0

    # ========================================================
    # 4. Load model
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )

    position_scale = float(
        checkpoint["position_scale"]
    )

    model = VisionBehaviorCloning(
        state_dim=12,
        pretrained=False,
        freeze_backbone=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    transform = ResNet18_Weights.DEFAULT.transforms()

    print(
        f"[MODEL] loaded {args.checkpoint} "
        f"device={device} "
        f"position_scale={position_scale}",
        flush=True
    )

    # ========================================================
    # 5. World setup
    # ========================================================

    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()

    franka = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka"
        )
    )

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=CUBE_START.copy(),
            scale=np.array([CUBE_SIZE] * 3),
            color=np.array([1.0, 0.65, 0.0]),
            mass=0.05
        )
    )

    world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            position=np.array([
                GOAL[0],
                GOAL[1],
                0.002
            ]),
            scale=np.array([0.08, 0.08, 0.004]),
            color=np.array([0.0, 1.0, 0.0])
        )
    )

    # ========================================================
    # 6. Camera
    # ========================================================

    camera = Camera(
        prim_path="/World/RecorderCamera",
        name="recorder_camera",
        position=np.array([1.2, 0.8, 1.0]),
        resolution=(224, 224)
    )

    world.reset()

    set_camera_view(
        eye=np.array([1.2, 0.8, 1.0]),
        target=np.array([0.50, 0.12, 0.08]),
        camera_prim_path="/World/RecorderCamera"
    )

    camera.initialize()
    camera.add_rgb_to_frame()

    print("[CAMERA] Starting warmup", flush=True)

    for i in range(30):
        world.step(render=True)

    print("[CAMERA] Warmup completed", flush=True)

    set_camera_view(
        eye=np.array([1.5, 1.5, 1.2]),
        target=np.array([0.45, 0.15, 0.1]),
        camera_prim_path="/OmniverseKit_Persp"
    )

    # ========================================================
    # 7. RMPFlow Controller
    # ========================================================

    controller = RMPFlowController(
        name="vision_eval_rmpflow",
        robot_articulation=franka
    )

    orientation = euler_angles_to_quat(
        np.array([0.0, np.pi, 0.0])
    )

    physics_dt = world.get_physics_dt()

    control_interval = max(
        1,
        round(1.0 / (CONTROL_HZ * physics_dt))
    )

    steps_per_episode = max(
        1,
        round(args.duration / physics_dt)
    )

    print(
        f"[SIM] physics_dt={physics_dt:.5f} "
        f"control_interval={control_interval} "
        f"effective_hz="
        f"{1/(control_interval*physics_dt):.2f}",
        flush=True
    )

    # ========================================================
    # 8. Policy observation
    # ========================================================

    def get_policy_input():
        rgb = camera.get_rgb()

        if rgb is None:
            raise RuntimeError("Camera returned None")

        rgb = np.asarray(rgb)

        if (
            rgb.shape[:2] != (224, 224)
            or rgb.ndim != 3
            or rgb.shape[2] < 3
        ):
            raise RuntimeError(
                f"Unexpected RGB shape: {rgb.shape}"
            )

        rgb = rgb[:, :, :3]

        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (
                np.clip(rgb, 0.0, 1.0) * 255
            ).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

        image = transform(
            torch.from_numpy(
                rgb.copy()
            ).permute(2, 0, 1)
        ).unsqueeze(0).to(device)

        joints = np.asarray(
            franka.get_joint_positions(),
            dtype=np.float32
        )

        ee_pos, _ = franka.end_effector.get_world_pose()

        ee_pos = np.asarray(
            ee_pos,
            dtype=np.float32
        )

        state = np.concatenate(
            (
                joints[:9],
                ee_pos
            )
        ).astype(np.float32)

        assert state.shape == (12,)

        state_tensor = torch.from_numpy(
            state
        ).unsqueeze(0).to(device)

        return image, state_tensor, ee_pos

    # ========================================================
    # 9. Closed-loop evaluation
    # ========================================================

    results = []

    for ep in range(args.episodes):

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
        gripper_open = True

        target = INITIAL_TARGET.copy()

        print(
            f"\n[EPISODE {ep+1}/{args.episodes}]",
            flush=True
        )

        # ----------------------------------------------------
        # Initialize robot pose
        # ----------------------------------------------------

        print(
            "[INIT] Moving EE to initial target...",
            flush=True
        )

        for init_step in range(240):

            action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation
            )

            franka.apply_action(action)

            world.step(render=True)

            if init_step % 60 == 0:

                ee_init, _ = (
                    franka.end_effector.get_world_pose()
                )

                print(
                    f"[INIT {init_step}] "
                    f"EE={np.round(ee_init, 3)} "
                    f"target={np.round(target, 3)}",
                    flush=True
                )

        ee_init, _ = franka.end_effector.get_world_pose()

        ee_init = np.asarray(
            ee_init,
            dtype=np.float64
        )

        initial_error = ee_init - INITIAL_TARGET

        print(
            f"[INIT DONE] "
            f"EE={np.round(ee_init, 4)} "
            f"target={np.round(INITIAL_TARGET, 4)} "
            f"error={np.round(initial_error, 4)} "
            f"error_norm="
            f"{np.linalg.norm(initial_error):.4f}m",
            flush=True
        )

        # ----------------------------------------------------
        # Policy rollout
        # ----------------------------------------------------

        policy_steps = 0

        delta = np.zeros(3)
        grip_prob = 1.0

        for step in range(steps_per_episode):

            if not simulation_app.is_running():
                break

            if not world.is_playing():
                world.step(render=True)
                continue

            if step % control_interval == 0:

                image, state, ee_pos = get_policy_input()

                # --------------------------------------------
                # Model inference
                # --------------------------------------------

                with torch.inference_mode():

                    pred_delta, pred_grip = model(
                        image,
                        state
                    )

                    delta = (
                        pred_delta[0]
                        .detach()
                        .cpu()
                        .numpy()
                        / position_scale
                    )

                    grip_prob = float(
                        torch.sigmoid(
                            pred_grip[0, 0]
                        ).item()
                    )

                if (
                    not np.isfinite(delta).all()
                    or not np.isfinite(grip_prob)
                ):
                    raise RuntimeError(
                        "Nonfinite policy output"
                    )

                # --------------------------------------------
                # Position action
                # --------------------------------------------

                delta = np.clip(
                    delta,
                    -0.025,
                    0.025
                )

                target = np.clip(
                    target + delta,
                    LIMITS_MIN,
                    LIMITS_MAX
                )

                # --------------------------------------------
                # Gripper action: MODEL CONTROLLED
                # No forced CLOSE / no has_grasped lock
                # --------------------------------------------

                next_gripper_open = grip_prob >= 0.5

                if next_gripper_open != gripper_open:

                    print(
                        f"[GRIPPER TRANSITION] "
                        f"step={policy_steps} "
                        f"from="
                        f"{'OPEN' if gripper_open else 'CLOSED'} "
                        f"to="
                        f"{'OPEN' if next_gripper_open else 'CLOSED'} "
                        f"prob={grip_prob:.4f}",
                        flush=True
                    )

                    if next_gripper_open:
                        franka.gripper.open()
                    else:
                        franka.gripper.close()

                    gripper_open = next_gripper_open

                # --------------------------------------------
                # Logging
                # --------------------------------------------

                if (
                    policy_steps
                    % max(1, args.log_every)
                    == 0
                ):

                    cube_pos, _ = cube.get_world_pose()

                    dist = float(
                        np.linalg.norm(
                            np.asarray(cube_pos[:2])
                            - GOAL[:2]
                        )
                    )

                    print(
                        f"[POLICY {policy_steps:04d}] "
                        f"delta={np.round(delta, 4)} "
                        f"target={np.round(target, 3)} "
                        f"ee={np.round(ee_pos, 3)} "
                        f"cube={np.round(cube_pos, 3)} "
                        f"grip_open_prob={grip_prob:.3f} "
                        f"gripper="
                        f"{'OPEN' if gripper_open else 'CLOSED'} "
                        f"cube_goal_xy={dist:.3f}",
                        flush=True
                    )

                policy_steps += 1

            # -----------------------------------------------
            # Apply RMPFlow action at physics frequency
            # -----------------------------------------------

            action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation
            )

            franka.apply_action(action)

            world.step(render=True)

        # ----------------------------------------------------
        # Episode result
        # ----------------------------------------------------

        cube_pos, _ = cube.get_world_pose()

        final_dist = float(
            np.linalg.norm(
                np.asarray(cube_pos[:2])
                - GOAL[:2]
            )
        )

        success = final_dist < SUCCESS_THRESHOLD

        results.append(
            (success, final_dist)
        )

        print(
            f"[RESULT] "
            f"success_xy={success} "
            f"final_distance={final_dist:.3f}m "
            f"policy_steps={policy_steps}",
            flush=True
        )

    # ========================================================
    # 10. Summary
    # ========================================================

    if results:

        success_rate = np.mean(
            [x[0] for x in results]
        )

        mean_distance = np.mean(
            [x[1] for x in results]
        )

        print(
            f"\n[SUMMARY] "
            f"success_rate={success_rate:.1%} "
            f"mean_final_xy_distance="
            f"{mean_distance:.3f}m",
            flush=True
        )

        print(
            "[NOTE] XY proximity alone does not prove "
            "successful grasp-and-place.",
            flush=True
        )

        print(
            "[NOTE] This checkpoint was trained on "
            "one demonstration episode.",
            flush=True
        )

finally:
    simulation_app.close()
