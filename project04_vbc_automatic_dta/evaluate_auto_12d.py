
"""Closed-loop evaluation of a 12D Vision BC policy in Isaac Sim 6.1."""

import argparse
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
from vbc_model import VisionBehaviorCloning

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="vision_bc_12d_auto_best.pt")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--duration", type=float, default=40.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--log-every", type=int, default=10)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720
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
        "isaacsim.robot.manipulators.examples", True
    )
    if not manager.is_extension_enabled(
        "isaacsim.robot.manipulators.examples"
    ):
        raise RuntimeError("Could not enable Franka examples extension")

    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController

    # Configuration
    CUBE_SIZE = 0.05
    GOAL = np.array([0.50, 0.25, 0.025], dtype=np.float64)
    INITIAL_TARGET = np.array(
        [0.4992, 0.0400, 0.2400],
        dtype=np.float64
    )
    LIMITS_MIN = np.array([0.25, -0.35, 0.035], dtype=np.float64)
    LIMITS_MAX = np.array([0.75, 0.45, 0.50], dtype=np.float64)
    CONTROL_HZ = 10.0
    XY_THRESHOLD = 0.05
    Z_THRESHOLD = 0.05
    LIFT_THRESHOLD = 0.07
    RELEASE_STABLE_STEPS = 30
    rng = np.random.default_rng(args.seed)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if int(checkpoint.get("state_dim", 12)) != 12:
        raise ValueError("Checkpoint must use 12D robot state")

    position_scale = float(checkpoint["position_scale"])
    model = VisionBehaviorCloning(
        state_dim=12, pretrained=False, freeze_backbone=True
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    transform = ResNet18_Weights.DEFAULT.transforms()

    print(
        f"[MODEL] {args.checkpoint} device={device} "
        f"scale={position_scale} epoch={checkpoint.get('epoch', 'unknown')}",
        flush=True
    )

    # Scene
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(
        Franka(prim_path="/World/Franka", name="franka")
    )
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=np.array([0.50, 0.00, 0.025]),
            scale=np.array([CUBE_SIZE] * 3),
            color=np.array([1.0, 0.65, 0.0]),
            mass=0.05
        )
    )
    world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            position=np.array([GOAL[0], GOAL[1], 0.002]),
            scale=np.array([0.08, 0.08, 0.004]),
            color=np.array([0.0, 1.0, 0.0])
        )
    )
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

    for _ in range(30):
        world.step(render=True)

    if not args.headless:
        set_camera_view(
            eye=np.array([1.5, 1.5, 1.2]),
            target=np.array([0.45, 0.15, 0.1]),
            camera_prim_path="/OmniverseKit_Persp"
        )

    controller = RMPFlowController(
        name="auto_eval_rmpflow",
        robot_articulation=franka
    )
    orientation = euler_angles_to_quat(
        np.array([0.0, np.pi, 0.0])
    )
    physics_dt = world.get_physics_dt()
    control_interval = max(1, round(1.0 / (CONTROL_HZ * physics_dt)))
    steps_per_episode = max(1, round(args.duration / physics_dt))

    print(
        f"[SIM] physics_dt={physics_dt:.5f} "
        f"control_interval={control_interval} "
        f"effective_hz={1/(control_interval*physics_dt):.2f}",
        flush=True
    )

    def get_policy_input():
        rgb = camera.get_rgb()
        if rgb is None:
            raise RuntimeError("Camera returned None")

        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[:2] != (224, 224) or rgb.shape[2] < 3:
            raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")

        rgb = rgb[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

        image = transform(
            torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        ).unsqueeze(0).to(device)

        joints = np.asarray(
            franka.get_joint_positions(), dtype=np.float32
        )
        ee_pos, _ = franka.end_effector.get_world_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float32)

        state = np.concatenate((joints[:9], ee_pos)).astype(np.float32)
        if state.shape != (12,):
            raise RuntimeError(f"Expected 12D state, got {state.shape}")

        return (
            image,
            torch.from_numpy(state).unsqueeze(0).to(device),
            ee_pos
        )

    def get_cube_metrics():
        pos, _ = cube.get_world_pose()
        pos = np.asarray(pos, dtype=np.float64)
        xy = float(np.linalg.norm(pos[:2] - GOAL[:2]))
        z_error = float(abs(pos[2] - GOAL[2]))
        return pos, xy, z_error

    results = []

    for ep in range(args.episodes):
        if not simulation_app.is_running():
            break

        cube_start = np.array([
            rng.uniform(0.44, 0.56),
            rng.uniform(-0.08, 0.08),
            CUBE_SIZE / 2
        ], dtype=np.float64)

        world.reset()
        controller.reset()
        cube.set_world_pose(position=cube_start.copy())
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        franka.gripper.open()

        gripper_open = True
        target = INITIAL_TARGET.copy()
        ever_closed = False
        ever_lifted = False
        ever_reopened = False
        release_stable = 0
        success = False
        policy_steps = 0
        grip_prob = 1.0
        delta = np.zeros(3)

        print(
            f"\n[EPISODE {ep+1}/{args.episodes}] "
            f"cube_start={np.round(cube_start, 4)}",
            flush=True
        )

        # Same initial EE target as the original evaluator
        for init_step in range(240):
            action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation
            )
            franka.apply_action(action)
            world.step(render=True)

        ee_init, _ = franka.end_effector.get_world_pose()
        print(
            f"[INIT] EE={np.round(ee_init, 3)} "
            f"error={np.linalg.norm(np.asarray(ee_init)-INITIAL_TARGET):.3f}m",
            flush=True
        )

        for step in range(steps_per_episode):
            if not simulation_app.is_running():
                break
            if not world.is_playing():
                world.step(render=True)
                continue

            if step % control_interval == 0:
                image, state, ee_pos = get_policy_input()

                with torch.inference_mode():
                    pred_delta, pred_grip = model(image, state)
                    delta = (
                        pred_delta[0].cpu().numpy() / position_scale
                    )
                    grip_prob = float(
                        torch.sigmoid(pred_grip[0, 0]).item()
                    )

                if not np.isfinite(delta).all() or not np.isfinite(grip_prob):
                    raise RuntimeError("Nonfinite model output")

                delta = np.clip(delta, -0.025, 0.025)
                target = np.clip(
                    target + delta, LIMITS_MIN, LIMITS_MAX
                )

                # Fully model-controlled gripper
                next_gripper_open = grip_prob >= 0.5
                if next_gripper_open != gripper_open:
                    print(
                        f"[GRIPPER] step={policy_steps} "
                        f"{'OPEN' if gripper_open else 'CLOSED'} -> "
                        f"{'OPEN' if next_gripper_open else 'CLOSED'} "
                        f"prob={grip_prob:.3f}",
                        flush=True
                    )
                    if next_gripper_open:
                        franka.gripper.open()
                        if ever_closed:
                            ever_reopened = True
                    else:
                        franka.gripper.close()
                        ever_closed = True
                    gripper_open = next_gripper_open

                cube_pos, xy, z_error = get_cube_metrics()
                if cube_pos[2] > LIFT_THRESHOLD:
                    ever_lifted = True

                # Success requires a lifted cube, a close/open cycle,
                # and a stable released cube near the goal.
                placed = (
                    ever_closed
                    and ever_lifted
                    and ever_reopened
                    and gripper_open
                    and xy < XY_THRESHOLD
                    and z_error < Z_THRESHOLD
                )

                if placed:
                    release_stable += 1
                else:
                    release_stable = 0

                if policy_steps % max(1, args.log_every) == 0:
                    print(
                        f"[POLICY {policy_steps:04d}] "
                        f"delta={np.round(delta, 4)} "
                        f"target={np.round(target, 3)} "
                        f"ee={np.round(ee_pos, 3)} "
                        f"cube={np.round(cube_pos, 3)} "
                        f"grip_prob={grip_prob:.3f} "
                        f"gripper={'OPEN' if gripper_open else 'CLOSED'} "
                        f"xy={xy:.3f} z_err={z_error:.3f} "
                        f"lifted={ever_lifted}",
                        flush=True
                    )

                policy_steps += 1

                if release_stable >= RELEASE_STABLE_STEPS:
                    success = True
                    print(
                        f"[SUCCESS] Stable placement at step={policy_steps}",
                        flush=True
                    )
                    break

            action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation
            )
            franka.apply_action(action)
            world.step(render=True)

        cube_pos, final_xy, final_z = get_cube_metrics()

        results.append({
            "success": success,
            "xy": final_xy,
            "z": final_z,
            "closed": ever_closed,
            "lifted": ever_lifted,
            "reopened": ever_reopened
        })

        print(
            f"[RESULT] ep={ep+1} success={success} "
            f"xy={final_xy:.3f} z_err={final_z:.3f} "
            f"closed={ever_closed} lifted={ever_lifted} "
            f"reopened={ever_reopened} "
            f"policy_steps={policy_steps}",
            flush=True
        )

    if results:
        n = len(results)
        print("\n========== EVALUATION SUMMARY ==========", flush=True)
        print(f"Episodes: {n}", flush=True)
        print(
            f"Closed gripper: {sum(r['closed'] for r in results)}/{n}",
            flush=True
        )
        print(
            f"Lifted cube: {sum(r['lifted'] for r in results)}/{n}",
            flush=True
        )
        print(
            f"Reopened gripper: {sum(r['reopened'] for r in results)}/{n}",
            flush=True
        )
        print(
            f"Placement success: {sum(r['success'] for r in results)}/{n} "
            f"({np.mean([r['success'] for r in results]):.1%})",
            flush=True
        )
        print(
            f"Mean final XY error: "
            f"{np.mean([r['xy'] for r in results]):.3f}m",
            flush=True
        )
        print(
            "[NOTE] Gripper release is inferred from commands and cube pose, "
            "not measured contact forces.",
            flush=True
        )

finally:
    simulation_app.close()
