"""Closed-loop Isaac Sim 6.1 evaluation for simplified deterministic ACT.

The policy observes only RGB + 9 joint positions + measured EE position.
Cube pose is used exclusively for logging and success metrics, never control.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
from act_model import ActionChunkTransformer

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='checkpoints/act_20_best.pt')
p.add_argument('--episodes', type=int, default=10)
p.add_argument('--duration', type=float, default=60.0)
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--headless', action='store_true')
p.add_argument('--log-every', type=int, default=10)
p.add_argument('--mode', choices=['chunk', 'receding'], default='receding',
               help='chunk: execute a full chunk; receding: replan each control step and use first action')
p.add_argument('--chunk-execute', type=int, default=0,
               help='For chunk mode: number of actions per prediction (0=checkpoint chunk size)')
p.add_argument('--csv', default='act_eval_results.csv')
args = p.parse_args()
if args.episodes < 1 or args.duration <= 0 or args.log_every < 1:
    p.error('episodes, duration, and log-every must be positive')

# SimulationApp must be created before importing Isaac Sim APIs.
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})
try:
    import omni.kit.app
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('isaacsim.robot.manipulators.examples', True)
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController

    SIZE = 0.05
    GOAL = np.array([0.50, 0.25, SIZE / 2], dtype=np.float64)
    INITIAL_TARGET = np.array([0.50, 0.00, 0.20], dtype=np.float64)
    MIN_TARGET = np.array([0.25, -0.35, 0.035], dtype=np.float64)
    MAX_TARGET = np.array([0.75, 0.45, 0.50], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    chunk_size = int(ckpt['chunk_size'])
    position_scale = float(ckpt['position_scale'])
    state_dim = int(ckpt.get('state_dim', 12))
    if state_dim != 12 or chunk_size < 1 or position_scale <= 0:
        raise ValueError('Checkpoint must specify state_dim=12, positive chunk_size/position_scale')
    # These architectural defaults match the previously supplied train_act.py.
    model = ActionChunkTransformer(state_dim=state_dim, chunk_size=chunk_size,
                                   pretrained=False,
                                   freeze_backbone=bool(ckpt.get('freeze_backbone', True)))
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    transform = ResNet18_Weights.DEFAULT.transforms()
    execute_n = args.chunk_execute or chunk_size
    if not 1 <= execute_n <= chunk_size:
        raise ValueError('--chunk-execute must be within [1, checkpoint chunk_size]')
    print(f'[MODEL] checkpoint={args.checkpoint} device={device} chunk={chunk_size} '
          f'scale={position_scale} mode={args.mode} execute_n={execute_n}', flush=True)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
    cube = world.scene.add(DynamicCuboid(prim_path='/World/Cube', name='cube',
        position=np.array([0.5, 0.0, 0.025]), scale=np.array([SIZE]*3),
        color=np.array([1., .65, 0.]), mass=0.05))
    world.scene.add(VisualCuboid(prim_path='/World/GoalMarker', name='goal_marker',
        position=np.array([GOAL[0], GOAL[1], .002]),
        scale=np.array([.08, .08, .004]), color=np.array([0., 1., 0.])))
    camera = Camera(prim_path='/World/RecorderCamera', name='recorder_camera',
                    position=np.array([1.2, .8, 1.]), resolution=(224, 224))
    world.reset()
    set_camera_view(eye=np.array([1.2, .8, 1.]), target=np.array([.5, .12, .08]),
                    camera_prim_path='/World/RecorderCamera')
    camera.initialize()
    camera.add_rgb_to_frame()
    for _ in range(30):
        world.step(render=True)
    if not args.headless:
        set_camera_view(eye=np.array([1.5, 1.5, 1.2]), target=np.array([.45, .15, .1]),
                        camera_prim_path='/OmniverseKit_Persp')
    controller = RMPFlowController(name='act_eval_rmpflow', robot_articulation=franka)
    orientation = euler_angles_to_quat(np.array([0., np.pi, 0.]))
    physics_dt = world.get_physics_dt()
    interval = max(1, round(1.0 / (10.0 * physics_dt)))
    max_controls = max(1, int(args.duration / (interval * physics_dt)))
    print(f'[SIM] dt={physics_dt:.5f} interval={interval} '
          f'effective_hz={1/(interval*physics_dt):.2f} max_controls={max_controls}', flush=True)

    def observation():
        rgb = camera.get_rgb()
        if rgb is None:
            raise RuntimeError('Camera returned None')
        rgb = np.asarray(rgb)
        if rgb.shape[:2] != (224, 224) or rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f'Unexpected camera image shape {rgb.shape}')
        rgb = rgb[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
        img = transform(torch.from_numpy(rgb.copy()).permute(2, 0, 1)).unsqueeze(0).to(device)
        joints = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        ee, _ = franka.end_effector.get_world_pose()
        ee = np.asarray(ee, dtype=np.float32)
        if len(joints) != 9:
            raise RuntimeError(f'Expected 9 joints, got {len(joints)}')
        state = np.concatenate([joints, ee]).astype(np.float32)
        return img, torch.from_numpy(state).unsqueeze(0).to(device), ee

    results = []
    for ep in range(args.episodes):
        start = np.array([rng.uniform(.44, .56), rng.uniform(-.08, .08), .025])
        world.reset()
        controller.reset()
        cube.set_world_pose(position=start.copy())
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        franka.gripper.open()
        gripper_open = True
        target = INITIAL_TARGET.copy()  # RMPFlow target, NOT measured EE position.
        for _ in range(240):
            action = controller.forward(target_end_effector_position=target,
                                        target_end_effector_orientation=orientation)
            franka.apply_action(action)
            world.step(render=True)
        ee_init, _ = franka.end_effector.get_world_pose()
        print(f'\n[EPISODE {ep+1}/{args.episodes}] cube_start={np.round(start,4)} '
              f'initial_target={INITIAL_TARGET} initial_EE={np.round(ee_init,4)}', flush=True)
        cached_pos = cached_grip = None
        chunk_index = 0
        closed = reopened = lifted = False
        max_cube_z = start[2]
        deltas = []
        policy_steps = 0
        for t in range(max_controls):
            if not simulation_app.is_running():
                break
            if args.mode == 'receding' or cached_pos is None or chunk_index >= execute_n:
                img, state, ee = observation()
                with torch.inference_mode():
                    pred_pos, pred_grip = model(img, state)
                    cached_pos = pred_pos[0].cpu().numpy() / position_scale
                    cached_grip = torch.sigmoid(pred_grip[0, :, 0]).cpu().numpy()
                chunk_index = 0
            else:
                ee, _ = franka.end_effector.get_world_pose()
            delta = np.asarray(cached_pos[chunk_index], dtype=np.float64)
            prob = float(cached_grip[chunk_index])
            if not np.isfinite(delta).all() or not np.isfinite(prob):
                raise RuntimeError('Non-finite ACT prediction')
            delta = np.clip(delta, -.025, .025)
            target = np.clip(target + delta, MIN_TARGET, MAX_TARGET)
            deltas.append(delta.copy())
            next_open = prob >= 0.5
            if next_open != gripper_open:
                print(f'[GRIPPER] step={t} {"OPEN" if gripper_open else "CLOSED"}'
                      f' -> {"OPEN" if next_open else "CLOSED"} prob={prob:.3f}', flush=True)
                if next_open:
                    franka.gripper.open()
                    if closed:
                        reopened = True
                else:
                    franka.gripper.close()
                    closed = True
                gripper_open = next_open
            for _ in range(interval):
                action = controller.forward(target_end_effector_position=target,
                                            target_end_effector_orientation=orientation)
                franka.apply_action(action)
                world.step(render=True)
            cube_pos, _ = cube.get_world_pose()
            cube_pos = np.asarray(cube_pos)
            max_cube_z = max(max_cube_z, float(cube_pos[2]))
            lifted |= cube_pos[2] > start[2] + .04
            xy = float(np.linalg.norm(cube_pos[:2] - GOAL[:2]))
            if t % args.log_every == 0:
                print(f'[POLICY {t:04d}] delta={np.round(delta,4)} '
                      f'target={np.round(target,3)} ee={np.round(ee,3)} '
                      f'cube={np.round(cube_pos,3)} grip_prob={prob:.3f} '
                      f'gripper={"OPEN" if gripper_open else "CLOSED"} '
                      f'xy={xy:.3f} lifted={lifted} chunk_idx={chunk_index}', flush=True)
            chunk_index += 1
            policy_steps += 1
        # Settle to verify that the cube was released onto the ground.
        for _ in range(30):
            world.step(render=True)
        final_cube, _ = cube.get_world_pose()
        final_cube = np.asarray(final_cube)
        xy = float(np.linalg.norm(final_cube[:2] - GOAL[:2]))
        z_err = float(abs(final_cube[2] - GOAL[2]))
        success = bool(xy < .05 and z_err < .05 and gripper_open and lifted and closed and reopened)
        smoothness = float(np.mean(np.linalg.norm(np.diff(deltas, axis=0), axis=1))) if len(deltas)>1 else float('nan')
        row = dict(episode=ep+1, success=success, final_xy_m=xy, final_z_error_m=z_err,
                   closed=closed, lifted=lifted, reopened=reopened,
                   gripper_open=gripper_open, max_cube_z_m=max_cube_z,
                   policy_steps=policy_steps, duration_s=policy_steps*interval*physics_dt,
                   mean_action_change_m=smoothness)
        results.append(row)
        print(f'[RESULT] ep={ep+1} success={success} xy={xy:.3f} z_err={z_err:.3f} '
              f'closed={closed} lifted={lifted} reopened={reopened} '
              f'policy_steps={policy_steps}', flush=True)
    if results:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        rate = np.mean([r['success'] for r in results])
        print(f'\n[SUMMARY] success={sum(r["success"] for r in results)}/{len(results)} '
              f'rate={rate:.1%} mean_xy={np.mean([r["final_xy_m"] for r in results]):.3f}m '
              f'csv={out}', flush=True)
        print('[NOTE] This is a simplified deterministic chunking transformer, not CVAE ACT.', flush=True)
finally:
    simulation_app.close()
