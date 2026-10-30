"""Isaac Sim 6.1 migration of the original simple BC evaluator.
Run: DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 python evaluate_simple_6_1_basic.py --franka-usd /path/to/panda.usd
"""
import traceback
import faulthandler

faulthandler.enable()

import argparse
from pathlib import Path
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--duration', type=float, default=60.0)
parser.add_argument('--checkpoint', default='bc_best.pt')
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})

try:
    import torch
    import omni.kit.app
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.stage import add_reference_to_stage

    # Isaac Sim 6.1 bundles the legacy Franka example in extsDeprecated.
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate('isaacsim.robot.manipulators.examples', True)
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
    from model import BehaviorCloning

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CUBE_SIZE = 0.05
    GOAL = np.array([0.40, 0.30, CUBE_SIZE / 2], dtype=np.float64)
    START_TARGET = np.array([0.50, 0.00, 0.20], dtype=np.float64)
    MIN_TARGET = np.array([0.25, -0.35, 0.035], dtype=np.float64)
    MAX_TARGET = np.array([0.75, 0.45, 0.50], dtype=np.float64)
    CONTROL_PERIOD = 0.1
    POSITION_CLIP = 0.02
    GOAL_TOLERANCE_XY = 0.05
    GOAL_TOLERANCE_Z = 0.05
    ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    obs_dim = int(checkpoint.get('obs_dim', 18))
    position_scale = float(checkpoint.get('position_scale', 100.0))
    model = BehaviorCloning(obs_dim=obs_dim).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'BC model loaded: {checkpoint_path}; obs_dim={obs_dim}; scale={position_scale}', flush=True)
    
    print("[DEBUG] Creating World...", flush=True)
    world = World(stage_units_in_meters=1.0)

    print("[DEBUG] Adding ground plane...", flush=True)
    world.scene.add_default_ground_plane()

    # Franka's constructor can attach to a pre-existing articulation prim.
    # print("[DEBUG] Loading Franka USD...", flush=True)
    # add_reference_to_stage(usd_path=args.franka_usd, prim_path='/World/Franka')
    print("[DEBUG] Creating Franka object...", flush=True)
    franka = world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
    print("[DEBUG] Creating cube...", flush=True)
    
    cube = world.scene.add(DynamicCuboid(
        prim_path='/World/Cube', name='cube',
        position=np.array([0.50, 0.00, 0.025]),
        scale=np.array([CUBE_SIZE] * 3),
        color=np.array([1.0, 0.1, 0.1]), mass=0.05,
    ))
    world.scene.add(VisualCuboid(
        prim_path='/World/Goal', name='goal_marker',
        position=np.array([GOAL[0], GOAL[1], 0.0025]),
        scale=np.array([0.06, 0.06, 0.005]),
        color=np.array([0.1, 0.8, 0.1]),
    ))
    world.reset()
    joint_names = list(franka.dof_names)
    print('Franka DOFs:', joint_names, flush=True)
    if len(joint_names) != 9 or sum('panda_joint' in n for n in joint_names) != 7:
        raise RuntimeError('Expected Franka Panda with 7 arm joints + 2 finger joints; check --franka-usd')
    if obs_dim != len(joint_names) + 9:
        raise RuntimeError(f'Checkpoint obs_dim={obs_dim}, but full joint observation requires {len(joint_names)+9}')

    rmpflow = RMPFlowController(name='bc_rmpflow', robot_articulation=franka)

    def get_observation():
        joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float64)
        ee_position, _ = franka.end_effector.get_world_pose()
        cube_position, _ = cube.get_world_pose()
        observation = np.concatenate([joint_positions, np.asarray(ee_position), np.asarray(cube_position), GOAL])
        if observation.shape != (obs_dim,) or not np.isfinite(observation).all():
            raise ValueError(f'Invalid observation shape/values: {observation}')
        return observation

    def predict_action(observation):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            position_pred, gripper_logits = model(obs_tensor)
        delta = position_pred.squeeze(0).cpu().numpy() / position_scale
        return np.clip(delta, -POSITION_CLIP, POSITION_CLIP), float(torch.sigmoid(gripper_logits).item())

    def reset_episode():
        world.reset()
        rmpflow.reset()
        spawn_position = np.array([np.random.uniform(0.44, 0.56), np.random.uniform(-0.08, 0.08), 0.025])
        cube.set_world_pose(position=spawn_position)
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        franka.gripper.open()
        for _ in range(60):
            world.step(render=not args.headless)
        return spawn_position

    def check_success():
        cube_position, _ = cube.get_world_pose()
        xy_error = float(np.linalg.norm(cube_position[:2] - GOAL[:2]))
        z_error = float(abs(cube_position[2] - GOAL[2]))
        return xy_error < GOAL_TOLERANCE_XY and z_error < GOAL_TOLERANCE_Z, xy_error, z_error

    success_count = 0
    for episode in range(args.episodes):
        start = reset_episode()
        print(f'Episode {episode+1}/{args.episodes} start={np.round(start, 3)}', flush=True)
        target = START_TARGET.copy()
        gripper_open = True
        action_count = 0
        episode_start_time = world.current_time
        next_control_time = world.current_time
        last_delta = np.zeros(3)
        last_p_open = 1.0
        last_action = None
        while simulation_app.is_running():
            if not world.is_playing():
                world.step(render=not args.headless)
                continue
            if world.current_time - episode_start_time >= args.duration:
                break
            if world.current_time >= next_control_time - 1e-6:
                next_control_time += CONTROL_PERIOD
                observation = get_observation()
                delta, p_open = predict_action(observation)
                target = np.clip(target + delta, MIN_TARGET, MAX_TARGET)
                if p_open < 0.35:
                    gripper_open = False
                elif p_open > 0.65:
                    gripper_open = True
                last_action = rmpflow.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=ORIENTATION,
                )
                if gripper_open:
                    franka.gripper.open()
                else:
                    franka.gripper.close()
                action_count += 1
                last_delta, last_p_open = delta, p_open
                if action_count % 50 == 0:
                    cube_position, _ = cube.get_world_pose()
                    ee_position, _ = franka.end_effector.get_world_pose()
                    relative = np.asarray(ee_position) - np.asarray(cube_position)
                    print(f't={world.current_time-episode_start_time:.1f}s '
                          f'cube={np.round(cube_position, 3)} EE-Cube={np.round(relative, 3)} '
                          f'target={np.round(target, 3)} BC_delta={np.round(last_delta, 4)} '
                          f'gripper_open={gripper_open} p_open={last_p_open:.3f}', flush=True)
            if last_action is not None:
                franka.apply_action(last_action)
            world.step(render=not args.headless)

        success, xy_error, z_error = check_success()
        success_count += int(success)
        final_cube_position, _ = cube.get_world_pose()
        print(f'Episode {episode+1}: success={success} XY_error={xy_error:.3f}m '
              f'Z_error={z_error:.3f}m final_cube={np.round(final_cube_position, 3)}', flush=True)

    if args.episodes:
        print(f'\nPosition success rate: {success_count}/{args.episodes} '
              f'({100.0 * success_count / args.episodes:.1f}%)', flush=True)
except Exception:
    print("EVALUATION ERROR:", flush=True)
    traceback.print_exc()
    raise

finally:
    print("Closing Isaac Sim...", flush=True)
    simulation_app.close()
