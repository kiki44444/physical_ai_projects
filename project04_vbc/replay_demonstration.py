"""Replay recorded expert actions in Isaac Sim 6.1, without a learned policy.

Run:
 DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 python -u replay_demonstration.py \
   --data-dir vision_demonstrations --init-steps 240 --log-every 10

Recorded action_delta is a *target displacement per 10 Hz interval*, not
an end-effector displacement. This replay accumulates it once per interval.
"""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='vision_demonstrations')
parser.add_argument('--episode', default=None, help='Optional explicit .npz path')
parser.add_argument('--init-steps', type=int, default=240)
parser.add_argument('--log-every', type=int, default=10)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()

files = sorted(Path(args.data_dir).glob('episode_*.npz'))
source = Path(args.episode) if args.episode else (files[0] if files else None)
if source is None or not source.is_file():
    raise FileNotFoundError('No demonstration .npz found')

with np.load(source, allow_pickle=False) as data:
    required = ['action_delta', 'gripper_open', 'ee_position', 'cube_position',
                'target_position', 'joint_positions']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f'{source}: missing {missing}')
    delta_recorded = np.asarray(data['action_delta'], dtype=np.float64).copy()
    grip_recorded = np.asarray(data['gripper_open'], dtype=np.int8).copy()
    ee_recorded = np.asarray(data['ee_position'], dtype=np.float64).copy()
    cube_recorded = np.asarray(data['cube_position'], dtype=np.float64).copy()
    target_recorded = np.asarray(data['target_position'], dtype=np.float64).copy()
    joints_recorded = np.asarray(data['joint_positions'], dtype=np.float64).copy()
    recorded_hz = float(data['control_hz']) if 'control_hz' in data else 10.0
    cube_start = np.asarray(data['cube_start'], dtype=np.float64).copy() if 'cube_start' in data else cube_recorded[0].copy()
    goal = np.asarray(data['goal_position'], dtype=np.float64).copy() if 'goal_position' in data else np.array([0.50, 0.25, 0.025])

n = len(delta_recorded)
if n == 0 or any(len(x) != n for x in (grip_recorded, ee_recorded, cube_recorded, target_recorded, joints_recorded)):
    raise ValueError('Empty or inconsistent demonstration lengths')
if not np.isfinite(delta_recorded).all() or not np.isfinite(target_recorded).all():
    raise ValueError('Non-finite recorded actions/targets')
if not np.isin(grip_recorded, [0, 1]).all():
    raise ValueError('Unexpected gripper labels')

from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})
try:
    import omni.kit.app
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('isaacsim.robot.manipulators.examples', True)
    if not manager.is_extension_enabled('isaacsim.robot.manipulators.examples'):
        raise RuntimeError('Could not enable Franka examples extension')
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController

    cube_size = 0.05
    limits_min = np.array([0.25, -0.35, 0.035])
    limits_max = np.array([0.75, 0.45, 0.50])
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
    cube = world.scene.add(DynamicCuboid(
        prim_path='/World/Cube', name='cube', position=cube_start.copy(),
        scale=np.array([cube_size] * 3), color=np.array([1.0, 0.65, 0.0]), mass=0.05))
    world.scene.add(VisualCuboid(
        prim_path='/World/GoalMarker', name='goal_marker',
        position=np.array([goal[0], goal[1], 0.002]),
        scale=np.array([0.08, 0.08, 0.004]), color=np.array([0.0, 1.0, 0.0])))
    world.reset()
    controller = RMPFlowController(name='expert_replay_rmpflow', robot_articulation=franka)
    orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))
    physics_dt = float(world.get_physics_dt())
    interval = max(1, round(1.0 / (recorded_hz * physics_dt)))
    print(f'[DATA] {source}; frames={n}; recorded_hz={recorded_hz:.3f}', flush=True)
    print(f'[SIM] physics_dt={physics_dt:.6f}; interval={interval}; replay_hz={1/(interval*physics_dt):.3f}', flush=True)
    print(f'[DATA] first target={np.round(target_recorded[0], 4)}; first EE={np.round(ee_recorded[0], 4)}', flush=True)

    def control_step(target):
        action = controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=orientation)
        franka.apply_action(action)
        world.step(render=True)

    # Reproduce the first recorded target as a best-effort initialization.
    # Note: this cannot recover the full pre-recording history or joint velocities.
    target = target_recorded[0].copy()
    franka.gripper.open()
    print(f'[INIT] settling toward recorded initial target for {args.init_steps} steps', flush=True)
    for _ in range(args.init_steps):
        if not simulation_app.is_running():
            raise RuntimeError('Simulation stopped during initialization')
        control_step(target)
    ee_init, _ = franka.end_effector.get_world_pose()
    cube_init, _ = cube.get_world_pose()
    joint_init = np.asarray(franka.get_joint_positions(), dtype=np.float64)
    print(f'[INIT DONE] EE={np.round(ee_init, 4)}; recorded={np.round(ee_recorded[0], 4)}; '
          f'EE_error={np.linalg.norm(np.asarray(ee_init)-ee_recorded[0]):.4f}m', flush=True)
    print(f'[INIT DONE] joint_error_norm={np.linalg.norm(joint_init-joints_recorded[0]):.4f}rad; '
          f'cube={np.round(cube_init, 4)}', flush=True)

    # Each saved row is an observation at the start of a 0.1s interval,
    # followed by its recorded target displacement and gripper state at the
    # end of that interval. Apply these once, then simulate one interval.
    grip_open = True
    ee_errors, cube_errors, target_errors = [], [], []
    for i in range(n):
        if not simulation_app.is_running():
            print('[STOP] Simulation closed before end of demonstration', flush=True)
            break
        ee_now, _ = franka.end_effector.get_world_pose()
        cube_now, _ = cube.get_world_pose()
        ee_error = float(np.linalg.norm(np.asarray(ee_now) - ee_recorded[i]))
        cube_error = float(np.linalg.norm(np.asarray(cube_now) - cube_recorded[i]))
        target_error = float(np.linalg.norm(target - target_recorded[i]))
        ee_errors.append(ee_error)
        cube_errors.append(cube_error)
        target_errors.append(target_error)
        if i % max(1, args.log_every) == 0 or i in (66, 67, 68, 178, 179, n-1):
            print(f'[REPLAY {i:04d}] EE={np.round(ee_now,3)} demo_EE={np.round(ee_recorded[i],3)} '
                  f'EE_err={ee_error:.3f}m cube={np.round(cube_now,3)} '
                  f'demo_cube={np.round(cube_recorded[i],3)} cube_err={cube_error:.3f}m '
                  f'grip={int(grip_open)} next_grip={int(grip_recorded[i])} '
                  f'delta={np.round(delta_recorded[i],4)} target_err={target_error:.3f}m', flush=True)

        target = np.clip(target + delta_recorded[i], limits_min, limits_max)
        next_grip_open = bool(grip_recorded[i])
        if next_grip_open != grip_open:
            if next_grip_open:
                franka.gripper.open()
            else:
                franka.gripper.close()
            grip_open = next_grip_open
            print(f'[GRIP] frame={i} open={grip_open}', flush=True)
        for _ in range(interval):
            control_step(target)

    cube_final, _ = cube.get_world_pose()
    final_distance = float(np.linalg.norm(np.asarray(cube_final[:2]) - goal[:2]))
    demo_final_distance = float(np.linalg.norm(cube_recorded[-1, :2] - goal[:2]))
    print(f'[RESULT] final_cube={np.round(cube_final,4)} demo_final_cube={np.round(cube_recorded[-1],4)} '
          f'final_xy_distance={final_distance:.4f}m demo_final_xy_distance={demo_final_distance:.4f}m '
          f'success_xy={final_distance < 0.05}', flush=True)
    print(f'[ERROR] mean_EE={np.mean(ee_errors):.4f}m max_EE={np.max(ee_errors):.4f}m '
          f'mean_cube={np.mean(cube_errors):.4f}m max_cube={np.max(cube_errors):.4f}m '
          f'mean_target={np.mean(target_errors):.4f}m', flush=True)
    print('[NOTE] Replay success checks action/control consistency, not model generalization.', flush=True)
finally:
    simulation_app.close()
