"""Language-conditioned expert demonstrations for Isaac Sim 6.1 / Franka.

Run (inside Isaac Sim Python environment):
 DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 python -u generate_vla_demos.py --episodes 8 --headless

Expert uses privileged object poses. Saved policy observations do NOT include them.
Verify Franka USD and camera alignment in your installation before large-scale collection.
"""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--episodes', type=int, default=8, help='Number of successful demonstrations to save')
p.add_argument('--max-attempts', type=int, default=100)
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--output-dir', default='demonstrations')
p.add_argument('--headless', action='store_true')
p.add_argument('--franka-usd', default='', help='Optional local Franka USD; if omitted use Franka example')
p.add_argument('--image-size', type=int, default=224)
p.add_argument('--grasp-z', type=float, default=0.035)
p.add_argument('--place-z', type=float, default=0.035)
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})

try:
    import omni.kit.app
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
    from isaacsim.core.utils.stage import add_reference_to_stage

    RNG = np.random.default_rng(args.seed)
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    DT = 1.0 / 60.0
    INTERVAL = 6
    SIZE = 0.05
    INITIAL = np.array([0.50, 0.00, 0.20], dtype=float)
    TARGET_MIN = np.array([0.25, -0.35, 0.035])
    TARGET_MAX = np.array([0.75, 0.45, 0.50])
    GOALS = {'left': np.array([0.40, 0.25, 0.025]),
             'right': np.array([0.60, 0.25, 0.025])}
    COLORS = {'red': np.array([0.9, 0.08, 0.08]),
              'blue': np.array([0.08, 0.25, 0.95])}
    COMMANDS = [(color, side, f'Pick up the {color} cube and place it on the {side}.')
                for color in ('red', 'blue') for side in ('left', 'right')]

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    world.scene.add_default_ground_plane()
    if args.franka_usd:
        add_reference_to_stage(args.franka_usd, '/World/Franka')
        franka = world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
    else:
        franka = world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
    controller = RMPFlowController(name='vla_rmpflow', robot_articulation=franka)
    orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

    cubes = {}
    for color, rgb in COLORS.items():
        cubes[color] = world.scene.add(DynamicCuboid(
            prim_path=f'/World/{color}_cube', name=f'{color}_cube',
            position=np.array([0.48 if color == 'red' else 0.54, 0.0, 0.025]),
            scale=np.array([SIZE] * 3), mass=0.05, color=rgb))
    for side, pos in GOALS.items():
        world.scene.add(VisualCuboid(
            prim_path=f'/World/{side}_goal', name=f'{side}_goal',
            position=np.array([pos[0], pos[1], 0.0025]),
            scale=np.array([0.08, 0.08, 0.005]),
            color=np.array([0.2, 0.8, 0.2]) if side == 'left' else np.array([0.9, 0.7, 0.1])))

    # Replicator camera: fixed viewpoint, identical across all tasks.
    camera = rep.create.camera(position=(1.2, 0.8, 1.0), look_at=(0.5, 0.12, 0.08))
    render_product = rep.create.render_product(camera, (args.image_size, args.image_size))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb_annotator.attach([render_product])

    def tick(n=1):
        for _ in range(n):
            world.step(render=True)

    def capture_rgb():
        # Annotator data may be RGBA; discard alpha.
        data = rgb_annotator.get_data()
        if data is None or np.asarray(data).size == 0:
            tick(2)
            data = rgb_annotator.get_data()
        arr = np.asarray(data)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(f'Unexpected camera frame shape: {arr.shape}')
        return arr[..., :3].astype(np.uint8).copy()

    def random_positions():
        # Keep cubes apart so expert can grasp one without colliding with the other.
        for _ in range(100):
            a = np.array([RNG.uniform(.44, .56), RNG.uniform(-.09, .09), .025])
            b = np.array([RNG.uniform(.44, .56), RNG.uniform(-.09, .09), .025])
            if np.linalg.norm(a[:2] - b[:2]) >= .11:
                return {'red': a, 'blue': b}
        raise RuntimeError('Could not sample separated cube positions')

    def collect_one(color, side, instruction, attempt):
        world.reset()
        controller.reset()
        poses = random_positions()
        for name, cube in cubes.items():
            cube.set_world_pose(position=poses[name])
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        franka.gripper.open()
        target = INITIAL.copy()
        gripper_open = True
        tick(240)
        fields = {key: [] for key in ('rgb', 'joint_positions', 'ee_position',
                                      'action_delta', 'gripper_open')}
        lifted = False

        def control_step(next_target=None, next_gripper=None):
            nonlocal target, gripper_open, lifted
            joints = np.asarray(franka.get_joint_positions(), dtype=np.float32).copy()
            ee, _ = franka.end_effector.get_world_pose()
            rgb = capture_rgb()
            previous = target.copy()
            if next_target is not None:
                target = np.clip(np.asarray(next_target, dtype=float), TARGET_MIN, TARGET_MAX)
            if next_gripper is not None and bool(next_gripper) != gripper_open:
                gripper_open = bool(next_gripper)
                if gripper_open:
                    franka.gripper.open()
                else:
                    franka.gripper.close()
            action = controller.forward(target_end_effector_position=target,
                                        target_end_effector_orientation=orientation)
            franka.apply_action(action)
            tick(INTERVAL)
            cp, _ = cubes[color].get_world_pose()
            if cp[2] > .065:
                lifted = True
            if joints.shape[0] != 9:
                raise RuntimeError(f'Expected 9 Franka joints, got {joints.shape[0]}')
            fields['rgb'].append(rgb)
            fields['joint_positions'].append(joints)
            fields['ee_position'].append(np.asarray(ee, dtype=np.float32))
            fields['action_delta'].append((target - previous).astype(np.float32))
            fields['gripper_open'].append(np.array([float(gripper_open)], dtype=np.float32))

        def move_to(destination, speed=.006, hold=8):
            start = target.copy()
            destination = np.asarray(destination, dtype=float)
            count = max(1, int(np.ceil(np.max(np.abs(destination - start)) / speed)))
            for k in range(1, count + 1):
                control_step(start + (destination - start) * k / count)
            for _ in range(hold):
                control_step()

        def wait(n):
            for _ in range(n):
                control_step()

        obj = poses[color]
        goal = GOALS[side]
        move_to([obj[0], obj[1], .20], hold=10)
        move_to([obj[0], obj[1], args.grasp_z], speed=.005, hold=15)
        control_step(next_gripper=False)
        wait(25)
        move_to([obj[0], obj[1], .20], speed=.005, hold=10)
        # Stop early if the grasp failed, avoiding a wasted transport trajectory.
        cp, _ = cubes[color].get_world_pose()
        if cp[2] < .065:
            return False, f'grasp failed: z={cp[2]:.3f}'
        move_to([goal[0], goal[1], .20], speed=.006, hold=12)
        move_to([goal[0], goal[1], args.place_z], speed=.004, hold=18)
        control_step(next_gripper=True)
        wait(25)
        move_to([goal[0], goal[1], .20], speed=.006, hold=15)
        wait(15)
        final, _ = cubes[color].get_world_pose()
        xy = float(np.linalg.norm(final[:2] - goal[:2]))
        z_err = float(abs(final[2] - goal[2]))
        success = lifted and xy < .05 and z_err < .05
        if not success:
            return False, f'placement failed: xy={xy:.3f} z_err={z_err:.3f}'
        filename = OUT / f'episode_{attempt:05d}.npz'
        np.savez_compressed(filename,
            **{k: np.asarray(v) for k, v in fields.items()},
            instruction=np.array(instruction), object_color=np.array(color),
            goal_side=np.array(side), goal_position=goal.astype(np.float32),
            success=np.array(True), control_hz=np.array(10.0))
        return True, f'{filename} frames={len(fields["rgb"])} xy={xy:.3f}'

    saved = 0
    for attempt in range(1, args.max_attempts + 1):
        if saved >= args.episodes:
            break
        # Balanced commands among successful episodes (retry failed commands).
        color, side, instruction = COMMANDS[saved % len(COMMANDS)]
        ok, message = collect_one(color, side, instruction, attempt)
        print(f'[ATTEMPT {attempt}] {color}/{side} success={ok} {message}', flush=True)
        if ok:
            saved += 1
    print(f'[SUMMARY] saved={saved}/{args.episodes} attempts={attempt}', flush=True)
finally:
    simulation_app.close()
