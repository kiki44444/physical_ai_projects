"""Automatic physical Franka pick/place demonstrations for Isaac Sim 6.1.
Keeps the manual recorder's NPZ keys and 10 Hz observation/action alignment.
Run: DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 python -u generate_expert_demos.py --episodes 20
Tune --grasp-z and --place-z after inspecting a few attempts; no teleport attachment.
"""
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--episodes', type=int, default=20, help='Number of successful episodes to collect')
p.add_argument('--max-attempts', type=int, default=100)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output-dir', default='vision_demonstrations_auto')
p.add_argument('--headless', action='store_true')
p.add_argument('--grasp-z', type=float, default=0.035, help='RMPFlow grasp target Z, calibrate experimentally')
p.add_argument('--place-z', type=float, default=0.035, help='RMPFlow release target Z, calibrate experimentally')
p.add_argument('--xy-tolerance', type=float, default=0.05)
args = p.parse_args()
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
    INITIAL = np.array([0.50, 0.00, 0.20], dtype=np.float64)
    MIN_TARGET = np.array([0.25, -0.35, 0.035])
    MAX_TARGET = np.array([0.75, 0.45, 0.50])
    FIELDS = ('rgb', 'joint_positions', 'joint_velocities', 'ee_position', 'cube_position', 'target_position', 'gripper_state', 'action_delta', 'gripper_open', 'gripper_command', 'sim_time')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
    cube = world.scene.add(DynamicCuboid(prim_path='/World/Cube', name='cube', position=np.array([.5, 0, .025]), scale=np.array([SIZE]*3), color=np.array([1., .65, 0.]), mass=.05))
    world.scene.add(VisualCuboid(prim_path='/World/GoalMarker', name='goal_marker', position=np.array([GOAL[0], GOAL[1], .002]), scale=np.array([.08,.08,.004]), color=np.array([0.,1.,0.])))
    camera = Camera(prim_path='/World/RecorderCamera', name='recorder_camera', position=np.array([1.2,.8,1.]), resolution=(224,224))
    world.reset()
    set_camera_view(eye=np.array([1.2,.8,1.]), target=np.array([.5,.12,.08]), camera_prim_path='/World/RecorderCamera')
    camera.initialize()
    camera.add_rgb_to_frame()
    for _ in range(30):
        world.step(render=True)
    if not args.headless:
        set_camera_view(eye=np.array([1.5,1.5,1.2]), target=np.array([.45,.15,.1]), camera_prim_path='/OmniverseKit_Persp')
    controller = RMPFlowController(name='auto_expert_rmpflow', robot_articulation=franka)
    orientation = euler_angles_to_quat(np.array([0.,np.pi,0.]))
    physics_dt = world.get_physics_dt()
    interval = max(1, round(1. / (10. * physics_dt)))
    actual_hz = 1. / (interval * physics_dt)
    print(f'[SETUP] physics_dt={physics_dt:.5f} control_interval={interval} rate={actual_hz:.2f}Hz', flush=True)
    target = INITIAL.copy()
    gripper_open = True
    sim_step = 0
    samples = None
    def capture():
        rgb = camera.get_rgb()
        if rgb is None:
            raise RuntimeError('Camera RGB unavailable')
        rgb = np.asarray(rgb)
        if rgb.shape != (224,224,3) and (rgb.ndim != 3 or rgb.shape[:2] != (224,224) or rgb.shape[2] < 3):
            raise RuntimeError(f'Bad camera shape: {rgb.shape}')
        rgb = rgb[:,:,:3]
        rgb = (np.clip(rgb,0,1)*255).astype(np.uint8) if np.issubdtype(rgb.dtype,np.floating) else rgb.astype(np.uint8)
        ee,_ = franka.end_effector.get_world_pose()
        cp,_ = cube.get_world_pose()
        return {'rgb':rgb.copy(), 'joint_positions':np.asarray(franka.get_joint_positions(),dtype=np.float32).copy(), 'joint_velocities':np.asarray(franka.get_joint_velocities(),dtype=np.float32).copy(), 'ee_position':np.asarray(ee,dtype=np.float32).copy(), 'cube_position':np.asarray(cp,dtype=np.float32).copy(), 'target_position':target.astype(np.float32).copy(), 'gripper_state':np.uint8(gripper_open), 'sim_time':np.float64(sim_step*physics_dt)}
    def physics_steps(n):
        global sim_step
        for _ in range(n):
            if not simulation_app.is_running():
                raise KeyboardInterrupt('Isaac Sim stopped')
            action = controller.forward(target_end_effector_position=target, target_end_effector_orientation=orientation)
            franka.apply_action(action)
            world.step(render=True)
            sim_step += 1
    def control_step(next_target=None, next_gripper=None):
        global target, gripper_open
        before = capture()
        old_target = target.copy()
        if next_target is not None:
            target = np.clip(np.asarray(next_target,dtype=np.float64), MIN_TARGET, MAX_TARGET)
        delta = target - old_target
        cmd = -1
        if next_gripper is not None and bool(next_gripper) != gripper_open:
            gripper_open = bool(next_gripper)
            cmd = int(gripper_open)
            if gripper_open:
                franka.gripper.open()
            else:
                franka.gripper.close()
        physics_steps(interval)
        if samples is not None:
            for key,value in before.items():
                samples[key].append(value)
            samples['action_delta'].append(delta.astype(np.float32))
            samples['gripper_open'].append(np.uint8(gripper_open))
            samples['gripper_command'].append(np.int8(cmd))
    def move_to(destination, speed=.008, hold=8):
        destination = np.asarray(destination,dtype=np.float64)
        start = target.copy()
        count = max(1, int(np.ceil(np.max(np.abs(destination-start))/speed)))
        for k in range(1,count+1):
            control_step(start+(destination-start)*k/count)
        for _ in range(hold):
            control_step()
    def wait(n):
        for _ in range(n):
            control_step()
    def cube_pose():
        pos,_ = cube.get_world_pose()
        return np.asarray(pos,dtype=np.float64)
    def save(data,start,attempt):
        payload = {k:np.stack(v) for k,v in data.items()}
        cp = cube_pose()
        dist = float(np.linalg.norm(cp[:2]-GOAL[:2]))
        payload.update(goal_position=GOAL.astype(np.float32), cube_start=start.astype(np.float32), physics_dt=np.asarray(physics_dt), control_interval=np.asarray(interval), control_hz=np.asarray(actual_hz), success=np.asarray(True), goal_distance=np.asarray(dist), format_version=np.asarray(3))
        filename = out / f'episode_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.npz'
        assert all(len(payload[k]) == len(payload['rgb']) for k in FIELDS)
        assert payload['joint_positions'].shape[1] == 9
        np.savez_compressed(filename, **payload)
        print(f'[SAVE] attempt={attempt} frames={len(payload["rgb"])} xy={dist:.3f} -> {filename}',flush=True)
    successes = 0
    for attempt in range(1,args.max_attempts+1):
        if successes >= args.episodes:
            break
        start = np.array([rng.uniform(.44,.56),rng.uniform(-.08,.08),SIZE/2])
        print(f'[ATTEMPT {attempt}] start={np.round(start,4)}',flush=True)
        world.reset()
        controller.reset()
        cube.set_world_pose(position=start.copy())
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        franka.gripper.open()
        gripper_open = True
        target = INITIAL.copy()
        samples = None
        physics_steps(240)
        samples = {key:[] for key in FIELDS}
        # Offset XY by the randomized cube position. Heights are RMPFlow target heights,
        # NOT actual finger/cube contact heights; calibrate grasp-z/place-z in your scene.
        approach = np.array([start[0],start[1],.20])
        grasp = np.array([start[0],start[1],args.grasp_z])
        lift = np.array([start[0],start[1],.20])
        above_goal = np.array([GOAL[0],GOAL[1],.20])
        place = np.array([GOAL[0],GOAL[1],args.place_z])
        move_to(approach,hold=10)
        move_to(grasp,speed=.005,hold=15)
        control_step(next_gripper=False)
        wait(25)
        move_to(lift,speed=.005,hold=10)
        lift_pos = cube_pose()
        lifted = lift_pos[2] > SIZE/2 + .035
        if not lifted:
            print(f'[REJECT] failed grasp/lift: cube={np.round(lift_pos,3)}',flush=True)
            samples = None
            continue
        move_to(above_goal,speed=.006,hold=12)
        move_to(place,speed=.004,hold=18)
        control_step(next_gripper=True)
        wait(25)
        move_to(above_goal,speed=.006,hold=15)
        wait(15)
        final = cube_pose()
        dist = float(np.linalg.norm(final[:2]-GOAL[:2]))
        on_ground = abs(final[2]-SIZE/2) < .012
        success = dist < args.xy_tolerance and on_ground and gripper_open
        print(f'[RESULT] attempt={attempt} success={success} xy={dist:.3f} z={final[2]:.3f} lifted={lifted}',flush=True)
        if success:
            save(samples,start,attempt)
            successes += 1
        samples = None
    print(f'[SUMMARY] saved={successes}/{args.episodes} attempts={attempt if args.max_attempts else 0}',flush=True)
finally:
    simulation_app.close()
