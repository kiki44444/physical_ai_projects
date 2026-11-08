"""
Isaac Sim 6.1 - Franka Demonstration Recorder
ROS 2 topics:
    /ee_delta         geometry_msgs/msg/Twist
    /gripper_command  std_msgs/msg/Bool
    /demo_command     std_msgs/msg/String
Keyboard:
    W/S : X movement
    A/D : Y movement
    Q/E : Z movement
    O   : Open gripper
    C   : Close gripper
    R   : Start recording
    P   : Save recording
    N   : Cancel recording and reset
    X   : Exit keyboard controller
Output:
    demonstrations/episode_YYYYmmdd_HHMMSS_ffffff.npz
"""
import sys
import types
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})
try:
    import omni.kit.app
    import omni.graph.core as og
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import (
        DynamicCuboid,
        VisualCuboid,
    )
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    # ============================================
    # 1. Enable Isaac Sim extensions
    # ============================================
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in (
        "isaacsim.ros2.bridge",
        "isaacsim.robot.manipulators.examples",
        "omni.graph.scriptnode",
    ):
        manager.set_extension_enabled_immediate(
            extension, True
        )
        if not manager.is_extension_enabled(extension):
            raise RuntimeError(
                f"Extension not enabled: {extension}"
            )
        print(f"[OK] {extension}", flush=True)
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController
    )
    # ============================================
    # 2. Configuration (random cube start per reset)
    # ============================================
    CUBE_SIZE = 0.05
    rng = np.random.default_rng()
    def sample_cube_start():
        return np.array([rng.uniform(0.44, 0.56), rng.uniform(-0.08, 0.08), CUBE_SIZE / 2], dtype=np.float64)
    CUBE_START = sample_cube_start()
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
    OUTPUT_DIR = Path("vision_demonstrations")
    CONTROL_HZ = 10.0
    IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ============================================
    # 3. Shared command queues
    # ============================================
    shared = types.ModuleType(
        "franka_recorder_shared"
    )
    shared.move_queue = deque()
    shared.gripper_queue = deque()
    shared.demo_queue = deque()
    sys.modules[shared.__name__] = shared
    # ============================================
    # 4. Create simulation world
    # ============================================
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
            scale=np.array([
                CUBE_SIZE,
                CUBE_SIZE,
                CUBE_SIZE,
            ]),
            color=np.array([1.0, 0.65, 0.0]),
            mass=0.05,
        )
    )
    # Visual-only goal marker.
    # It does not physically collide with the cube.
    goal_marker = world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            position=np.array([
                GOAL[0],
                GOAL[1],
                0.002,
            ]),
            scale=np.array([
                0.08,
                0.08,
                0.004,
            ]),
            color=np.array([0.0, 1.0, 0.0]),
        )
    )
    camera = Camera(
        prim_path="/World/RecorderCamera",
        name="recorder_camera",
        position=np.array([1.2, 0.8, 1.0]),
        resolution=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    world.reset()
    print(f"[RANDOM CUBE] initial={np.round(CUBE_START, 4)}", flush=True)
    set_camera_view(
        eye=np.array([1.2, 0.8, 1.0]),
        target=np.array([0.50, 0.12, 0.08]),
        camera_prim_path="/World/RecorderCamera",
    )
    camera.initialize()
    camera.add_rgb_to_frame()
    for _ in range(30):
        world.step(render=True)
    set_camera_view(
        eye=np.array([1.5, 1.5, 1.2]),
        target=np.array([0.45, 0.15, 0.1]),
        camera_prim_path="/OmniverseKit_Persp",
    )
    # ============================================
    # 5. RMPFlow controller
    # ============================================
    controller = RMPFlowController(
        name="recorder_rmpflow",
        robot_articulation=franka,
    )
    orientation = euler_angles_to_quat(
        np.array([0.0, np.pi, 0.0])
    )
    target = INITIAL_TARGET.copy()
    gripper_open = True
    # ============================================
    # 6. ROS 2 OmniGraph
    # ============================================
    graph_path = "/World/ROS2Recorder"
    keys = og.Controller.Keys
    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
        },
        {
            keys.CREATE_NODES: [
                (
                    "Tick",
                    "omni.graph.action.OnPlaybackTick",
                ),
                (
                    "Context",
                    "isaacsim.ros2.bridge.ROS2Context",
                ),
                (
                    "Move",
                    "isaacsim.ros2.bridge.ROS2SubscribeTwist",
                ),
                (
                    "Gripper",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                (
                    "Demo",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                (
                    "MoveScript",
                    "omni.graph.scriptnode.ScriptNode",
                ),
                (
                    "GripperScript",
                    "omni.graph.scriptnode.ScriptNode",
                ),
                (
                    "DemoScript",
                    "omni.graph.scriptnode.ScriptNode",
                )
            ],
            keys.SET_VALUES: [
                (
                    "Move.inputs:topicName",
                    "/ee_delta",
                ),
                (
                    "Gripper.inputs:topicName",
                    "/gripper_command",
                ),
                (
                    "Gripper.inputs:messagePackage",
                    "std_msgs",
                ),
                (
                    "Gripper.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "Gripper.inputs:messageName",
                    "Bool",
                ),
                (
                    "Demo.inputs:topicName",
                    "/demo_command",
                ),
                (
                    "Demo.inputs:messagePackage",
                    "std_msgs",
                ),
                (
                    "Demo.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "Demo.inputs:messageName",
                    "String",
                )
            ],
            keys.CONNECT: [
                (
                    "Tick.outputs:tick",
                    "Move.inputs:execIn",
                ),
                (
                    "Tick.outputs:tick",
                    "Gripper.inputs:execIn",
                ),
                (
                    "Tick.outputs:tick",
                    "Demo.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "Move.inputs:context",
                ),
                (
                    "Context.outputs:context",
                    "Gripper.inputs:context",
                ),
                (
                    "Context.outputs:context",
                    "Demo.inputs:context",
                ),
                (
                    "Move.outputs:execOut",
                    "MoveScript.inputs:execIn",
                ),
                (
                    "Gripper.outputs:execOut",
                    "GripperScript.inputs:execIn",
                ),
                (
                    "Demo.outputs:execOut",
                    "DemoScript.inputs:execIn",
                )
            ],
        },
    )
    # ============================================
    # 7. Script Node custom attributes
    # ============================================
    move_node = og.Controller.node(
        f"{graph_path}/MoveScript"
    )
    gripper_node = og.Controller.node(
        f"{graph_path}/GripperScript"
    )
    demo_node = og.Controller.node(
        f"{graph_path}/DemoScript"
    )
    move_attr = og.Controller.create_attribute(
        move_node,
        "delta",
        "vectord[3]",
        og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    )
    gripper_attr = og.Controller.create_attribute(
        gripper_node,
        "open",
        "bool",
        og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    )
    demo_attr = og.Controller.create_attribute(
        demo_node,
        "command",
        "string",
        og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    )
    if not all((
        move_attr.is_valid(),
        gripper_attr.is_valid(),
        demo_attr.is_valid(),
    )):
        raise RuntimeError(
            "Failed to create Script Node attributes"
        )
    og.Controller.connect(
        f"{graph_path}/Move.outputs:linearVelocity",
        f"{graph_path}/MoveScript.inputs:delta",
    )
    og.Controller.connect(
        f"{graph_path}/Gripper.outputs:data",
        f"{graph_path}/GripperScript.inputs:open",
    )
    og.Controller.connect(
        f"{graph_path}/Demo.outputs:data",
        f"{graph_path}/DemoScript.inputs:command",
    )
    # ============================================
    # 8. Script Node Python code
    # ============================================
    move_script = '''
import franka_recorder_shared as shared
def compute(db):
    delta = tuple(
        float(x) for x in db.inputs.delta
    )
    shared.move_queue.append(delta)
    return True
'''
    gripper_script = '''
import franka_recorder_shared as shared
def compute(db):
    shared.gripper_queue.append(
        bool(db.inputs.open)
    )
    return True
'''
    demo_script = '''
import franka_recorder_shared as shared
import omni.graph.core as og
def compute(db):
    attr = og.Controller.attribute(
        "/World/ROS2Recorder/Demo.outputs:data"
    )
    source_value = og.Controller.get(attr)
    input_value = db.inputs.command
    print(
        f"[DEBUG] source={source_value!r}, "
        f"script_input={input_value!r}",
        flush=True,
    )
    command = str(source_value).strip().lower()
    if command in ("r", "p", "n"):
        shared.demo_queue.append(command)
        print(f"[REC] Queued: {command}", flush=True)
    return True
'''
    og.Controller.set(
        og.Controller.attribute(
            f"{graph_path}/MoveScript.inputs:script"
        ),
        move_script,
    )
    og.Controller.set(
        og.Controller.attribute(
            f"{graph_path}/GripperScript.inputs:script"
        ),
        gripper_script,
    )
    og.Controller.set(
        og.Controller.attribute(
            f"{graph_path}/DemoScript.inputs:script"
        ),
        demo_script,
    )
    print("[OK] ROS 2 graph created", flush=True)
    demo_output = og.Controller.attribute(
        f"{graph_path}/Demo.outputs:data"
    )
    demo_input = og.Controller.attribute(
        f"{graph_path}/DemoScript.inputs:command"
    )
    print("[CHECK] Demo output valid:", demo_output.is_valid())
    print("[CHECK] Demo input valid:", demo_input.is_valid())
    print(
        "[CHECK] Demo output value:",
        repr(og.Controller.get(demo_output)),
    )
    print(
        "[CHECK] Demo input value:",
        repr(og.Controller.get(demo_input)),
    )
    # ============================================
    # 9. Vision demonstration recorder (10 Hz)
    # ============================================
    FIELDS = (
        "rgb", "joint_positions", "joint_velocities", "ee_position",
        "cube_position", "target_position", "gripper_state",
        "action_delta", "gripper_open", "gripper_command", "sim_time",
    )
    def new_episode():
        return {key: [] for key in FIELDS}
    episode = new_episode()
    recording = False
    pending = None
    pending_delta = np.zeros(3, dtype=np.float64)
    pending_gripper_command = -1
    physics_dt = world.get_physics_dt()
    control_interval = max(1, round(1.0 / (CONTROL_HZ * physics_dt)))
    sim_step = 0
    episode_step = 0
    print(f"[VISION] physics_dt={physics_dt:.5f}, sample_every={control_interval} steps", flush=True)
    def capture_observation():
        rgb = camera.get_rgb()
        if rgb is None:
            raise RuntimeError("RGB camera returned None; check rendering/initialization")
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")
        rgb = rgb[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
        ee_pos, _ = franka.end_effector.get_world_pose()
        cube_pos, _ = cube.get_world_pose()
        return {
            "rgb": rgb.copy(),
            "joint_positions": np.asarray(franka.get_joint_positions(), dtype=np.float32).copy(),
            "joint_velocities": np.asarray(franka.get_joint_velocities(), dtype=np.float32).copy(),
            "ee_position": np.asarray(ee_pos, dtype=np.float32).copy(),
            "cube_position": np.asarray(cube_pos, dtype=np.float32).copy(),
            "target_position": target.astype(np.float32).copy(),
            "gripper_state": np.uint8(gripper_open),
            "sim_time": np.float64(sim_step * physics_dt),
        }
    def finalize_pending():
        if pending is None:
            return
        for key, value in pending.items():
            episode[key].append(value)
        episode["action_delta"].append(pending_delta.astype(np.float32).copy())
        episode["gripper_open"].append(np.uint8(gripper_open))
        episode["gripper_command"].append(np.int8(pending_gripper_command))
    def save_episode(data):
        steps = len(data["sim_time"])
        if not steps:
            print("[REC] Empty episode. Nothing saved.", flush=True)
            return
        cube_pos, _ = cube.get_world_pose()
        distance = float(np.linalg.norm(np.asarray(cube_pos[:2]) - GOAL[:2]))
        success = distance < SUCCESS_THRESHOLD
        filename = OUTPUT_DIR / f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz"
        payload = {key: np.stack(values) for key, values in data.items()}
        payload.update(
            goal_position=GOAL.astype(np.float32),
            cube_start=CUBE_START.astype(np.float32),
            physics_dt=np.asarray(physics_dt),
            control_interval=np.asarray(control_interval),
            control_hz=np.asarray(1.0 / (control_interval * physics_dt)),
            success=np.asarray(success),
            goal_distance=np.asarray(distance),
            format_version=np.asarray(3),
        )
        assert all(len(payload[k]) == steps for k in FIELDS)
        np.savez_compressed(filename, **payload)
        print(f"[REC] Saved {steps} synchronized samples -> {filename}", flush=True)
        print(f"[REC] Final XY distance: {distance:.3f} m; success={success}", flush=True)
    def reset_environment():
        world.reset()
        controller.reset()
        CUBE_START[:] = sample_cube_start()
        cube.set_world_pose(position=CUBE_START.copy())
        print(f"[RANDOM CUBE] start={np.round(CUBE_START, 4)}", flush=True)
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        franka.gripper.open()
        for _ in range(10):
            world.step(render=True)
        return INITIAL_TARGET.copy()
    print("[VISION RECORDER] W/S X | A/D Y | Q/E Z | O/C grip | R start | P save | N random reset", flush=True)
    while simulation_app.is_running():
        if not world.is_playing():
            world.step(render=True)
            continue
        # R/P/N are handled before the observation-action interval.
        while shared.demo_queue:
            command = shared.demo_queue.popleft()
            if command == "r":
                if not recording:
                    episode = new_episode()
                    recording = True
                    pending = None
                    pending_delta = np.zeros(3, dtype=np.float64)
                    pending_gripper_command = -1
                    episode_step = 0
                    print("[REC] Recording started", flush=True)
            elif command == "p":
                if recording:
                    # Discard incomplete final interval to retain fixed-duration actions.
                    recording = False
                    pending = None
                    save_episode(episode)
                    episode = new_episode()
                else:
                    print("[REC] Nothing to save", flush=True)
            elif command == "n":
                recording = False
                pending = None
                episode = new_episode()
                shared.move_queue.clear()
                shared.gripper_queue.clear()
                target = reset_environment()
                gripper_open = True
                episode_step = 0
                print("[REC] Environment reset", flush=True)
        # Finalize the previous 0.1 s action, then sample the next observation.
        if recording and episode_step % control_interval == 0:
            if pending is not None:
                finalize_pending()
            pending = capture_observation()
            pending_delta = np.zeros(3, dtype=np.float64)
            pending_gripper_command = -1
        # Apply all ROS 2 commands received since the previous physics tick.
        while shared.move_queue:
            delta = np.asarray(shared.move_queue.popleft(), dtype=np.float64)
            if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                continue
            delta = np.clip(delta, -0.025, 0.025)
            old_target = target.copy()
            target = np.clip(target + delta, LIMITS_MIN, LIMITS_MAX)
            if recording and pending is not None:
                pending_delta += target - old_target
        while shared.gripper_queue:
            gripper_open = bool(shared.gripper_queue.popleft())
            if recording and pending is not None:
                pending_gripper_command = int(gripper_open)
            if gripper_open:
                franka.gripper.open()
            else:
                franka.gripper.close()
        action = controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=orientation,
        )
        franka.apply_action(action)
        world.step(render=True)
        sim_step += 1
        if recording:
            episode_step += 1
except KeyboardInterrupt:
    print(
        "Interrupted by user",
        flush=True,
    )
except Exception:
    import traceback
    traceback.print_exc()
finally:
    simulation_app.close()
