
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
    # 2. Configuration
    # ============================================

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

    OUTPUT_DIR = Path("demonstrations")
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

    world.reset()

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
                ),
                ("Demo", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("DemoScript", "omni.graph.scriptnode.ScriptNode"),
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
                ),
                ("Demo.inputs:topicName", "/demo_command"),
                ("Demo.inputs:messagePackage", "std_msgs"),
                ("Demo.inputs:messageSubfolder", "msg"),
                ("Demo.inputs:messageName", "String"),
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
                ),
                ("Tick.outputs:tick", "Demo.inputs:execIn"),
                ("Context.outputs:context", "Demo.inputs:context"),
                ("Demo.outputs:execOut", "DemoScript.inputs:execIn"),
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

def compute(db):
    command = str(
        db.inputs.command
    ).strip().lower()

    if command in ("r", "p", "n"):
        shared.demo_queue.append(command)

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

    # ============================================
    # 9. Dataset structure
    # ============================================

    FIELDS = (
        "joint_positions",
        "joint_velocities",
        "ee_position",
        "cube_position",
        "target_position",
        "action_delta",
        "gripper_open",
        "gripper_command",
        "sim_time",
    )

    def new_episode():
        return {
            field: []
            for field in FIELDS
        }

    episode = new_episode()
    recording = False

    sim_step = 0
    physics_dt = world.get_physics_dt()

    # ============================================
    # 10. Save demonstration
    # ============================================

    def save_episode(data):
        steps = len(data["sim_time"])

        if steps == 0:
            print(
                "[REC] Empty episode. Nothing saved.",
                flush=True,
            )
            return

        cube_pos, _ = cube.get_world_pose()

        goal_distance = float(
            np.linalg.norm(
                np.asarray(cube_pos[:2]) - GOAL[:2]
            )
        )

        success = goal_distance < SUCCESS_THRESHOLD

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            OUTPUT_DIR / f"episode_{timestamp}.npz"
        )

        payload = {
            key: np.asarray(values)
            for key, values in data.items()
        }

        payload["goal_position"] = GOAL.copy()
        payload["cube_start"] = CUBE_START.copy()
        payload["physics_dt"] = np.asarray(physics_dt)
        payload["success"] = np.asarray(success)
        payload["goal_distance"] = np.asarray(goal_distance)
        payload["format_version"] = np.asarray(2)

        np.savez_compressed(
            filename,
            **payload,
        )

        print(
            f"[REC] Saved {steps} steps -> {filename}",
            flush=True,
        )

        print(
            f"[REC] Goal distance: {goal_distance:.3f} m",
            flush=True,
        )

        print(
            f"[REC] Goal reached (XY): {success}",
            flush=True,
        )

    # ============================================
    # 11. Reset environment
    # ============================================

    def reset_environment():
        nonlocal_dummy = None  # No effect; reset uses returned state.

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

        return INITIAL_TARGET.copy()

    # ============================================
    # 12. Main simulation loop
    # ============================================

    print("", flush=True)
    print("================================", flush=True)
    print(" FRANKA DEMONSTRATION RECORDER", flush=True)
    print("================================", flush=True)
    print("W/S : X movement", flush=True)
    print("A/D : Y movement", flush=True)
    print("Q/E : Z movement", flush=True)
    print("O   : Open gripper", flush=True)
    print("C   : Close gripper", flush=True)
    print("R   : Start recording", flush=True)
    print("P   : Save recording", flush=True)
    print("N   : Reset environment", flush=True)
    print("================================", flush=True)

    while simulation_app.is_running():

        if not world.is_playing():
            world.step(render=True)
            continue

        # ------------------------------------
        # Handle recording commands
        # ------------------------------------

        while shared.demo_queue:

            command = shared.demo_queue.popleft()

            if command == "r":

                if recording:
                    print(
                        "[REC] Already recording",
                        flush=True,
                    )
                else:
                    episode = new_episode()
                    recording = True

                    print(
                        "[REC] Recording started",
                        flush=True,
                    )

            elif command == "p":

                if recording:
                    recording = False

                    save_episode(episode)

                    episode = new_episode()

                else:
                    print(
                        "[REC] Nothing to save",
                        flush=True,
                    )

            elif command == "n":

                recording = False
                episode = new_episode()

                shared.move_queue.clear()
                shared.gripper_queue.clear()

                target = reset_environment()
                gripper_open = True

                sim_step = 0

                print(
                    "[REC] Environment reset",
                    flush=True,
                )

        # ------------------------------------
        # Read current observation
        # BEFORE applying new commands
        # ------------------------------------

        if recording:

            joints = np.asarray(
                franka.get_joint_positions(),
                dtype=np.float64,
            ).copy()

            velocities = np.asarray(
                franka.get_joint_velocities(),
                dtype=np.float64,
            ).copy()

            ee_pos, _ = (
                franka.end_effector.get_world_pose()
            )

            cube_pos, _ = cube.get_world_pose()

            obs_ee = np.asarray(
                ee_pos,
                dtype=np.float64,
            ).copy()

            obs_cube = np.asarray(
                cube_pos,
                dtype=np.float64,
            ).copy()

        # ------------------------------------
        # Process movement commands
        # ------------------------------------

        applied_delta = np.zeros(
            3,
            dtype=np.float64,
        )

        while shared.move_queue:

            delta = np.asarray(
                shared.move_queue.popleft(),
                dtype=np.float64,
            )

            if delta.shape != (3,):
                continue

            if not np.all(np.isfinite(delta)):
                continue

            delta = np.clip(
                delta,
                -0.025,
                0.025,
            )

            old_target = target.copy()

            target = np.clip(
                target + delta,
                LIMITS_MIN,
                LIMITS_MAX,
            )

            applied_delta += target - old_target

            print(
                "[MOVE] target:",
                np.round(target, 3),
                flush=True,
            )

        # ------------------------------------
        # Process gripper commands
        # ------------------------------------

        gripper_command = -1

        while shared.gripper_queue:

            gripper_open = bool(
                shared.gripper_queue.popleft()
            )

            gripper_command = int(gripper_open)

            if gripper_open:
                franka.gripper.open()
            else:
                franka.gripper.close()

            print(
                "[GRIPPER]",
                "OPEN" if gripper_open else "CLOSE",
                flush=True,
            )

        # ------------------------------------
        # Record observation-action pair
        # ------------------------------------

        if recording:

            episode["joint_positions"].append(
                joints
            )

            episode["joint_velocities"].append(
                velocities
            )

            episode["ee_position"].append(
                obs_ee
            )

            episode["cube_position"].append(
                obs_cube
            )

            episode["target_position"].append(
                target.copy()
            )

            episode["action_delta"].append(
                applied_delta.copy()
            )

            episode["gripper_open"].append(
                int(gripper_open)
            )

            episode["gripper_command"].append(
                gripper_command
            )

            episode["sim_time"].append(
                sim_step * physics_dt
            )

        # ------------------------------------
        # Apply RMPFlow action
        # ------------------------------------

        action = controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=orientation,
        )

        franka.apply_action(action)

        world.step(render=True)

        sim_step += 1

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
