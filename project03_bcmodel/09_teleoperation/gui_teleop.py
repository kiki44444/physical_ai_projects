
import sys
import types
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
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.viewports import set_camera_view

    # ---------------------------------------
    # 1. Enable extensions
    # ---------------------------------------

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

    # ---------------------------------------
    # 2. Shared command queues
    # ---------------------------------------
    # Script Nodes push incoming ROS commands.
    # The main loop consumes them.

    shared = types.ModuleType("franka_teleop_shared")
    shared.move_queue = deque()
    shared.gripper_queue = deque()

    sys.modules["franka_teleop_shared"] = shared

    # ---------------------------------------
    # 3. Create world
    # ---------------------------------------

    world = World(stage_units_in_meters=1.0)
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
            position=np.array([0.50, 0.00, 0.025]),
            scale=np.array([0.05, 0.05, 0.05]),
            mass=0.05,
        )
    )

    world.reset()

    set_camera_view(
        eye=np.array([1.5, 1.5, 1.2]),
        target=np.array([0.45, 0.15, 0.1]),
        camera_prim_path="/OmniverseKit_Persp",
    )

    controller = RMPFlowController(
        name="teleop_rmpflow",
        robot_articulation=franka,
    )

    orientation = euler_angles_to_quat(
        np.array([0.0, np.pi, 0.0])
    )

    target = np.array(
        [0.50, 0.00, 0.20],
        dtype=np.float64,
    )

    limits_min = np.array([0.25, -0.35, 0.035])
    limits_max = np.array([0.75, 0.45, 0.50])

    # ---------------------------------------
    # 4. Create ROS 2 Action Graph
    # ---------------------------------------

    graph_path = "/World/ROS2Teleop"
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
                    "MoveScript",
                    "omni.graph.scriptnode.ScriptNode",
                ),
                (
                    "GripperScript",
                    "omni.graph.scriptnode.ScriptNode",
                ),
            ],
            keys.SET_VALUES: [
                ("Move.inputs:topicName", "/ee_delta"),
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
                    "Context.outputs:context",
                    "Move.inputs:context",
                ),
                (
                    "Context.outputs:context",
                    "Gripper.inputs:context",
                ),
                (
                    "Move.outputs:execOut",
                    "MoveScript.inputs:execIn",
                ),
                (
                    "Gripper.outputs:execOut",
                    "GripperScript.inputs:execIn",
                ),
            ],
        },
    )

    # ---------------------------------------
    # 5. Create dynamic Script Node inputs
    # ---------------------------------------

    move_script_node = og.Controller.node(
        f"{graph_path}/MoveScript"
    )

    gripper_script_node = og.Controller.node(
        f"{graph_path}/GripperScript"
    )

    move_attr = og.Controller.create_attribute(
        move_script_node,
        "delta",
        "vectord[3]",
        og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    )

    gripper_attr = og.Controller.create_attribute(
        gripper_script_node,
        "open",
        "bool",
        og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT,
    )

    if not move_attr.is_valid() or not gripper_attr.is_valid():
        raise RuntimeError(
            "Failed to create Script Node input attributes"
        )

    og.Controller.connect(
        f"{graph_path}/Move.outputs:linearVelocity",
        f"{graph_path}/MoveScript.inputs:delta",
    )

    og.Controller.connect(
        f"{graph_path}/Gripper.outputs:data",
        f"{graph_path}/GripperScript.inputs:open",
    )

    # ---------------------------------------
    # 6. Define Script Node Python functions
    # ---------------------------------------

    move_script_code = '''
import franka_teleop_shared as shared

def compute(db):
    delta = tuple(float(x) for x in db.inputs.delta)
    shared.move_queue.append(delta)
    return True
'''

    gripper_script_code = '''
import franka_teleop_shared as shared

def compute(db):
    shared.gripper_queue.append(
        bool(db.inputs.open)
    )
    return True
'''

    og.Controller.set(
        og.Controller.attribute(
            f"{graph_path}/MoveScript.inputs:script"
        ),
        move_script_code,
    )

    og.Controller.set(
        og.Controller.attribute(
            f"{graph_path}/GripperScript.inputs:script"
        ),
        gripper_script_code,
    )

    print(
        "[OK] ROS 2 OmniGraph created",
        flush=True,
    )

    # ---------------------------------------
    # 7. Simulation loop
    # ---------------------------------------

    print(
        "[READY] Waiting for keyboard commands...",
        flush=True,
    )

    while simulation_app.is_running():

        if not world.is_playing():
            world.step(render=True)
            continue

        # Consume each movement command once.
        while shared.move_queue:
            delta = np.asarray(
                shared.move_queue.popleft(),
                dtype=np.float64,
            )

            if not np.all(np.isfinite(delta)):
                print(
                    "[WARN] Invalid delta ignored",
                    flush=True,
                )
                continue

            delta = np.clip(
                delta,
                -0.025,
                0.025,
            )

            target = np.clip(
                target + delta,
                limits_min,
                limits_max,
            )

            print(
                f"[MOVE] delta={np.round(delta, 3)} "
                f"target={np.round(target, 3)}",
                flush=True,
            )

        # Consume gripper commands.
        while shared.gripper_queue:
            open_gripper = shared.gripper_queue.popleft()

            if open_gripper:
                franka.gripper.open()
                print("[GRIPPER] OPEN", flush=True)
            else:
                franka.gripper.close()
                print("[GRIPPER] CLOSE", flush=True)

        # RMPFlow runs every simulation frame.
        action = controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=orientation,
        )

        franka.apply_action(action)

        world.step(render=True)

except KeyboardInterrupt:
    print("Interrupted", flush=True)

except Exception:
    import traceback
    traceback.print_exc()

finally:
    simulation_app.close()
