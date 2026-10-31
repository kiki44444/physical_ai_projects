
import numpy as np
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.sensors.camera import Camera
    from PIL import Image

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=np.array([0.50, 0.0, 0.025]),
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.3, 0.0]),
        )
    )

    camera = Camera(
        prim_path="/World/FrontCamera",
        name="front_camera",
        position=np.array([1.2, 0.8, 1.0]),
        resolution=(320, 240),
    )

    world.reset()

    # Orient camera toward the workspace.
    camera.set_world_pose(
        position=np.array([1.2, 0.8, 1.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        camera_axes="world",
    )

    camera.initialize()
    camera.add_rgb_to_frame()

    # Allow render pipeline to warm up.
    for _ in range(30):
        world.step(render=True)

    rgb = camera.get_rgb()

    if rgb is None:
        raise RuntimeError("Camera returned no RGB image")

    rgb = np.asarray(rgb)

    print("RGB shape:", rgb.shape)
    print("RGB dtype:", rgb.dtype)

    Path("camera_outputs").mkdir(exist_ok=True)

    Image.fromarray(
        rgb[:, :, :3].astype(np.uint8)
    ).save("camera_outputs/test_rgb.png")

    print("Saved: camera_outputs/test_rgb.png")

finally:
    simulation_app.close()
