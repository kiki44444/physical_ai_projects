
import numpy as np
from pathlib import Path

files = sorted(Path("vision_demonstrations").glob("episode_*.npz"))

if not files:
    raise FileNotFoundError("No demonstration files found")

with np.load(files[-1]) as data:
    print("File:", files[-1].name)

    for key in data.files:
        print(f"{key:20s} {data[key].shape}")

    rgb = data["rgb"]
    actions = data["action_delta"]

    print("\nRGB dtype:", rgb.dtype)
    print("RGB range:", rgb.min(), rgb.max())

    moving = np.linalg.norm(actions, axis=1) > 1e-6
    print("Moving frames:", moving.sum(), "/", len(actions))
    print("Moving ratio:", f"{moving.mean() * 100:.2f}%")

    print("All RGB frames aligned:",
          len(rgb) == len(actions) == len(data["joint_positions"]))
