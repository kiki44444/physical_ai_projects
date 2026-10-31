
import numpy as np
from pathlib import Path
from dataset import RobotDataset

DATA_DIR = "../10_recoder/demonstrations"

# 1. Load dataset
dataset = RobotDataset(DATA_DIR)

# 2. Check shapes
obs, action = dataset[0]

print("\n=== Dataset Check ===")
print("Observation shape:", obs.shape)
print("Action shape:", action.shape)
print("First observation:", obs.numpy())
print("First action:", action.numpy())

# 3. Check action distribution
actions = dataset.actions.numpy()

stationary = np.all(
    np.isclose(actions[:, :3], 0.0),
    axis=1,
)

print("\n=== Action Statistics ===")
print("Total frames:", len(actions))
print(f"Stationary ratio: {stationary.mean():.2%}")
print(f"Moving ratio: {(~stationary).mean():.2%}")

print("Action min:", actions[:, :3].min(axis=0))
print("Action max:", actions[:, :3].max(axis=0))

# 4. Check individual episodes
print("\n=== Episode Statistics ===")

for path in sorted(Path(DATA_DIR).glob("episode_*.npz")):
    with np.load(path) as data:
        delta = data["action_delta"]

        moving = np.any(
            ~np.isclose(delta, 0.0),
            axis=1,
        )

        print(
            f"{path.name}: "
            f"{len(delta)} frames, "
            f"{moving.mean():.1%} moving"
        )
