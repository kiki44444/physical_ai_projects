
import numpy as np
import torch
from pathlib import Path

from dataset import RobotDataset
from model import BehaviorCloning

DATA_DIR = "../10_recoder/datasets/demonstrations"
MODEL_PATH = "bc_best.pt"

# Load dataset
dataset = RobotDataset(DATA_DIR)

actions = dataset.actions.numpy()
gripper = actions[:, 3]

print("\n========== GRIPPER DATA ==========")
print("Total samples:", len(gripper))
print("Unique values:", np.unique(gripper, return_counts=True))
print("Min:", gripper.min())
print("Max:", gripper.max())
print("Mean:", gripper.mean())

print("\n========== PER EPISODE ==========")

for path in sorted(Path(DATA_DIR).glob("episode_*.npz")):
    with np.load(path) as data:
        g = data["action"][:, 3]
        values, counts = np.unique(g, return_counts=True)

        print(
            path.name,
            "values:", dict(zip(values.tolist(), counts.tolist())),
            "first:", g[:5],
            "last:", g[-5:],
        )

# Load trained model
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = BehaviorCloning(obs_dim=18)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict on recorded observations
with torch.no_grad():
    logits = []
    for batch in dataset.observations.split(1024):
        _, batch_logits = model(batch)
        logits.append(batch_logits)

    probabilities = torch.sigmoid(
        torch.cat(logits, dim=0)
    ).squeeze(-1).numpy()

print("\n========== MODEL PREDICTIONS ==========")
print("Probability min:", probabilities.min())
print("Probability max:", probabilities.max())
print("Probability mean:", probabilities.mean())

print("Predicted Open (> 0.65):", np.mean(probabilities > 0.65))
print("Predicted Close (< 0.35):", np.mean(probabilities < 0.35))

for label in np.unique(gripper):
    mask = gripper == label
    print(
        f"\nRecorded label {label}:",
        f"samples={mask.sum()}",
        f"mean predicted probability={probabilities[mask].mean():.4f}",
        f"predicted >0.65={np.mean(probabilities[mask] > 0.65):.3f}",
        f"predicted <0.35={np.mean(probabilities[mask] < 0.35):.3f}",
    )
print("\n========== CLOSE STATE POSITIONS ==========")

obs = dataset.observations.numpy()

# Observation layout:
# joint positions: 0:9
# EE position:      9:12
# cube position:   12:15
# goal position:   15:18

ee = obs[:, 9:12]
cube = obs[:, 12:15]

relative = ee - cube

for label in [0.0, 1.0]:
    mask = gripper == label

    print(f"\nGripper label = {label}")
    print("Samples:", mask.sum())
    print("Mean EE - Cube:", relative[mask].mean(axis=0))
    print("Min EE - Cube:", relative[mask].min(axis=0))
    print("Max EE - Cube:", relative[mask].max(axis=0))

    distance_xy = np.linalg.norm(relative[mask, :2], axis=1)

    print("Median XY distance:", np.median(distance_xy))
    print("Median Z difference:", np.median(relative[mask, 2]))

print("\n========== GRIPPER CLOSE TRANSITIONS ==========")

for path in sorted(Path(DATA_DIR).glob("episode_*.npz")):
    with np.load(path) as data:
        actions = data["action"]
        ee = data["ee_position"]
        cube_pos = data["cube_position"]

        gripper_labels = actions[:, 3]

        # Detect Open (1) -> Close (0)
        close_indices = np.where(
            (gripper_labels[:-1] >= 0.5) &
            (gripper_labels[1:] < 0.5)
        )[0] + 1

        print(f"\n{path.name}")

        if len(close_indices) == 0:
            print("No Open -> Close transition found")
            continue

        for idx in close_indices:
            relative = ee[idx] - cube_pos[idx]

            print(
                f"step={idx} "
                f"EE-Cube={np.round(relative, 4)} "
                f"XY_distance={np.linalg.norm(relative[:2]):.4f}m "
                f"Z_difference={relative[2]:.4f}m"
            )