import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from dataset import RobotDataset
from model import BehaviorCloning


torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "../10_recoder/datasets/demonstrations"
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3

dataset = RobotDataset(DATA_DIR)

# Reconstruct episode boundaries from the original files.
episode_lengths = []
for path in sorted(Path(DATA_DIR).glob("episode_*.npz")):
    with np.load(path) as data:
        episode_lengths.append(len(data["time"]))

assert sum(episode_lengths) == len(dataset)

episode_indices = []
start = 0

for length in episode_lengths:
    episode_indices.append(
        np.arange(start, start + length)
    )
    start += length

rng = np.random.default_rng(42)
order = rng.permutation(len(episode_indices))

n_val = max(1, int(len(order) * 0.2))
val_episodes = order[:n_val]
train_episodes = order[n_val:]

train_indices = np.concatenate(
    [episode_indices[i] for i in train_episodes]
)
val_indices = np.concatenate(
    [episode_indices[i] for i in val_episodes]
)

train_loader = DataLoader(
    Subset(dataset, train_indices.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    Subset(dataset, val_indices.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = BehaviorCloning().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Scale position targets from metres to centimetres
# so that position and gripper losses have comparable magnitudes.
POSITION_SCALE = 100.0


def compute_loss(position, gripper_logits, actions):
    target_position = actions[:, :3] * POSITION_SCALE
    target_gripper = actions[:, 3:4]

    position_loss = F.mse_loss(
        position, target_position
    )

    gripper_loss = F.binary_cross_entropy_with_logits(
        gripper_logits, target_gripper
    )

    total_loss = position_loss + gripper_loss

    return total_loss, position_loss, gripper_loss


best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for obs, actions in train_loader:
        obs = obs.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        position, gripper_logits = model(obs)

        loss, _, _ = compute_loss(
            position, gripper_logits, actions
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(obs)

    train_loss /= len(train_indices)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for obs, actions in val_loader:
            obs = obs.to(device)
            actions = actions.to(device)

            position, gripper_logits = model(obs)

            loss, _, _ = compute_loss(
                position, gripper_logits, actions
            )

            val_loss += loss.item() * len(obs)

    val_loss /= len(val_indices)

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "position_scale": POSITION_SCALE,
                "obs_dim": 18,
            },
            "bc_best.pt"
        )

print("Training complete!")
print("Best validation loss:", best_val_loss)