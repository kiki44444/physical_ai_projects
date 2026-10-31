
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VisionDataset
from vbc_model import VisionBehaviorCloning


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="vision_bc_overfit.pt")
parser.add_argument("--data-dir", default="vision_demonstrations")
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = VisionDataset(args.data_dir)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=True,
)

model = VisionBehaviorCloning(
    state_dim=10,
    pretrained=False,
    freeze_backbone=True,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

position_scale = float(checkpoint["position_scale"])

pred_deltas = []
pred_open_probs = []

with torch.inference_mode():
    for images, states, actions in loader:
        images = images.to(device)
        states = states.to(device)

        pred_delta, pred_grip = model(images, states)

        pred_deltas.append(
            (pred_delta / position_scale).cpu().numpy()
        )
        pred_open_probs.append(
            torch.sigmoid(pred_grip).cpu().numpy().reshape(-1)
        )

pred_deltas = np.concatenate(pred_deltas)
pred_open_probs = np.concatenate(pred_open_probs)

expert_actions = dataset.actions.numpy()
expert_deltas = expert_actions[:, :3]
expert_open = expert_actions[:, 3]

pred_open = (pred_open_probs >= 0.5).astype(np.float32)

moving = np.linalg.norm(expert_deltas, axis=1) > 1e-6
stationary = ~moving
closed = expert_open < 0.5
opened = ~closed

position_error = np.abs(pred_deltas - expert_deltas)


def report_position(name, mask):
    if mask.sum() == 0:
        print(f"{name}: no samples")
        return

    mae = position_error[mask].mean(axis=0)
    print(
        f"{name}: n={mask.sum()}, "
        f"MAE_xyz_m={np.round(mae, 6)}, "
        f"mean_MAE_m={mae.mean():.6f}"
    )


def report_gripper(name, mask):
    if mask.sum() == 0:
        print(f"{name}: no samples")
        return

    accuracy = (pred_open[mask] == expert_open[mask]).mean()
    print(
        f"{name}: n={mask.sum()}, "
        f"accuracy={accuracy:.2%}, "
        f"mean_pred_open_prob={pred_open_probs[mask].mean():.4f}"
    )


print(f"\n[MODEL] {args.checkpoint}")
print(f"[DATASET] {len(dataset)} frames")

print("\n[POSITION]")
report_position("All", np.ones(len(dataset), dtype=bool))
report_position("Moving", moving)
report_position("Stationary", stationary)

print("\n[GRIPPER]")
report_gripper("All", np.ones(len(dataset), dtype=bool))
report_gripper("Expert CLOSED", closed)
report_gripper("Expert OPEN", opened)

print("\n[GRIPPER TRANSITIONS]")
transitions = np.flatnonzero(np.diff(expert_open) != 0) + 1
for idx in transitions:
    print(
        f"Frame {idx}: "
        f"expert_open={int(expert_open[idx])}, "
        f"pred_open_prob={pred_open_probs[idx]:.4f}"
    )
