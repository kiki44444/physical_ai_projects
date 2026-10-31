
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VisionDataset
from vbc_model import VisionBehaviorCloning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load original training demonstrations
dataset = VisionDataset("vision_demonstrations")
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load trained checkpoint
ckpt = torch.load(
    "vision_bc_sanity.pt",
    map_location="cpu",
    weights_only=True,
)

model = VisionBehaviorCloning(
    state_dim=13,
    pretrained=False,
    freeze_backbone=True,
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.to(device).eval()

position_scale = float(ckpt["position_scale"])

all_true_grip = []
all_pred_grip = []
all_open_prob = []
all_true_delta = []
all_pred_delta = []

with torch.inference_mode():
    for images, states, actions in loader:
        images = images.to(device)
        states = states.to(device)

        pred_pos, pred_logits = model(images, states)

        pred_delta = pred_pos / position_scale
        open_prob = torch.sigmoid(pred_logits).squeeze(-1)

        all_true_grip.append(actions[:, 3].numpy())
        all_pred_grip.append((open_prob >= 0.5).int().cpu().numpy())
        all_open_prob.append(open_prob.cpu().numpy())

        all_true_delta.append(actions[:, :3].numpy())
        all_pred_delta.append(pred_delta.cpu().numpy())

true_grip = np.concatenate(all_true_grip).astype(int)
pred_grip = np.concatenate(all_pred_grip).astype(int)
open_prob = np.concatenate(all_open_prob)

true_delta = np.concatenate(all_true_delta)
pred_delta = np.concatenate(all_pred_delta)

accuracy = np.mean(true_grip == pred_grip)
mae = np.mean(np.abs(true_delta - pred_delta), axis=0)

print("\n=== GRIPPER ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"True closed: {(true_grip == 0).sum()}")
print(f"Predicted closed: {(pred_grip == 0).sum()}")
print(f"True open: {(true_grip == 1).sum()}")
print(f"Predicted open: {(pred_grip == 1).sum()}")

print("\nConfusion matrix (rows=true, columns=predicted)")
print("             Closed   Open")
for label, name in [(0, "Closed"), (1, "Open  ")]:
    print(
        f"{name}       "
        f"{np.sum((true_grip == label) & (pred_grip == 0)):5d} "
        f"{np.sum((true_grip == label) & (pred_grip == 1)):6d}"
    )

print("\nMean predicted open probability:")
print(f"True closed frames: {open_prob[true_grip == 0].mean():.4f}")
print(f"True open frames:   {open_prob[true_grip == 1].mean():.4f}")

print("\n=== POSITION ===")
print("MAE [dx, dy, dz] (meters):", np.round(mae, 6))
print("True delta mean:", np.round(true_delta.mean(axis=0), 6))
print("Pred delta mean:", np.round(pred_delta.mean(axis=0), 6))
