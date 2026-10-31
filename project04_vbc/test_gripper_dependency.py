
import numpy as np
import torch

from dataset import VisionDataset
from vbc_model import VisionBehaviorCloning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = VisionDataset("vision_demonstrations")
checkpoint = torch.load(
    "vision_bc_overfit.pt",
    map_location="cpu",
    weights_only=True,
)

model = VisionBehaviorCloning(
    state_dim=13,
    pretrained=False,
    freeze_backbone=True,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Expert가 gripper를 닫는 시점 전후
frames = [50, 60, 66, 67, 68, 80, 100, 150, 178, 179]

print("Frame | Expert | P(open | state=OPEN) | P(open | state=CLOSED)")
print("-" * 65)

with torch.inference_mode():
    for idx in frames:
        image, state, action = dataset[idx]

        image = image.unsqueeze(0).to(device)

        state_open = state.clone()
        state_open[-1] = 1.0

        state_closed = state.clone()
        state_closed[-1] = 0.0

        _, logit_open = model(
            image, state_open.unsqueeze(0).to(device)
        )
        _, logit_closed = model(
            image, state_closed.unsqueeze(0).to(device)
        )

        p_open = torch.sigmoid(logit_open).item()
        p_closed = torch.sigmoid(logit_closed).item()

        print(
            f"{idx:5d} | "
            f"{int(action[-1].item()):6d} | "
            f"{p_open:20.4f} | "
            f"{p_closed:22.4f}"
        )
