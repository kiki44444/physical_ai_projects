
import argparse
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from vbc_model import VisionBehaviorCloning


class DemoDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)

        self.rgb = d["rgb"]
        self.joints = d["joint_positions"].astype(np.float32)
        self.ee = d["ee_position"].astype(np.float32)
        self.actions = d["action_delta"].astype(np.float32)
        self.gripper = d["gripper_open"].astype(np.float32)

        assert self.joints.shape[1] == 9
        assert len(self.rgb) == len(self.actions)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        image = torch.from_numpy(
            self.rgb[i].copy()
        ).permute(2, 0, 1).float() / 255.0

        state = torch.from_numpy(
            np.concatenate([self.joints[i], self.ee[i]])
        )

        action = torch.from_numpy(self.actions[i] * 100.0)

        # Assumes gripper_command: 1 = OPEN, 0 = CLOSED
        grip = torch.tensor(
            float(self.gripper[i]), dtype=torch.float32
        )

        return image, state, action, grip, i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", required=True)
    p.add_argument("--checkpoint", default="vision_bc_12d_new.pt")
    p.add_argument("--output", default="vision_bc_12d_place_ft.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--place-start", type=int, default=230)
    p.add_argument("--place-end", type=int, default=290)
    p.add_argument("--place-weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = DemoDataset(args.demo)

    weights = np.ones(len(dataset), dtype=np.float64)
    weights[args.place_start:args.place_end] = args.place_weight

    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0
    )

    model = VisionBehaviorCloning(state_dim=12).to(device)

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    # Keep the pretrained visual encoder frozen.
    if hasattr(model, "vision_encoder"):
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for image, state, action, grip, indices in loader:
            image = image.to(device)
            state = state.to(device)
            action = action.to(device)
            grip = grip.to(device)

            optimizer.zero_grad()

            # Expected: position output in centimeters,
            # gripper output as logits.
            pred_action, pred_grip = model(image, state)

            position_error = F.smooth_l1_loss(
                pred_action, action, reduction="none"
            )

            # Give extra attention to downward Z commands.
            downward = (action[:, 2] < -1e-5).float()
            z_weight = 1.0 + 3.0 * downward

            xyz_loss = (
                position_error[:, 0]
                + position_error[:, 1]
                + position_error[:, 2] * z_weight
            ).mean()

            gripper_loss = F.binary_cross_entropy_with_logits(
                pred_grip.reshape(-1),
                grip.reshape(-1)
            )

            loss = xyz_loss + 0.2 * gripper_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(image)

        avg_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} "
            f"loss={avg_loss:.6f}",
            flush=True
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "position_scale": 100.0,
                },
                args.output
            )
            print(f"  saved {args.output}", flush=True)

    print(f"\nFinished. Best sampled training loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
