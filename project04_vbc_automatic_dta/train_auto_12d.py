
"""Train a new 12D Vision Behavior Cloning policy from automatic expert demos."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights

from vbc_model import VisionBehaviorCloning


class VisionDemoDataset(Dataset):
    def __init__(self, files):
        self.images = []
        self.states = []
        self.actions = []

        for path in files:
            with np.load(path, allow_pickle=False) as data:
                rgb = data["rgb"].astype(np.uint8)
                joints = data["joint_positions"].astype(np.float32)
                ee = data["ee_position"].astype(np.float32)
                delta = data["action_delta"].astype(np.float32)
                gripper = data["gripper_open"].astype(np.float32).reshape(-1, 1)

                states = np.concatenate([joints, ee], axis=1)
                actions = np.concatenate([delta, gripper], axis=1)

                n = len(rgb)

                if rgb.shape != (n, 224, 224, 3):
                    raise ValueError(f"Unexpected RGB shape in {path}: {rgb.shape}")

                if states.shape != (n, 12):
                    raise ValueError(f"Unexpected state shape in {path}: {states.shape}")

                if actions.shape != (n, 4):
                    raise ValueError(f"Unexpected action shape in {path}: {actions.shape}")

                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise ValueError(f"Non-finite values in {path}")

                self.images.append(rgb)
                self.states.append(states)
                self.actions.append(actions)

        self.images = np.concatenate(self.images, axis=0)
        self.states = torch.from_numpy(np.concatenate(self.states, axis=0))
        self.actions = torch.from_numpy(np.concatenate(self.actions, axis=0))

        self.transform = ResNet18_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1)
        image = self.transform(image)

        state = self.states[index]
        action = self.actions[index]

        return image, state, action


def compute_loss(model, images, states, actions, position_scale, gripper_weight):
    predicted_delta, gripper_logits = model(images, states)

    target_delta = actions[:, :3] * position_scale
    target_gripper = actions[:, 3:4]

    position_loss = F.mse_loss(predicted_delta, target_delta)
    gripper_loss = F.binary_cross_entropy_with_logits(
        gripper_logits,
        target_gripper
    )

    total_loss = position_loss + gripper_weight * gripper_loss

    return total_loss, position_loss, gripper_loss


def run_epoch(model, loader, optimizer, device, position_scale, gripper_weight):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_position = 0.0
    total_gripper = 0.0
    total_samples = 0

    for images, states, actions in loader:
        images = images.to(device)
        states = states.to(device)
        actions = actions.to(device)

        with torch.set_grad_enabled(training):
            loss, position_loss, gripper_loss = compute_loss(
                model,
                images,
                states,
                actions,
                position_scale,
                gripper_weight
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = len(images)

        total_loss += loss.item() * batch_size
        total_position += position_loss.item() * batch_size
        total_gripper += gripper_loss.item() * batch_size
        total_samples += batch_size

    return {
        "total": total_loss / total_samples,
        "position": total_position / total_samples,
        "gripper": total_gripper / total_samples
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="vision_demonstrations_auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-scale", type=float, default=100.0)
    parser.add_argument("--gripper-weight", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="vision_bc_12d_auto_best.pt")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("episode_*.npz"))

    if len(files) < 5:
        raise RuntimeError(
            f"At least 5 episodes are required. Found {len(files)} in {data_dir}"
        )

    # Episode-level train/validation split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(files))

    num_val = max(1, round(len(files) * args.val_fraction))

    val_files = [files[i] for i in indices[:num_val]]
    train_files = [files[i] for i in indices[num_val:]]

    print(f"[DATA] Total episodes: {len(files)}")
    print(f"[DATA] Train episodes: {len(train_files)}")
    print(f"[DATA] Validation episodes: {len(val_files)}")

    train_dataset = VisionDemoDataset(train_files)
    val_dataset = VisionDemoDataset(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[DEVICE] {device}")
    print(f"[FRAMES] Train: {len(train_dataset)}")
    print(f"[FRAMES] Validation: {len(val_dataset)}")

    # New policy with ImageNet-pretrained visual backbone
    model = VisionBehaviorCloning(
        state_dim=12,
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.position_scale,
            args.gripper_weight
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            args.position_scale,
            args.gripper_weight
        )

        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} "
            f"val_pos={val_metrics['position']:.5f} "
            f"val_grip={val_metrics['gripper']:.5f}",
            flush=True
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "state_dim": 12,
                "position_scale": args.position_scale,
                "freeze_backbone": True,
                "pretrained_backbone": True,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "train_episodes": [p.name for p in train_files],
                "val_episodes": [p.name for p in val_files],
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "gripper_weight": args.gripper_weight
            }

            torch.save(checkpoint, output_path)

            print(
                f"[SAVE] {output_path} "
                f"best_val={best_val_loss:.5f}",
                flush=True
            )

    print("[DONE] Training complete.")
    print(f"[BEST] Validation loss: {best_val_loss:.5f}")
    print(f"[MODEL] {args.output}")


if __name__ == "__main__":
    main()
