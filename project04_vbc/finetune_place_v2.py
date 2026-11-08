
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import VisionDataset
from vbc_model import VisionBehaviorCloning
from train import compute_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="vision_demonstrations")
    p.add_argument("--checkpoint", default="vision_bc_12d_new.pt")
    p.add_argument("--output", default="vision_bc_12d_place_v2.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--place-start", type=int, default=230)
    p.add_argument("--place-end", type=int, default=290)
    p.add_argument("--place-weight", type=float, default=2.0)
    args = p.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = VisionDataset(args.data_dir)

    # This experiment assumes exactly the 397-frame demo.
    if len(dataset) != 397:
        raise ValueError(
            f"Expected 397 samples, got {len(dataset)}. "
            "Use a directory containing only the target demo."
        )

    # Verify gripper targets are binary.
    for _, _, actions in DataLoader(
        dataset, batch_size=64, shuffle=False
    ):
        labels = actions[:, 3]
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(
                "Dataset gripper labels must be 0 or 1."
            )

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

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]
    position_scale = float(
        checkpoint.get("position_scale", 100.0)
    )

    model = VisionBehaviorCloning(
        state_dim=12,
        pretrained=False,
        freeze_backbone=True
    ).to(device)

    model.load_state_dict(state_dict, strict=True)

    # Freeze visual backbone explicitly.
    for param in model.visual_encoder.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    print(f"[MODEL] {args.checkpoint}")
    print(f"[DATA] {len(dataset)} frames")
    print(f"[PLACE] {args.place_start}:{args.place_end}")
    print(f"[WEIGHT] {args.place_weight}")
    print(f"[SCALE] {position_scale}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0.0
        total_pos = 0.0
        total_grip = 0.0
        count = 0

        for images, states, actions in loader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)

            pred_delta, pred_grip = model(images, states)

            loss, pos_loss, grip_loss = compute_loss(
                pred_delta=pred_delta,
                pred_grip=pred_grip,
                actions=actions,
                position_scale=position_scale,
                moving_weight=5.0,
                closed_weight=2.0,
                z_down_weight=3.0,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = images.shape[0]
            count += n
            total += loss.item() * n
            total_pos += pos_loss.item() * n
            total_grip += grip_loss.item() * n

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={total/count:.6f} "
            f"pos={total_pos/count:.6f} "
            f"grip={total_grip/count:.6f}",
            flush=True
        )

        # Save checkpoints for comparison at different epochs.
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        epoch_path = output.with_name(
            f"{output.stem}_ep{epoch:02d}{output.suffix}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "state_dim": 12,
                "position_scale": position_scale,
                "freeze_backbone": True,
                "epochs": epoch,
                "source_checkpoint": args.checkpoint,
            },
            epoch_path
        )

        print(f"[SAVED] {epoch_path}", flush=True)


if __name__ == "__main__":
    main()
