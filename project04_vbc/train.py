
"""Sanity-check overfitting on a small Visual BC demonstration dataset.

Experiments:
    E0: moving_weight=1, closed_weight=1 (baseline)
    E1: moving_weight=5, closed_weight=1
    E2: moving_weight=1, closed_weight=2
    E3: moving_weight=5, closed_weight=2
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import VisionDataset
from vbc_model import VisionBehaviorCloning


def compute_loss(
    pred_delta,
    pred_grip,
    actions,
    position_scale,
    moving_weight,
    closed_weight,
):
    # Position targets: metres -> centimetres
    target_delta = actions[:, :3] * position_scale

    # Detect moving frames in original metre units
    moving = torch.linalg.norm(actions[:, :3], dim=1) > 1e-6

    pos_weights = torch.where(
        moving,
        torch.full_like(moving, moving_weight, dtype=target_delta.dtype),
        torch.ones_like(moving, dtype=target_delta.dtype),
    )

    # Per-sample MSE, averaged across X/Y/Z
    pos_per_sample = F.mse_loss(
        pred_delta,
        target_delta,
        reduction="none",
    ).mean(dim=1)

    # Normalize by total sample weights
    pos_loss = (
        pos_weights * pos_per_sample
    ).sum() / pos_weights.sum()

    # Gripper: 1 = open, 0 = closed
    grip_target = actions[:, 3:4]

    grip_per_sample = F.binary_cross_entropy_with_logits(
        pred_grip,
        grip_target,
        reduction="none",
    )

    # Closed frames receive higher weight
    grip_weights = torch.where(
        grip_target < 0.5,
        torch.full_like(grip_target, closed_weight),
        torch.ones_like(grip_target),
    )

    grip_loss = (
        grip_weights * grip_per_sample
    ).sum() / grip_weights.sum()

    total_loss = pos_loss + grip_loss

    return total_loss, pos_loss, grip_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="vision_demonstrations",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--output",
        default="vision_bc_weighted.pt",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--moving-weight",
        type=float,
        default=5.0,
        help="Weight for nonzero position actions.",
    )
    parser.add_argument(
        "--closed-weight",
        type=float,
        default=2.0,
        help="Weight for closed-gripper frames.",
    )

    args = parser.parse_args()

    if args.moving_weight <= 0 or args.closed_weight <= 0:
        parser.error("Loss weights must be positive.")

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = VisionDataset(args.data_dir)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = VisionBehaviorCloning(
        state_dim=10,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    position_scale = 100.0

    print("=" * 60)
    print("Visual Behavior Cloning — Sanity Training")
    print("=" * 60)
    print(f"Device:         {device}")
    print(f"Samples:        {len(dataset)}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Position scale: {position_scale}")
    print(f"Moving weight:  {args.moving_weight}")
    print(f"Closed weight:  {args.closed_weight}")
    print(f"Output:         {args.output}")
    print("=" * 60, flush=True)

    # Dataset action distribution
    moving_count = 0
    stationary_count = 0
    open_count = 0
    closed_count = 0

    for _, _, actions in DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    ):
        moving = torch.linalg.norm(
            actions[:, :3], dim=1
        ) > 1e-6

        moving_count += moving.sum().item()
        stationary_count += (~moving).sum().item()

        open_count += (actions[:, 3] >= 0.5).sum().item()
        closed_count += (actions[:, 3] < 0.5).sum().item()

    print(
        f"[DATA] Moving={moving_count} "
        f"Stationary={stationary_count}"
    )
    print(
        f"[DATA] Open={open_count} "
        f"Closed={closed_count}",
        flush=True,
    )

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()

        total_pos = 0.0
        total_grip = 0.0
        total_loss = 0.0

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
                moving_weight=args.moving_weight,
                closed_weight=args.closed_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(images)

            total_loss += loss.item() * batch_size
            total_pos += pos_loss.item() * batch_size
            total_grip += grip_loss.item() * batch_size

        n = len(dataset)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={total_loss/n:.5f} | "
            f"position={total_pos/n:.5f} | "
            f"gripper={total_grip/n:.5f}",
            flush=True,
        )

    # Save checkpoint
    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": model.state_encoder[0].in_features,
            "position_scale": position_scale,
            "freeze_backbone": True,
            "pretrained": not args.no_pretrained,
            "epochs": args.epochs,
            "training_samples": len(dataset),
            "moving_weight": args.moving_weight,
            "closed_weight": args.closed_weight,
        },
        output,
    )

    print(f"\n[SAVED] {output}")
    print(
        "[NOTE] Training loss alone does not establish "
        "closed-loop success or generalization."
    )


if __name__ == "__main__":
    main()
