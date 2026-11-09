"""Train a deterministic action-chunk Transformer on automatic Franka demos."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from act_dataset import ACTDataset
from act_model import ActionChunkTransformer


def epoch_pass(model, loader, optimizer, device, gripper_weight):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for images, states, chunks, valid in loader:
        images, states = images.to(device), states.to(device)
        chunks, valid = chunks.to(device), valid.to(device)
        with torch.set_grad_enabled(training):
            pred_pos, pred_grip = model(images, states)
            pos_per_step = (pred_pos - chunks[..., :3]).square().mean(dim=-1)
            grip_per_step = F.binary_cross_entropy_with_logits(
                pred_grip.squeeze(-1), chunks[..., 3], reduction='none')
            denom = valid.sum().clamp(min=1)
            pos_loss = (pos_per_step * valid).sum() / denom
            grip_loss = (grip_per_step * valid).sum() / denom
            loss = pos_loss + gripper_weight * grip_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        b = len(images)
        totals += np.array([loss.item(), pos_loss.item(), grip_loss.item()]) * b
        count += b
    return totals / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='vision_demonstrations_auto')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=20)
    p.add_argument('--position-scale', type=float, default=100.0)
    p.add_argument('--gripper-weight', type=float, default=2.0)
    p.add_argument('--output', default='checkpoints/act_best.pt')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    files = sorted(Path(args.data_dir).glob('episode_*.npz'))
    if len(files) < 5:
        raise RuntimeError(f'Expected >=5 episodes, found {len(files)}')
    order = np.random.default_rng(args.seed).permutation(len(files))
    nval = max(1, round(len(files) * args.val_fraction))
    val_files = [files[i] for i in order[:nval]]
    train_files = [files[i] for i in order[nval:]]
    train_ds = ACTDataset(train_files, args.chunk_size, args.position_scale)
    val_ds = ACTDataset(val_files, args.chunk_size, args.position_scale)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')
    model = ActionChunkTransformer(chunk_size=args.chunk_size).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best = float('inf')
    print(f'[DATA] train episodes={len(train_files)}, val episodes={len(val_files)}; '
          f'train frames={len(train_ds)}, val frames={len(val_ds)}; device={device}', flush=True)
    for epoch in range(1, args.epochs + 1):
        tr = epoch_pass(model, train_loader, optimizer, device, args.gripper_weight)
        va = epoch_pass(model, val_loader, None, device, args.gripper_weight)
        print(f'[EPOCH {epoch:03d}] train={tr[0]:.5f} val={va[0]:.5f} '
              f'val_pos={va[1]:.5f} val_grip={va[2]:.5f}', flush=True)
        if va[0] < best:
            best = float(va[0])
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'state_dim': 12, 'chunk_size': args.chunk_size,
                'position_scale': args.position_scale,
                'gripper_weight': args.gripper_weight,
                'freeze_backbone': True, 'pretrained_backbone': True,
                'epoch': epoch, 'best_val_loss': best,
                'train_episodes': [x.name for x in train_files],
                'val_episodes': [x.name for x in val_files],
            }, output)
            print(f'[SAVE] {output} best_val={best:.5f}', flush=True)
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
