"""Episode-safe action-chunk dataset for Franka vision demonstrations."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights


class ACTDataset(Dataset):
    def __init__(self, files, chunk_size=20, position_scale=100.0):
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        self.chunk_size = chunk_size
        self.position_scale = position_scale
        self.episodes = []
        self.index = []
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                rgb = d['rgb'].astype(np.uint8)
                joints = d['joint_positions'].astype(np.float32)
                ee = d['ee_position'].astype(np.float32)
                delta = d['action_delta'].astype(np.float32)
                grip = d['gripper_open'].astype(np.float32).reshape(-1, 1)
            n = len(rgb)
            state = np.concatenate([joints, ee], axis=-1)
            actions = np.concatenate([delta, grip], axis=-1)
            if rgb.shape != (n, 224, 224, 3) or state.shape != (n, 12) or actions.shape != (n, 4):
                raise ValueError(f'Invalid shapes in {path}: rgb={rgb.shape}, state={state.shape}, action={actions.shape}')
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                raise ValueError(f'Non-finite data in {path}')
            actions[:, :3] *= position_scale
            ep_idx = len(self.episodes)
            self.episodes.append((rgb, state, actions))
            self.index.extend((ep_idx, t) for t in range(n))
        if not self.index:
            raise ValueError('No frames loaded')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep_idx, t = self.index[i]
        rgb, states, actions = self.episodes[ep_idx]
        image = self.transform(torch.from_numpy(rgb[t].copy()).permute(2, 0, 1))
        state = torch.from_numpy(states[t].copy())
        chunk = np.zeros((self.chunk_size, 4), dtype=np.float32)
        valid = min(self.chunk_size, len(actions) - t)
        chunk[:valid] = actions[t:t+valid]
        mask = np.zeros(self.chunk_size, dtype=np.bool_)
        mask[:valid] = True
        return image, state, torch.from_numpy(chunk), torch.from_numpy(mask)
