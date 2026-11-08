"""RGB + robot-state dataset for Isaac Sim Visual Behavior Cloning."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights


class VisionDataset(Dataset):
    def __init__(self, data_dir="vision_demonstrations"):
        files = sorted(Path(data_dir).glob("episode_*.npz"))
        if not files:
            raise FileNotFoundError(f"No episode_*.npz found in {data_dir}")
        images, states, actions = [], [], []
        for file in files:
            with np.load(file, allow_pickle=False) as d:
                required = ("rgb", "joint_positions", "ee_position", "gripper_state", "action_delta", "gripper_open")
                missing = [k for k in required if k not in d]
                if missing:
                    raise KeyError(f"{file}: missing {missing}")
                n = len(d["rgb"])
                state = np.concatenate(
                    (
                        d["joint_positions"].astype(np.float32), # 9D
                        d["ee_position"].astype(np.float32), # 3D
                    ),
                    axis=1,
                )

                assert state.shape == (n, 12)
                action = np.concatenate((
                    d["action_delta"].astype(np.float32),
                    d["gripper_open"].astype(np.float32).reshape(-1, 1),
                ), axis=1)
                if d["rgb"].shape != (n, 224, 224, 3) or state.shape != (n, 12) or action.shape != (n, 4):
                    raise ValueError(f"Unexpected shapes in {file}: rgb={d['rgb'].shape}, state={state.shape}, action={action.shape}")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"Nonfinite values in {file}")
                images.append(d["rgb"].astype(np.uint8))
                states.append(state)
                actions.append(action)
        self.images = np.concatenate(images)
        self.states = torch.from_numpy(np.concatenate(states))
        self.actions = torch.from_numpy(np.concatenate(actions))
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        print(f"Loaded {len(self.images)} frames from {len(files)} episodes")
        moving = (np.linalg.norm(self.actions[:, :3].numpy(), axis=1) > 1e-6).mean()
        print(f"Moving frames: {moving:.1%}; gripper-open labels: {self.actions[:, 3].mean().item():.1%}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Torchvision's pretrained ResNet18 preprocessing: resize/crop + ImageNet normalization.
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1)
        return self.transform(image), self.states[index], self.actions[index]
