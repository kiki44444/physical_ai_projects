import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    def __init__(self, data_dir):
        observations = []
        actions = []

        files = sorted(Path(data_dir).glob("episode_*.npz"))

        for file in files:
            with np.load(file) as data:
                n = len(data["time"])

                goal = np.tile(np.array([0.40, 0.30, 0.025]), (n, 1))
                obs = np.concatenate(
                    [
                        data["joint_positions"],
                        data["ee_position"],
                        data["cube_position"],
                        goal,
                    ],
                    axis=1
                )

                action = data["action"]

                observations.append(obs)
                actions.append(action)

        if not observations:
            raise ValueError("No demonstration files found")

        self.observations = torch.tensor(np.concatenate(observations), dtype=torch.float32)
        self.actions = torch.tensor(np.concatenate(actions), dtype=torch.float32)

        print("Observations:", self.observations.shape)
        print("Actions:", self.actions.shape)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]