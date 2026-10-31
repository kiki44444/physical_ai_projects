
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    def __init__(self, data_dir):
        observations = []
        actions = []

        files = sorted(
            Path(data_dir).glob("episode_*.npz")
        )

        if not files:
            raise ValueError(
                f"No demonstration files found in {data_dir}"
            )

        for file in files:
            with np.load(file) as data:
                # Number of recorded frames
                n = len(data["sim_time"])

                # Goal position is constant within an episode
                goal = np.tile(
                    data["goal_position"].reshape(1, 3),
                    (n, 1),
                )

                # Observation
                obs = np.concatenate(
                    [
                        data["joint_positions"],
                        data["ee_position"],
                        data["cube_position"],
                        goal,
                    ],
                    axis=1,
                )

                # Action: dx, dy, dz, gripper_open
                action = np.concatenate(
                    [
                        data["action_delta"],
                        data["gripper_open"].reshape(-1, 1),
                    ],
                    axis=1,
                )

                if obs.shape[1] != 18:
                    raise ValueError(
                        f"{file.name}: expected 18 observation "
                        f"features, got {obs.shape[1]}"
                    )

                if action.shape != (n, 4):
                    raise ValueError(
                        f"{file.name}: unexpected action shape "
                        f"{action.shape}"
                    )

                observations.append(obs)
                actions.append(action)

        self.observations = torch.tensor(
            np.concatenate(observations),
            dtype=torch.float32,
        )

        self.actions = torch.tensor(
            np.concatenate(actions),
            dtype=torch.float32,
        )

        print("Episodes:", len(files))
        print("Observations:", self.observations.shape)
        print("Actions:", self.actions.shape)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
        )
