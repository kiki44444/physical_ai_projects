import torch
import torch.nn as nn


class BehaviorCloning(nn.Module):
    def __init__(self, obs_dim=18, hidden_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # End-effector movement: dx, dy, dz
        self.position_head = nn.Linear(hidden_dim, 3)

        # Gripper: open / close
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.backbone(obs)

        position = self.position_head(features)
        gripper_logits = self.gripper_head(features)

        return position, gripper_logits