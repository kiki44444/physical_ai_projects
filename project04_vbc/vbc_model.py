"""Pretrained ResNet18 + proprioception for Visual Behavior Cloning."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class VisionBehaviorCloning(nn.Module):
    def __init__(self, state_dim=13, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.visual_encoder = resnet18(weights=weights)
        self.visual_encoder.fc = nn.Identity()  # 512 features
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad_(False)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.policy = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.position_head = nn.Linear(128, 3)
        self.gripper_head = nn.Linear(128, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.visual_encoder.eval()  # Keep frozen BatchNorm statistics fixed.
        return self

    def forward(self, images, robot_states):
        visual_features = self.visual_encoder(images)
        state_features = self.state_encoder(robot_states)
        fused = torch.cat((visual_features, state_features), dim=1)
        features = self.policy(fused)
        return self.position_head(features), self.gripper_head(features)
