"""Simplified deterministic Transformer action chunker (not original CVAE ACT)."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ActionChunkTransformer(nn.Module):
    def __init__(self, state_dim=12, chunk_size=20, d_model=256, nhead=8,
                 num_layers=3, dropout=0.1, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.chunk_size = chunk_size
        self.freeze_backbone = freeze_backbone
        self.visual_encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.visual_encoder.fc = nn.Identity()
        if freeze_backbone:
            for p in self.visual_encoder.parameters():
                p.requires_grad_(False)
        self.visual_projection = nn.Linear(512, d_model)
        self.state_projection = nn.Sequential(nn.Linear(state_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.modality_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.query_embedding = nn.Embedding(chunk_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.position_head = nn.Linear(d_model, 3)
        self.gripper_head = nn.Linear(d_model, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.visual_encoder.eval()
        return self

    def forward(self, images, states):
        features = self.visual_encoder(images)
        vision_token = self.visual_projection(features) + self.modality_embedding[0]
        state_token = self.state_projection(states) + self.modality_embedding[1]
        memory = torch.stack([vision_token, state_token], dim=1)
        queries = self.query_embedding.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.position_head(decoded), self.gripper_head(decoded)
