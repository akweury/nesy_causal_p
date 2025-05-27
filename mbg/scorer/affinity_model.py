# Created by MacBook Pro at 27.05.25

import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityPredictor(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.ReLU()
        )

    def forward(self, patches, mask):
        # patches: [B, N, C, H, W]
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)
        embeddings = self.encoder(patches).view(B, N, -1)  # [B, N, D]

        # Normalize
        normed = F.normalize(embeddings, dim=-1)

        # Compute pairwise cosine similarity
        aff_matrix = torch.bmm(normed, normed.transpose(1, 2))  # [B, N, N]

        # Optional: mask invalid regions (if needed)
        return aff_matrix

