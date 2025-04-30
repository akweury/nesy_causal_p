# Created by MacBook Pro at 28.04.25


# mbg/neural_group_features.py

import torch
import torch.nn as nn
from typing import List
from mbg import mbg_config as param
class PatchSetEncoder(nn.Module):
    """
    Takes (B, P, L, D) contour-patches and returns (B, E) object embeddings.
    Here we share the same architecture as your patch classifier’s encoder.
    """
    def __init__(self, input_dim=2, hidden_dim=64, patch_len=16, embed_dim=64):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.patch_encoder = nn.Sequential(
            nn.Linear(hidden_dim * patch_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, patch_sets: torch.Tensor) -> torch.Tensor:
        # patch_sets: (B, P, L, D)
        B, P, L, D = patch_sets.shape
        x = patch_sets.view(B * P * L, D)                # (B·P·L, D)
        x = self.point_encoder(x)                        # (B·P·L, H)
        x = x.view(B * P, L, -1).flatten(1)               # (B·P, L·H)
        x = self.patch_encoder(x)                        # (B·P, E)
        x = x.view(B, P, -1).mean(1)                     # (B, E)
        return x                                         # one vector per object

class NeuralGroupEncoder(nn.Module):
    """
    Given a set of object embeddings, returns a single group embedding.
    """
    def __init__(self, obj_embed_dim=64, hidden_dim=128, group_embed_dim=128):
        super().__init__()
        self.obj_encoder = PatchSetEncoder(
            input_dim=2,
            hidden_dim=hidden_dim//2,
            patch_len=param.POINTS_PER_PATCH,
            embed_dim=obj_embed_dim
        )
        # after aggregating object embeddings, pass through a small MLP
        self.group_mlp = nn.Sequential(
            nn.Linear(obj_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, group_embed_dim)
        )

    def forward(self, patch_sets: torch.Tensor) -> torch.Tensor:
        """
        patch_sets: (G, P, L, 2) or (1, G, P, L, 2) if batched
        returns: (G',) or (1, group_embed_dim) batched group embedding
        """
        # if unbatched, add batch‐dim
        if patch_sets.dim() == 4:
            patch_sets = patch_sets.unsqueeze(0)          # (1, G, P, L, 2)
        B, G, P, L, D = patch_sets.shape
        # reshape to treat all objects as batch
        objs = patch_sets.view(B * G, P, L, D)           # (B·G, P, L, D)
        obj_embs = self.obj_encoder(objs)                # (B·G, E)
        obj_embs = obj_embs.view(B, G, -1)               # (B, G, E)
        # simple mean‐pool across objects
        grp_feat = obj_embs.mean(1)                      # (B, E)
        h_g = self.group_mlp(grp_feat)                   # (B, group_embed_dim)
        return h_g.squeeze(0) if h_g.size(0)==1 else h_g