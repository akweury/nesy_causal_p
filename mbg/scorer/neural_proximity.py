# Created by MacBook Pro at 24.04.25
# neural_proximity_scorer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralProximityScorer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, patch_len=16, patch_embed_dim=64):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.patch_encoder = nn.Sequential(
            nn.Linear(hidden_dim * patch_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, patch_embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * patch_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output logits
        )

    def encode_patch_set(self, patch_set):
        """
        patch_set: (B, P, L, 2)
        returns: (B, embed_dim)
        """
        B, P, L, D = patch_set.shape
        x = patch_set.view(B * P * L, D)                 # (B*P*L, 2)
        x = self.point_encoder(x)                        # (B*P*L, hidden)
        x = x.view(B * P, L, -1).flatten(start_dim=1)    # (B*P, L*hidden)
        patch_emb = self.patch_encoder(x)                # (B*P, embed_dim)
        patch_emb = patch_emb.view(B, P, -1).mean(dim=1) # (B, embed_dim)
        return patch_emb

    def forward(self, contour_i, contour_j):
        """
        contour_i, contour_j: (B, P, L, 2)
        return: (B,) logits for each pair
        """
        emb_i = self.encode_patch_set(contour_i)         # (B, embed_dim)
        emb_j = self.encode_patch_set(contour_j)         # (B, embed_dim)
        pair_emb = torch.cat([emb_i, emb_j], dim=1)      # (B, 2 * embed_dim)
        logits = self.classifier(pair_emb).squeeze(1)    # (B,)
        return logits
