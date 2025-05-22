# =============================
# Model: context_proximity_scorer.py
# =============================
# context_proximity_scorer.py
import torch
import torch.nn as nn


class ContextProximityScorer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        patch_len: int = 16,
        patch_embed_dim: int = 64,
    ):
        super().__init__()
        self.patch_embed_dim = patch_embed_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.patch_encoder = nn.Sequential(
            nn.Linear(hidden_dim * patch_len, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, patch_embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * patch_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode_patch_set(self, patch_sets: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of patch sets.
        Input: (B, P, L, D)
        Output: (B, patch_embed_dim)
        """
        B, P, L, D = patch_sets.shape  # e.g., (1, 6, 16, 2)
        x = patch_sets.view(B * P * L, D)
        x = self.point_encoder(x)
        x = x.view(B * P, L, -1).flatten(1)
        x = self.patch_encoder(x)
        x = x.view(B, P, -1).mean(dim=1)  # mean over P patch sets
        return x  # shape: (B, patch_embed_dim)

    def forward(
        self,
        contour_i: torch.Tensor,     # (1, 6, 16, 2)
        contour_j: torch.Tensor,     # (1, 6, 16, 2)
        context_list: torch.Tensor   # (1, N, 6, 16, 2)
    ) -> torch.Tensor:
        B = contour_i.size(0)
        # Encode contour_i and contour_j
        emb_i = self.encode_patch_set(contour_i)  # (1, C)
        emb_j = self.encode_patch_set(contour_j)  # (1, C)


        if context_list.size(1) == 0:
            # N = 0: Use zero vector for context embedding
            ctx_emb = torch.zeros(B, self.patch_embed_dim, device=contour_i.device)
        else:
            # Flatten and encode context
            B, N, P, L, D = context_list.shape
            ctx_flat = context_list.view(B * N, P, L, D)
            ctx_emb = self.encode_patch_set(ctx_flat)  # (B*N, C)
            ctx_emb = ctx_emb.view(B, N, -1).mean(dim=1)  # (B, C)

        # Concatenate embeddings
        pair_emb = torch.cat([emb_i, emb_j, ctx_emb], dim=1)  # (B, 3C)
        logit = self.classifier(pair_emb).squeeze(-1)  # (B,)
        return logit
