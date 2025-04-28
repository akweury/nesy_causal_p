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

        # 这里我们不用 LSTM，只简单做平均：
        self.classifier = nn.Sequential(
            nn.Linear(3 * patch_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode_patch_set(self, patch_sets: torch.Tensor) -> torch.Tensor:
        B, P, L, D = patch_sets.shape
        x = patch_sets.view(B * P * L, D)
        x = self.point_encoder(x)
        x = x.view(B * P, L, -1).flatten(1)
        x = self.patch_encoder(x)
        x = x.view(B, P, -1).mean(dim=1)
        return x  # (B, patch_embed_dim)

    def forward(
            self,
            contour_i: torch.Tensor,
            contour_j: torch.Tensor,
            context_list: torch.Tensor
    ) -> torch.Tensor:
        context_list = context_list.unsqueeze(0)
        try:

            B, num_ctx, P, L, D = context_list.shape
        except ValueError:
            context_list = torch.zeros(1, 1, 6, 16, 2)
            B, num_ctx, P, L, D = context_list.shape
        emb_i = self.encode_patch_set(contour_i)
        emb_j = self.encode_patch_set(contour_j)

        # context 平均
        ctx_flat = context_list.view(B * num_ctx, P, L, D)
        ctx_emb_flat = self.encode_patch_set(ctx_flat)  # (B*num_ctx, C)
        ctx_emb = ctx_emb_flat.view(B, num_ctx, -1)  # (B, num_ctx, C)
        emb_ctx = ctx_emb.mean(dim=1)  # (B, C)

        pair_emb = torch.cat([emb_i, emb_j, emb_ctx], dim=1)
        logit = self.classifier(pair_emb).squeeze(-1)
        return logit
