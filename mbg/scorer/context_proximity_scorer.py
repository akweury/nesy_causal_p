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
            patch_len: int = 4,
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
        contour_i: torch.Tensor,     # (B, 4, 2)
        contour_j: torch.Tensor,     # (B, 4, 2)
        context_list: list           # list of (N_ctx, 4, 2), variable-sized
    ) -> torch.Tensor:
        B = contour_i.size(0)

        # Convert to (B, 1, 4, 2) for encode_patch_set
        contour_i = contour_i.unsqueeze(1)
        contour_j = contour_j.unsqueeze(1)
        emb_i = self.encode_patch_set(contour_i)
        emb_j = self.encode_patch_set(contour_j)

        # Handle context (variable-length across batch)
        emb_ctx_list = []
        for ctx in context_list:
            ctx = ctx.unsqueeze(0)  # (1, N_ctx, 4, 2)
            emb_ctx = self.encode_patch_set(ctx)  # (1, patch_embed_dim)
            emb_ctx_list.append(emb_ctx)
        emb_ctx = torch.cat(emb_ctx_list, dim=0)  # (B, patch_embed_dim)

        pair_emb = torch.cat([emb_i, emb_j, emb_ctx], dim=1)
        logit = self.classifier(pair_emb).squeeze(-1)  # (B,)
        return logit
