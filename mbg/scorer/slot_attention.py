# Created by MacBook Pro at 27.05.25
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContourWithCenterEncoder(nn.Module):
    def __init__(self, input_dim=192, out_dim=192):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, emb):
        """
        emb: [B, N, 192] — flattened contour (assumed normalized xy pairs)
        """
        center_x = emb[:, :, ::2].mean(dim=2, keepdim=True)  # x̄: mean of even indices
        center_y = emb[:, :, 1::2].mean(dim=2, keepdim=True)  # ȳ: mean of odd indices
        centers = torch.cat([center_x, center_y], dim=2)      # [B, N, 2]
        enriched = torch.cat([emb, centers], dim=2)           # [B, N, 194]
        return self.project(enriched)                         # [B, N, out_dim]


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] tensor of object embeddings
        Returns:
            slots: [B, K, D]
            attn: [B, N, K] attention weights (soft group assignments)
        """
        B, N, D = x.shape
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_logsigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        x = self.norm_input(x)
        k, v = self.to_k(x), self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)

            attn_logits = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn.transpose(1, 2)  # [B, K, D], [B, N, K]


