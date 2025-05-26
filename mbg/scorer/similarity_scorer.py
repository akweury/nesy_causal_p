# Created by MacBook Pro at 23.05.25
import torch
import torch.nn as nn

class ContextualSimilarityScorer(nn.Module):
    def __init__(self, input_dims=None, context_dim=128, hidden_dim=128):
        super().__init__()

        # input_dims: dict with "color", "size", "shape" feature dims
        if input_dims is None:
            input_dims = {"color": 3, "size": 1, "shape": 4}
        self.color_dim = input_dims["color"]
        self.size_dim = input_dims["size"]
        self.shape_dim = input_dims["shape"]
        total_dim = self.color_dim + self.size_dim + self.shape_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(total_dim * 2 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, objects, i, j):
        """
        objects: list of dicts {"color": Tensor, "size": Tensor, "shape": Tensor}
        i, j: indices of the two objects to compare
        returns: similarity score ∈ [0,1]
        """

        def encode(o):
            return torch.cat([o["color"], o["size"], o["shape"]], dim=-1)

        feats = torch.stack([encode(o) for o in objects])  # [N, D]
        context = self.context_encoder(feats.mean(dim=0))  # [context_dim]

        pair_feat = torch.cat([encode(i), encode(j), context], dim=-1)
        return self.pair_encoder(pair_feat)
