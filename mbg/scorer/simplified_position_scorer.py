# =============================
# Model: simplified_position_scorer.py
# =============================
# Simplified scorer that only uses x, y positions
import torch
import torch.nn as nn


class SimplifiedPositionScorer(nn.Module):
    """
    A simplified scorer that takes (x, y) positions, RGB colors, and shape labels of objects.
    
    Input shapes:
        - pos_i: (batch, 9) - position, color, and shape of object i (x, y, r, g, b, shape_one_hot[4])
        - pos_j: (batch, 9) - position, color, and shape of object j (x, y, r, g, b, shape_one_hot[4])
        - context_positions: (batch, N, 9) - positions, colors, and shapes of context objects
    
    Output:
        - logits: (batch,) - score for whether objects i and j belong to same group
    
    Dimension masking:
        - mask_dims: list of strings specifying which dimensions to mask during training/testing
        - Options: 'position' (x,y), 'color' (r,g,b), 'shape' (one-hot)
        - Example: ['position'] will zero out x,y dimensions
        - Example: ['position', 'color'] will zero out x,y,r,g,b dimensions
    """
    def __init__(
        self,
        position_dim: int = 9,
        hidden_dim: int = 64,
        context_embed_dim: int = 32,
        mask_dims: list = None,
    ):
        super().__init__()
        self.position_dim = position_dim
        self.context_embed_dim = context_embed_dim
        self.mask_dims = mask_dims if mask_dims is not None else []
        
        # Create mask for dimensions
        # Dimension layout: [x, y, r, g, b, shape_0, shape_1, shape_2, shape_3]
        # Indices: position=0:2, color=2:5, shape=5:9
        self.mask = torch.ones(position_dim)
        print(f"mask_dims: {self.mask_dims}")
        if 'position' in self.mask_dims:
            self.mask[0:2] = 0  # Mask x, y
        if 'color' in self.mask_dims:
            self.mask[2:5] = 0  # Mask r, g, b
        if 'shape' in self.mask_dims:
            self.mask[5:9] = 0  # Mask shape one-hot
        
        # Encoder for individual object positions
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, context_embed_dim),
        )
        
        # Context aggregator
        self.context_aggregator = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, context_embed_dim),
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * context_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        pos_i: torch.Tensor,           # (batch, 9)
        pos_j: torch.Tensor,           # (batch, 9)
        context_positions: torch.Tensor  # (batch, N, 9) where N is number of context objects
    ) -> torch.Tensor:
        """
        Forward pass for the simplified position scorer.
        
        Args:
            pos_i: Position, color, and shape of object i, shape (batch, 9) - (x, y, r, g, b, shape_one_hot[4])
            pos_j: Position, color, and shape of object j, shape (batch, 9) - (x, y, r, g, b, shape_one_hot[4])
            context_positions: Positions, colors, and shapes of context objects, shape (batch, N, 9)
        
        Returns:
            Logits for grouping score, shape (batch,)
        """
        batch_size = pos_i.size(0)
        
        # Apply dimension masking
        mask = self.mask.to(pos_i.device)
        pos_i = pos_i * mask
        pos_j = pos_j * mask
        context_positions = context_positions * mask
        
        # Encode positions of the two objects
        emb_i = self.position_encoder(pos_i)  # (batch, context_embed_dim)
        emb_j = self.position_encoder(pos_j)  # (batch, context_embed_dim)
        
        # Handle context
        if context_positions.size(1) == 0:
            # No context objects
            ctx_emb = torch.zeros(batch_size, self.context_embed_dim, device=pos_i.device)
        else:
            # Encode and aggregate context
            N = context_positions.size(1)
            ctx_flat = context_positions.view(batch_size * N, self.position_dim)
            ctx_encoded = self.context_aggregator(ctx_flat)  # (batch*N, context_embed_dim)
            ctx_encoded = ctx_encoded.view(batch_size, N, self.context_embed_dim)
            ctx_emb = ctx_encoded.mean(dim=1)  # (batch, context_embed_dim) - mean pooling
        
        # Concatenate embeddings
        pair_emb = torch.cat([emb_i, emb_j, ctx_emb], dim=1)  # (batch, 3*context_embed_dim)
        
        # Classify
        logit = self.classifier(pair_emb).squeeze(-1)  # (batch,)
        
        return logit
