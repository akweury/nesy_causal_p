# =============================
# Model: transformer_position_scorer.py
# =============================
# Transformer-based scorer that uses attention over context objects
import torch
import torch.nn as nn


class TransformerPositionScorer(nn.Module):
    """
    A transformer-based scorer that takes (x, y) positions, RGB colors, shape labels, and contours of objects.
    Uses attention mechanism to aggregate context information instead of mean pooling.
    
    Input shapes:
        - pos_i: (batch, 137) - position, color, shape, and contour of object i
                 (x, y, r, g, b, shape_one_hot[4], contour[128])
        - pos_j: (batch, 137) - position, color, shape, and contour of object j
        - context_positions: (batch, N, 137) - positions, colors, shapes, and contours of context objects

    Output:
        - logits: (batch,) - score for whether objects i and j belong to same group
    
    Dimension masking:
        - mask_dims: list of strings specifying which dimensions to mask during training/testing
        - Options: 'position' (x,y), 'color' (r,g,b), 'shape' (one-hot), 'contour' (128 values)
        - Example: ['position'] will zero out x,y dimensions
        - Example: ['position', 'color'] will zero out x,y,r,g,b dimensions
        - Example: ['contour'] will zero out contour features
    """
    def __init__(
        self,
        position_dim: int = 137,  # 9 basic + 128 contour
        hidden_dim: int = 64,
        context_embed_dim: int = 32,
        mask_dims: list = None,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.position_dim = position_dim
        self.context_embed_dim = context_embed_dim
        self.mask_dims = mask_dims if mask_dims is not None else []
        self.num_heads = num_heads
        
        # Create mask for dimensions
        # Dimension layout: [x, y, r, g, b, shape_0, shape_1, shape_2, shape_3, contour[128]]
        # Indices: position=0:2, color=2:5, shape=5:9, contour=9:137
        self.mask = torch.ones(position_dim)
        if 'position' in self.mask_dims:
            self.mask[0:2] = 0  # Mask x, y
        if 'color' in self.mask_dims:
            self.mask[2:5] = 0  # Mask r, g, b
        if 'shape' in self.mask_dims:
            self.mask[5:9] = 0  # Mask shape one-hot
        if 'contour' in self.mask_dims:
            self.mask[9:137] = 0  # Mask contour features

        # Encoder for individual object positions
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, context_embed_dim),
        )
        
        # Context encoder (projects to context_embed_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, context_embed_dim),
        )
        
        # Transformer encoder for context aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention to aggregate context based on pair
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=context_embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
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
        pos_i: torch.Tensor,           # (batch, 137)
        pos_j: torch.Tensor,           # (batch, 137)
        context_positions: torch.Tensor  # (batch, N, 137) where N is number of context objects
    ) -> torch.Tensor:
        """
        Forward pass for the transformer position scorer.
        
        Args:
            pos_i: Position, color, shape, and contour of object i, shape (batch, 137)
                   (x, y, r, g, b, shape_one_hot[4], contour[128])
            pos_j: Position, color, shape, and contour of object j, shape (batch, 137)
            context_positions: Positions, colors, shapes, and contours of context objects, shape (batch, N, 137)

        Returns:
            Logits for grouping score, shape (batch,)
        """
        batch_size = pos_i.size(0)
        
        # Apply dimension masking
        mask = self.mask.to(pos_i.device)
        pos_i = pos_i * mask
        pos_j = pos_j * mask
        # Only apply mask to context if it has objects (avoid broadcast error when N=0)
        if context_positions.size(1) > 0:
            context_positions = context_positions * mask

        # Encode positions of the two objects
        emb_i = self.position_encoder(pos_i)  # (batch, context_embed_dim)
        emb_j = self.position_encoder(pos_j)  # (batch, context_embed_dim)
        
        # Handle context
        if context_positions.size(1) == 0:
            # No context objects
            ctx_emb = torch.zeros(batch_size, self.context_embed_dim, device=pos_i.device)
        else:
            # Encode context objects
            N = context_positions.size(1)
            ctx_flat = context_positions.view(batch_size * N, self.position_dim)
            ctx_encoded = self.context_encoder(ctx_flat)  # (batch*N, context_embed_dim)
            ctx_encoded = ctx_encoded.view(batch_size, N, self.context_embed_dim)  # (batch, N, context_embed_dim)
            
            # Apply transformer encoder to context
            ctx_transformed = self.transformer_encoder(ctx_encoded)  # (batch, N, context_embed_dim)
            
            # Use cross-attention to aggregate context based on the pair (i, j)
            # Query: concatenated embeddings of i and j
            pair_query = (emb_i + emb_j).unsqueeze(1) / 2.0  # (batch, 1, context_embed_dim)
            
            # Attend to context objects
            ctx_attended, _ = self.cross_attention(
                query=pair_query,  # (batch, 1, context_embed_dim)
                key=ctx_transformed,  # (batch, N, context_embed_dim)
                value=ctx_transformed  # (batch, N, context_embed_dim)
            )
            ctx_emb = ctx_attended.squeeze(1)  # (batch, context_embed_dim)
        
        # Concatenate embeddings
        pair_emb = torch.cat([emb_i, emb_j, ctx_emb], dim=1)  # (batch, 3*context_embed_dim)
        
        # Classify
        logit = self.classifier(pair_emb).squeeze(-1)  # (batch,)
        
        return logit
