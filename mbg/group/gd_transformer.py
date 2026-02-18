import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class ShapeEmbedding(nn.Module):
    """
    Given symbolic object info of one image, return shape embedding for each object.

    Input fields per object:
        - shape_id: int (0..K-1)
        - angle: float in radians
        - solidity: float
        - contour_feat: optional tensor of shape (N, C_contour)
        - patch_feat:   optional tensor of shape (N, C_patch)

    Output:
        shape_embedding: tensor (N, shape_dim)
    """

    def __init__(self,
                 num_shapes: int,
                 contour_dim: int = 0,
                 hidden_dim: int = 32,
                 shape_dim: int = 16):
        super().__init__()

        # ----------------------------------------------------
        # 1) Shape ID embedding (learned)
        # ----------------------------------------------------
        self.shape_embed = nn.Embedding(num_shapes, hidden_dim)

        # ----------------------------------------------------
        # 2) MLP to unify all components (symbolic + optional)
        # ----------------------------------------------------
        in_dim = hidden_dim + contour_dim
        # shape_embed + contour + patch

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 2 * shape_dim),
            nn.GELU(),
            nn.Linear(2 * shape_dim, shape_dim)
        )

        self.shape_dim = shape_dim
        self.contour_dim = contour_dim

    def forward(self, shape_id, contour_feat=None, patch_feat=None):
        """
        shape_id:    (N,) long
        angle:       (N,) float
        solidity:    (N,) float
        contour_feat: optional (N, contour_dim)
        patch_feat:   optional (N, patch_dim)
        """
        N = shape_id.shape[0]

        # 1. Embedding for shape ID
        e_shape = self.shape_embed(shape_id)           # (N, hidden_dim)

        # 4. Optional contour embedding
        if self.contour_dim == 0:
            contour_feat = torch.zeros(N, 0, device=shape_id.device)
        else:
            assert contour_feat is not None

        # 6. Concatenate all symbolic + neural cues
        x = torch.cat([
            e_shape,            # (N, hidden_dim)
            contour_feat,       # (N, contour_dim)
        ], dim=-1)

        # 7. Final shape embedding (N, shape_dim)
        return self.fc(x)


def contour_to_fd8(contour_xy: torch.Tensor):
    """
    contour_xy: (P,2) float tensor, P contour points normalized to [0,1]
    Returns: (8,) float Fourier descriptor
    """

    # convert to complex signal
    x, y = contour_xy[:, 0], contour_xy[:, 1]
    z = torch.complex(x, y)  # (P,)

    # FFT
    Z = torch.fft.fft(z)     # (P,)

    # Take first 4 non-DC components
    # Z[0] is DC (centroid), we skip it
    fd = Z[1:5]              # shape: (4,)

    # represent each complex number by (real, imag)
    fd_real = fd.real
    fd_imag = fd.imag
    out = torch.cat([fd_real, fd_imag], dim=0)  # (8,)

    # Optional: normalization for scale invariance
    out = out / (Z[1].abs() + 1e-6)

    return out



# ----------------------------------------------------------------------
# 1. Simple MLP block
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 2. Relative Geometry Encoder
#    Produces attention bias b_{ij}
# ----------------------------------------------------------------------
class RelativeGeometry(nn.Module):
    """
    Input: pos[N,2], size[N,1]
    Output: rel_bias[N,N,rel_dim]
    """
    def __init__(self, rel_dim=64):
        super().__init__()
        self.encoder = MLP(in_dim=1 + 1 + 1 + 1, out_dim=rel_dim)  # dist + dx + dy + size_diff = 4
        # distance, dx, dy, size_diff

    def forward(self, pos: Tensor, size: Tensor):
        """
        pos:   (B, N, 2)
        size:  (B, N, 1)
        """
        B, N, _ = pos.shape

        # pairwise diffs
        dx = pos[:, :, None, 0] - pos[:, None, :, 0]  # (B,N,N)
        dy = pos[:, :, None, 1] - pos[:, None, :, 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)

        size_diff  = size[:, :, None, :] - size[:, None, :, :]    # (B,N,N,1)

        geom = torch.cat([
            dist[..., None],   # (B,N,N,1)
            dx[..., None],
            dy[..., None],
            size_diff          # (B,N,N,1)
        ], dim=-1)  # → (B,N,N,1+2+1 = 4)

        rel = self.encoder(geom)  # (B,N,N,rel_dim)
        return rel


# ----------------------------------------------------------------------
# 3. Multi-head Attention with Relative Bias
# ----------------------------------------------------------------------
class RelMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, rel_dim):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

        # Map rel_bias (rel_dim) → scalar attention bias for each head
        self.rel_proj = nn.Linear(rel_dim, num_heads)

    def forward(self, x, rel_bias):
        """
        x: (B, N, D)
        rel_bias: (B, N, N, rel_dim)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # shapes: (B, N, H, Hd)

        # Attention scores
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k) / (self.head_dim ** 0.5)

        # Add relative bias for each head
        # rel_bias → (B, N, N, H)
        rel = self.rel_proj(rel_bias)
        rel = rel.permute(0, 3, 1, 2)  # (B, H, N, N)

        attn = attn + rel
        attn = attn.softmax(dim=-1)

        # Weighted sum
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        return self.out(out)


# ----------------------------------------------------------------------
# 4. Transformer block
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, rel_dim, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = RelMultiHeadAttention(dim, num_heads, rel_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, rel_bias):
        x = x + self.attn(self.ln1(x), rel_bias)
        x = x + self.mlp(self.ln2(x))
        return x


# ----------------------------------------------------------------------
# 5. Entire Grouping Transformer
# ----------------------------------------------------------------------
class GroupingTransformer(nn.Module):
    """
    Input:
        pos   : (B,N,2)
        size  : (B,N,1)
        optional appearance: (B,N,D_app)

    Output:
        affinity matrix: (B,N,N) between objects
    """
    def __init__(self,
                 shape_dim=16,
                 app_dim=0,
                 d_model=128,
                 num_heads=4,
                 depth=4,
                 rel_dim=64):
        super().__init__()

        self.use_app = (app_dim > 0)
        in_dim = 2 + 1 + app_dim  # pos+size+app

        # Object-token encoder
        self.obj_encoder = MLP(in_dim, d_model)

        # Relative geometry encoder
        self.rel_geo = RelativeGeometry(rel_dim=rel_dim)

        # Transformer stack
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, rel_dim)
            for _ in range(depth)
        ])

        # Final grouping head → pairwise affinity
        self.affinity_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, pos, size, appearance=None):
        """
        Returns:
            affinity: (B,N,N)
        """
        B, N, _ = pos.shape

        if appearance is None:
            appearance = torch.zeros(B, N, 0, device=pos.device)

        # 1. Encode object tokens
        x = torch.cat([pos, size, appearance], dim=-1)
        x = self.obj_encoder(x)  # (B,N,D)

        # 2. Relative geometric embedding
        rel = self.rel_geo(pos, size)  # (B,N,N,rel_dim)

        # 3. Transformer layers
        for blk in self.layers:
            x = blk(x, rel)  # each layer uses same relative geometry

        # 4. Pairwise affinity prediction
        # Broadcast object tokens
        xi = x[:, :, None, :].expand(B, N, N, x.size(-1))
        xj = x[:, None, :, :].expand(B, N, N, x.size(-1))

        pair = torch.cat([xi, xj], dim=-1)  # (B,N,N,2*d_model)
        affinity = self.affinity_head(pair).squeeze(-1)

        # Symmetrize
        affinity = 0.5 * (affinity + affinity.transpose(1, 2))

        return affinity
    
    
    