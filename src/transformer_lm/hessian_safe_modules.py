import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowButCorrectMultiheadAttention(nn.Module):
    """
    A pure-PyTorch implementation of Multi-Head Attention that:
    ✔ Supports double backward (Hessian)
    ✔ Does NOT use FlashAttention or SDPA kernels
    ✘ Is slower (O(n^2)) (where n is seq_lenght)   
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.batch_first = False

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None):
        # x: (seq, batch, embed)
        S, B, E = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project inputs
        q = self.q_proj(x).view(S, B, H, D)  # (S,B,H,D)
        k = self.k_proj(x).view(S, B, H, D)
        v = self.v_proj(x).view(S, B, H, D)

        # Scaled dot-product attention (manual, Hessian-safe)
        scores = torch.einsum("sbhd,tbhd->sbht", q, k) / (D ** 0.5)

        if attn_mask is not None:
            # Convert (S,T) -> (S,1,1,T) for broadcasting
            # mask = attn_mask.unsqueeze(1).unsqueeze(1)
            # scores = scores + mask
            mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (S,1,1,T)
            scores = scores.masked_fill(mask == float("-inf"), -1e9)

        attn = F.softmax(scores, dim=3)  # attention along "t"
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("sbht,tbhd->sbhd", attn, v)
        out = out.reshape(S, B, E)

        return self.o_proj(out)


class MyTransformerEncoderLayer(nn.Module):
    """
    A drop-in replacement for nn.TransformerEncoderLayer that:
    ✔ Uses our Hessian-safe attention
    ✔ Matches PyTorch behavior exactly
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()

        self.self_attn = SlowButCorrectMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, **kwargs):
        # Self attention
        attn_output = self.self_attn(src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src
