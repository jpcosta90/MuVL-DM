from typing import Optional
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided to AttentionPooling")
        b, seq_len, d = tokens.shape
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
        tokens_t = tokens.transpose(0, 1)
        q = self.query.unsqueeze(1).expand(1, b, d)
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else ~(mask.bool())
        else:
            key_padding_mask = None
        attn_out, _ = self.mha(q, tokens_t, tokens_t, key_padding_mask=key_padding_mask)
        return self.ln(attn_out.squeeze(0))