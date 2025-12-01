from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim: Optional[int] = None,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal

        self.q_dim = q_dim if q_dim else embed_dim
        self.k_dim = k_dim if k_dim else embed_dim
        self.v_dim = v_dim if v_dim else embed_dim
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.k_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.v_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (..., L, E_q) Batch ..., Target sequence length, Query Dim
            key:   (..., S, E_k) Batch ..., Source sequence length, Key Dim
            value: (..., S, E_v) Batch ..., Source sequence length, Value Dim
            attn_mask: Optional mask

        Returns:
            output: (..., L, E)
        """
        q_shape = query.shape
        L = query.size(-2)
        S = key.size(-2)
        use_causal = self.is_causal and (L == S) and (attn_mask is None)
        if query.dim() > 3:
            query = query.view(-1, L, self.q_dim)
            key = key.view(-1, S, self.k_dim)
            value = value.view(-1, S, self.v_dim)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(-1, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, S, self.num_heads, self.head_dim).transpose(1, 2)
        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(-1, L, self.embed_dim)
        output = self.out_proj(attention_output)
        if len(q_shape) > 1:
            return output.view(*q_shape[:-1], self.embed_dim)
        return output


class SpatialSoftmax(nn.Module):
    """ """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_flat = x_flat / self.temperature
        prob_flat = F.softmax(x_flat, dim=-1)
        prob = prob_flat.view(B, C, H, W)
        return prob


class ExpectSpatialSoftmax(nn.Module):
    """ """

    def __init__(self, height: int, width: int, temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        xs = torch.linspace(-1.0, 1.0, width)
        ys = torch.linspace(-1.0, 1.0, height)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
        coord = torch.stack([xx, yy], dim=-1)
        self.register_buffer("coord", coord)  # [H, W, 2]
        self.coord: torch.Tensor

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            keypoints: [B, C, 2]
        """
        B, C, H, W = x.shape
        assert (
            H == self.height and W == self.width
        ), f"got ({H}, {W}), expected ({self.height}, {self.width})"

        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        x_flat = x_flat / self.temperature
        prob_flat = F.softmax(x_flat, dim=-1)  # [B, C, H*W]
        # coord: [H, W, 2] -> [H*W, 2]
        coord = self.coord.view(-1, 2)  # [H*W, 2]
        # (B, C, H*W) @ (H*W, 2) -> (B, C, 2)
        keypoints = prob_flat @ coord  # [B, C, 2]
        return keypoints
