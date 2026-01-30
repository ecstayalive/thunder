import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """MultiHead Cross Attention with Output Gate
    Details:
        https://arxiv.org/pdf/1706.03762
        http://arxiv.org/pdf/2505.06708
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
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
        self.kv_dim = kv_dim if kv_dim else embed_dim
        self.dropout = dropout
        self.q_proj = nn.Linear(self.q_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.gate_proj = nn.Linear(self.q_dim, self.embed_dim, bias=True, **factory_kwargs)
        self.kv_proj = nn.Linear(self.kv_dim, 2 * self.embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        nn.init.orthogonal_(self.q_proj.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.kv_proj.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.out_proj.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.gate_proj.weight, math.sqrt(gain))
        nn.init.constant_(self.gate_proj.bias, 1.0)

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (..., L, E_q) Batch ..., Target sequence length, Query Dim
            kv:   (..., S, E_kv) Batch ..., Source sequence length, Key Dim
            attn_mask: Optional mask
        Returns:
            output: (..., L, E)
        """
        q_shape = query.shape
        L = query.size(-2)
        S = kv.size(-2)
        use_causal = self.is_causal and (L == S) and (attn_mask is None)
        if query.dim() > 3:
            query = query.view(-1, L, self.q_dim)
            kv = kv.view(-1, S, self.kv_dim)
        q = self.q_proj(query)
        gate = torch.sigmoid(self.gate_proj(query))
        k, v = torch.chunk(self.kv_proj(kv), 2, -1)
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
        attention_output = attention_output * gate
        output = self.out_proj(attention_output)
        if len(q_shape) > 1:
            return output.view(*q_shape[:-1], self.embed_dim)
        return output


class MultiHeadLinearCrossAttention(nn.Module):
    """
    Non-Causal Multi-Head Linear Cross Attention with Output Gate
    Details:
        https://arxiv.org/pdf/2006.16236
        http://arxiv.org/pdf/2505.06708
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim if q_dim else embed_dim
        self.kv_dim = kv_dim if kv_dim else embed_dim
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
        self.q_proj = nn.Linear(self.q_dim, embed_dim, bias=bias, **factory_kwargs)
        self.gate_proj = nn.Linear(self.q_dim, self.embed_dim, bias=True, **factory_kwargs)
        self.kv_proj = nn.Linear(self.kv_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        nn.init.orthogonal_(self.q_proj.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.kv_proj.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.out_proj.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.gate_proj.weight, math.sqrt(gain))
        nn.init.constant_(self.gate_proj.bias, 1.0)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            query (torch.Tensor): [B, L, D_kv]
            kv (torch.Tensor): [B, S, D_kv]

        Returns:
            torch.Tensor: _description_
        """
        q_shape = query.shape
        q = self.q_proj(query)
        gate = torch.sigmoid(self.gate_proj(query))
        k, v = torch.chunk(self.kv_proj(kv), 2, -1)
        # [B, L, H, D_head] -> [B, H, L, D_head]
        q = q.view(*q_shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.view(*kv.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.view(*kv.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        q = self.feature_map(q)
        k = self.feature_map(k)
        # (Q @ (K.T @ V)) / (Q @ K.sum)
        # K: [B, H, S, D], V: [B, H, S, D] -> [B, H, D, D]
        kv_summary = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2)
        # Z: Q [B, H, L, D] @ K_sum.T [B, H, D, 1] -> [B, H, L, 1]
        z = 1.0 / (torch.matmul(q, k_sum.unsqueeze(-1)) + self.eps)
        # Q [B, H, L, D] @ KV_summary [B, H, D, D] -> [B, H, L, D]
        attn_out = torch.matmul(q, kv_summary)
        attn_out = attn_out * z
        # [B, H, L, D] -> [B, L, H, D] -> [B, L, Embed_Dim]
        attn_out = attn_out.transpose(-3, -2).reshape(*q_shape[:-1], self.embed_dim)
        output = attn_out * gate
        output = self.out_proj(output)
        output = self.dropout(output)
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


class SpatialSoftmaxUncertainty(nn.Module):
    """
    Spatial softmax returning:
      - prob map:      [B, C, H, W]
      - entropy:       [B, C, 1]  (normalized by log(HW), ~ in [0,1])
      - peak_prob:     [B, C, 1]  (max probability, confidence proxy)
      - perplexity:    [B, C, 1]  (= exp(entropy), normalized version also provided)
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-8, normalize_entropy: bool = True):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.normalize_entropy = normalize_entropy

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] logits (unnormalized)
        Returns:
            prob:        [B, C, H, W]
            entropy:     [B, C, 1]  (normalized if normalize_entropy=True)
            peak_prob:   [B, C, 1]
            perplexity:  [B, C, 1]  (normalized if entropy normalized)
        """
        B, C, H, W = x.shape
        hw = H * W
        logits = x.view(B, C, -1) / self.temperature  # [B, C, HW]
        prob_flat = F.softmax(logits, dim=-1)  # [B, C, HW]
        prob = prob_flat.view(B, C, H, W)
        logp = torch.log(prob_flat.clamp_min(self.eps))  # [B, C, HW]
        ent = -(prob_flat * logp).sum(dim=-1, keepdim=True)  # [B, C, 1]
        if self.normalize_entropy:
            ent = ent / math.log(hw)
        peak = prob_flat.max(dim=-1, keepdim=True).values  # [B, C, 1]
        perplexity = torch.exp(ent * (math.log(hw) if self.normalize_entropy else 1.0))
        return prob, ent, peak, perplexity


class SpatialArgSoftmax(nn.Module):
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


class SpatialArgSoftmaxUncertainty(nn.Module):
    """
    Soft-argmax keypoints with uncertainty.
    Returns:
      - keypoints:        [B, C, 2]    expected (x,y) in [-1,1] coords
      - var_xy:           [B, C, 2]    variance of x and y
      - peak:             [B, C, 1]
    """

    def __init__(self, height: int, width: int, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        self.eps = eps

        xs = torch.linspace(-1.0, 1.0, width)
        ys = torch.linspace(-1.0, 1.0, height)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
        coord = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        self.register_buffer("coord", coord)
        self.register_buffer("coord_pow2", coord**2)
        self.coord: torch.Tensor
        self.coord_pow2: torch.Tensor

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] logits (unnormalized)
        Returns:
            keypoints: [B, C, 2]
            var_xy:    [B, C, 2]
            peak:      [B, C, 1]
        """
        B, C, H, W = x.shape
        assert (
            H == self.height and W == self.width
        ), f"got ({H}, {W}), expected ({self.height}, {self.width})"
        logits = x.view(B, C, -1) / self.temperature  # [B, C, HW]
        prob = F.softmax(logits, dim=-1)  # [B, C, HW]
        coord = self.coord.view(-1, 2)  # [HW, 2]
        coord_pow2 = self.coord_pow2.view(-1, 2)
        keypoints = prob @ coord  # [B, C, 2]
        # variance and peak
        ex2 = prob @ coord_pow2  # [B, C, 2]
        var = (ex2 - keypoints**2).clamp_min(0.0)  # [B, C, 2]
        peak = prob.max(dim=-1, keepdim=True).values
        return keypoints, var, peak


class ChannelAttention(nn.Module):
    """ """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """ """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out: [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """Coordinate Attention"""

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, C, H, W]
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] -> [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).permute(0, 1, 3, 2).sigmoid()  # [B, C, 1, W]
        out = identity * a_h * a_w

        return out
