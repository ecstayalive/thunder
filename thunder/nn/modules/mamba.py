import math

import torch
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from thunder.nn.mapping import ACTIVATION_CLS_NAME

__all__ = ["MambaBlock"]


class MambaBlock(nn.Module):
    """A implementation of the MambaBlock as described in the paper
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    "http://arxiv.org/abs/2312.00752"
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1.0e-4,
        bias: bool = False,
        conv_bias: bool = True,
        kernel_fused_opt: bool = True,
        activation: str = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.kernel_fused_opt = kernel_fused_opt

        # Projection
        self.in_proj = nn.Linear(
            self.d_model, out_features=self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.activate_layer = getattr(nn, ACTIVATION_CLS_NAME[activation])()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # More stable and efficient inverse of softplus
        #  https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit=True

        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the input tensor, time series data
                :shape: `[B, L, D]`
        ED: The abbreviation "ED" stands for Expanded Dimension
        """

        _, seq_len, _ = input.shape
        xz = self.in_proj(input)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[..., :seq_len]  # causal convolution
        x = x.transpose(1, 2)  # (B, L, ED)
        x = self.activate_layer(x)

        y = self.ssm(x, z)
        output = self.out_proj(y)
        return output

    def ssm(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        TODO: Rewrite selective_scan_fn to avoid relying on third-party package
        """
        A = -torch.exp(self.A_log)  # (ED, N)
        # D = self.D  # (ED,)
        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # NB: delta, B, C may need RMSLayerNorm
        #
        delta = self.dt_proj.weight @ delta.transpose(
            1, 2
        )  # (ED, dt_rank)@(B, dt_rank, L)->(B, ED, L)
        x = x.transpose(1, 2)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)
        z = z.transpose(1, 2)
        y = selective_scan_fn(
            x,
            delta,
            A,
            B,
            C,
            self.D,
            z=z,
            delta_softplus=True,
            delta_bias=self.dt_proj.bias.float(),
        )
        y = y.transpose(1, 2)  # (B, L, ED)
        return y
