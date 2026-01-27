import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

import thunder.nn.torch.ops as ops
from thunder.nn.torch.mapping import ACTIVATION_CLS_NAME


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

    def forward(
        self, input: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input: the input tensor, time series data
                :shape: `[B, L, D]`
            state: conv_state and ssm_state
        ED: The abbreviation "ED" stands for Expanded Dimension
        """
        if state is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = state
        # _, _, _ = input.shape
        xz = self.in_proj(input)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        x = x.transpose(1, 2)  # (B, ED, L)
        x, last_conv_state = self._conv_forward(x, conv_state)
        x = x.transpose(1, 2)  # (B, L, ED)
        x = self.activate_layer(x)
        y, last_ssm_state = self.ssm(x, z, ssm_state)
        output = self.out_proj(y)
        return output, (last_conv_state, last_ssm_state)

    def step(
        self,
        input_t: torch.Tensor,  # (B, D) or (B, 1, D)
        state: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a single time step (or a length-1 segment).
        Returns:
            output_t: (B, D)
            new_state: (ssm_state, conv_state)
        """
        if input_t.dim() == 2:
            input_t = input_t.unsqueeze(1)  # (B, 1, D)
        out, last_state = self.forward(input_t, state)  # (B, 1, D)
        return out[:, -1, :], last_state

    def _conv_forward(
        self, x: torch.Tensor, conv_state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply depthwise causal conv with correct streaming semantics.
        Returns:
            y: (B, ED, L)  - conv output for the *new* L tokens
            new_conv_state: (B, ED, k-1)
        """
        B, ED, L = x.shape
        k = self.d_conv
        if k <= 1:
            y = self.conv1d(x)[..., :L]
            new_state = x.new_zeros(B, ED, 0)
            return y, new_state
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        if conv_state is None:
            y_full = self.conv1d(x)  # (B, ED, L + k - 1)
            y = y_full[..., :L]  # causal outputs for this chunk
            if L >= k - 1:
                last_conv_state = x[..., -(k - 1) :]
            else:
                pad_len = (k - 1) - L
                zero_pad = x.new_zeros(B, ED, pad_len)
                last_conv_state = torch.cat([zero_pad, x], dim=-1)
        else:
            # x_full: (B, ED, k-1 + L)
            x_full = torch.cat([conv_state, x], dim=-1)
            # Conv with padding=0: output length is (k-1+L) - k + 1 = L
            y = F.conv1d(
                x_full, weight=weight, bias=bias, stride=1, padding=0, groups=self.d_inner
            )  # (B, ED, L)
            # update conv_state to last k-1 pre-conv inputs
            last_conv_state = x_full[..., -(k - 1) :]

        return y, last_conv_state

    def ssm(
        self, x: torch.Tensor, z: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
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
        y, last_state = ops.selective_scan(
            x, delta, A, B, C, self.D, z, self.dt_proj.bias.float(), state, delta_softplus=True
        )

        y = y.transpose(1, 2)  # (B, L, ED)
        return y, last_state


class Mamba2Block(nn.Module):
    """
    A implementation of the Mamba2Block as described in the paper
    "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
    "https://arxiv.org/abs/2405.21060"

    Args:

    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 128,
        block_len: int = 64,
        A_init_range=(1.0, 16.0),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        official_ops: bool = False,
        activation: str = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.official_ops = official_ops
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.block_len = block_len

        assert (
            self.d_inner % self.headdim == 0
        ), f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
        self.nheads = self.d_inner // self.headdim  # H

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias, **factory_kwargs)

        # depthwise causal conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        #  dt:   (B, L, nheads)
        #  B,C:  (B, L, nheads, d_state)
        self.ssm_proj = nn.Linear(
            self.d_inner,
            self.nheads + 2 * self.nheads * self.d_state,  # dt  # B / C
            bias=False,
            **factory_kwargs,
        )

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)  # (H,)
        self.A_log._no_weight_decay = True

        # D skip per head
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # dt bias initialization：softplus(dt_bias) belong to [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)  # (H,)
        self.dt_bias._no_weight_decay = True
        act_cls = getattr(nn, ACTIVATION_CLS_NAME.get(activation, "SiLU"))
        self.activate_layer = act_cls()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(
        self, input: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input: [B, L, D]
            return: [B, L, D]
        """
        if state is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = state
        # _, L, _ = input.shape
        xz = self.in_proj(input)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        x = x.transpose(1, 2)  # (B, ED, L)
        x, last_conv_state = self._conv_forward(x, conv_state)
        x = x.transpose(1, 2)  # (B, L, ED)
        x = self.activate_layer(x)  # (B, L, ED)

        y, last_ssm_state = self._ssm_mamba2(x, ssm_state)  # (B, L, ED)

        # sigmoid gating
        y = y * torch.sigmoid(z)

        out = self.out_proj(y)  # (B, L, D)
        return out, (last_conv_state, last_ssm_state)

    def _conv_forward(
        self, x: torch.Tensor, conv_state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply depthwise causal conv with correct streaming semantics.
        Returns:
            y: (B, ED, L)  - conv output for the *new* L tokens
            new_conv_state: (B, ED, k-1)
        """
        B, ED, L = x.shape
        k = self.d_conv
        if k <= 1:
            y = self.conv1d(x)[..., :L]
            new_state = x.new_zeros(B, ED, 0)
            return y, new_state
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        if conv_state is None:
            y_full = self.conv1d(x)  # (B, ED, L + k - 1)
            y = y_full[..., :L]  # causal outputs for this chunk
            if L >= k - 1:
                last_conv_state = x[..., -(k - 1) :]
            else:
                pad_len = (k - 1) - L
                zero_pad = x.new_zeros(B, ED, pad_len)
                last_conv_state = torch.cat([zero_pad, x], dim=-1)
        else:
            # x_full: (B, ED, k-1 + L)
            x_full = torch.cat([conv_state, x], dim=-1)
            # Conv with padding=0: output length is (k-1+L) - k + 1 = L
            y = F.conv1d(
                x_full, weight=weight, bias=bias, stride=1, padding=0, groups=self.d_inner
            )  # (B, ED, L)

            # update conv_state to last k-1 pre-conv inputs
            last_conv_state = x_full[..., -(k - 1) :]

        return y, last_conv_state

    def _ssm_mamba2(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project dt, B, C from x. Construct discrete SSM parameters A_t, X_t
        """
        Bsz, L, ED = x.shape
        H = self.nheads
        P = self.headdim
        N = self.d_state
        # dt_raw, B, C
        ssm_params = self.ssm_proj(x)  # (B, L, H + 2*H*N)
        dt_raw, B_raw, C_raw = torch.split(
            ssm_params,
            [H, H * N, H * N],
            dim=-1,
        )
        dt = F.softplus(dt_raw + self.dt_bias.view(1, 1, H))  # (B, L, H)
        B_mat = B_raw.view(Bsz, L, H, N)  # (B, L, H, N)
        C_mat = C_raw.view(Bsz, L, H, N)  # (B, L, H, N)
        X = x.view(Bsz, L, H, P)  # (B, L, H, P)
        # A -> A_t
        A_cont = -torch.exp(self.A_log)  # (H,)
        A_t = dt * A_cont.view(1, 1, H)  # (B, L, H)
        X_disc = X * dt.unsqueeze(-1)  # (B, L, H, P)
        A_disc = A_t  # (B, L, H)
        # SSD + initial_states
        if self.official_ops:
            Y_heads, last_state = mamba_chunk_scan_combined(
                X_disc, dt, A_cont, B_mat, C_mat, self.block_len, return_final_states=True
            )
        else:
            Y_heads, last_state = ops.ssd_minimal(
                X_disc,
                A_disc,
                B_mat,
                C_mat,
                self.block_len,
                initial_states=state,  # (B, H, P, N) or None
            )  # Y_heads: (B, L, H, P)
        # final_state: (B, H, P, N)
        # 5) D skip
        Y_heads = Y_heads + self.D.view(1, 1, H, 1) * X
        y = Y_heads.reshape(Bsz, L, ED)  # (B, L, ED)
        return y, last_state

    def step(
        self,
        input_t: torch.Tensor,  # (B, D) or (B, 1, D)
        state: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-timestep / step-wise inference.
        Returns:
            output_t: (B, D)
            new_state: (ssm_state, conv_state)
        """
        if input_t.dim() == 2:
            input_t = input_t.unsqueeze(1)  # (B, 1, D)
        out, last_state = self.forward(input_t, state)  # (B, 1, D)
        return out[:, -1, :], last_state
