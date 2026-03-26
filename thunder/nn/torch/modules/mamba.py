import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as _OfficialRMSNormGated
except ImportError:
    _OfficialRMSNormGated = None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

import thunder.nn.torch.ops as ops
from thunder.nn.torch.mapping import ACTIVATION_CLS_NAME


class RMSNormGated(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms).to(dtype=x_dtype)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.norm_before_gate:
            y = self._rms_norm(x) * F.silu(z)
        else:
            y = self._rms_norm(x * F.silu(z))
        return y * self.weight


if _OfficialRMSNormGated is not None:
    RMSNormGated = _OfficialRMSNormGated


def _depthwise_causal_conv1d_chunk(
    x: torch.Tensor,
    conv1d: nn.Conv1d,
    conv_state: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked causal depthwise Conv1d with explicit streaming state."""
    batch, channels, seqlen = x.shape
    kernel_size = conv1d.kernel_size[0]

    if kernel_size <= 1:
        y = conv1d(x)[..., :seqlen]
        return y, x.new_zeros(batch, channels, 0)

    weight = conv1d.weight
    bias = conv1d.bias
    if conv_state is None:
        y_full = conv1d(x)
        y = y_full[..., :seqlen]
        if seqlen >= kernel_size - 1:
            last_conv_state = x[..., -(kernel_size - 1) :]
        else:
            pad_len = (kernel_size - 1) - seqlen
            last_conv_state = torch.cat([x.new_zeros(batch, channels, pad_len), x], dim=-1)
    else:
        x_full = torch.cat([conv_state, x], dim=-1)
        y = F.conv1d(
            x_full,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            groups=conv1d.groups,
        )
        last_conv_state = x_full[..., -(kernel_size - 1) :]

    return y, last_conv_state


def _is_triton_kernel_failure(exc: BaseException) -> bool:
    msg = str(exc)
    markers = (
        "computeCapability not supported",
        "PassManager::run failed",
        "Triton",
        "MLIR",
        "tritongpu",
        "ssd_chunk_state",
    )
    return any(marker in msg for marker in markers)


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
        official_ops: bool = True,
        rmsnorm: bool = True,
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
        self.official_ops = official_ops
        self.rmsnorm = rmsnorm
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

        if self.rmsnorm:
            pass  # TODO:

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
        if self.rmsnorm:
            pass
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
            new_state: (conv_state, ssm_state)
        """
        if input_t.dim() == 2:
            input_t = input_t.unsqueeze(1)  # (B, 1, D)
        out, last_state = self.forward(input_t, state)  # (B, 1, D)
        return out[:, -1, :], last_state

    def _conv_forward(
        self, x: torch.Tensor, conv_state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _depthwise_causal_conv1d_chunk(x, self.conv1d, conv_state)

    def ssm(
        self, x: torch.Tensor, z: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = -torch.exp(self.A_log.float())
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = self.dt_proj.weight @ delta.transpose(1, 2)
        x = x.transpose(1, 2)
        B = B.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()
        z = z.transpose(1, 2)
        if self.official_ops and selective_scan_fn is not None and state is None:
            y, last_state = selective_scan_fn(
                x,
                delta,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=True,
            )
        else:
            y, last_state = ops.selective_scan(
                x,
                delta,
                A,
                B,
                C,
                self.D.float(),
                z,
                self.dt_proj.bias.float(),
                state,
                delta_softplus=True,
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
        rmsnorm: bool = True,
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
        self.ngroups = 1
        self._official_ops_disabled = False

        assert (
            self.d_inner % self.headdim == 0
        ), f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
        self.nheads = self.d_inner // self.headdim  # H
        self.rmsnorm = rmsnorm
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads,
            bias=bias,
            **factory_kwargs,
        )

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=conv_dim,
            bias=conv_bias,
            **factory_kwargs,
        )
        act_cls = getattr(nn, ACTIVATION_CLS_NAME.get(activation, "SiLU"))
        self.activate_layer = act_cls()
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        if self.rmsnorm:
            self.norm = RMSNormGated(self.d_inner, norm_before_gate=False, **factory_kwargs)
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
        zxbcdt = self.in_proj(input)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        xBC = xBC.transpose(1, 2)
        xBC, last_conv_state = self._conv_forward(xBC, conv_state)
        xBC = self.activate_layer(xBC.transpose(1, 2))
        y, last_ssm_state = self._ssm_mamba2(xBC, dt, ssm_state)
        if self.rmsnorm:
            y = self.norm(y, z)
        else:
            y = y * F.silu(z)
        out = self.out_proj(y)  # (B, L, D)
        return out, (last_conv_state, last_ssm_state)

    def _conv_forward(
        self, x: torch.Tensor, conv_state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _depthwise_causal_conv1d_chunk(x, self.conv1d, conv_state)

    def _ssm_mamba2(
        self,
        xBC: torch.Tensor,
        dt: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mamba-2 non-fused path with chunked state support.
        """
        Bsz, L, _ = xBC.shape
        H = self.nheads
        P = self.headdim
        N = self.d_state
        dt = F.softplus(dt + self.dt_bias.view(1, 1, H))
        x, B_raw, C_raw = torch.split(
            xBC,
            [self.d_inner, self.ngroups * N, self.ngroups * N],
            dim=-1,
        )
        X = x.view(Bsz, L, H, P)
        B_mat = B_raw.view(Bsz, L, self.ngroups, N)
        C_mat = C_raw.view(Bsz, L, self.ngroups, N)
        A = -torch.exp(self.A_log.float())
        if (
            self.official_ops
            and not self._official_ops_disabled
            and mamba_chunk_scan_combined is not None
        ):
            try:
                Y_heads, last_state = mamba_chunk_scan_combined(
                    X,
                    dt,
                    A,
                    B_mat,
                    C_mat,
                    self.block_len,
                    D=self.D.float(),
                    z=None,
                    dt_bias=None,
                    initial_states=state,
                    dt_softplus=False,
                    return_final_states=True,
                )
            except RuntimeError as exc:
                if not _is_triton_kernel_failure(exc):
                    raise
                self._official_ops_disabled = True
                warnings.warn(
                    "Falling back to the local Mamba-2 SSD implementation because the official "
                    f"Triton kernel failed to compile or launch on this device: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                Y_heads = None
                last_state = None
        else:
            Y_heads = None
            last_state = None

        if Y_heads is None:
            A_disc = dt * A.view(1, 1, H)
            X_disc = X * dt.unsqueeze(-1)
            Y_heads, last_state = ops.ssd_minimal(
                X_disc,
                A_disc.contiguous(),
                B_mat,
                C_mat,
                self.block_len,
                initial_states=state,
            )
            Y_heads = Y_heads + self.D.view(1, 1, H, 1) * X
        y = Y_heads.reshape(Bsz, L, self.d_inner)  # (B, L, d_inner)
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
            new_state: (conv_state, ssm_state)
        """
        if input_t.dim() == 2:
            input_t = input_t.unsqueeze(1)  # (B, 1, D)
        out, last_state = self.forward(input_t, state)  # (B, 1, D)
        return out[:, -1, :], last_state
