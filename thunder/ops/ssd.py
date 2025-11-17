from typing import Optional

import torch
import torch.nn.functional as F


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """
    Naive segment sum
    x: (..., T)
    return: (..., T, T), [i, j] = sum_{k=j..i} x[..., k] (i >= j), otherwise: -inf
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)  # (..., T)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]  # (..., T, T)
    mask = torch.tril(
        torch.ones(T, T, device=x.device, dtype=torch.bool),
        diagonal=0,
    )
    x_segsum = x_segsum.masked_fill(~mask, float("-inf"))
    return x_segsum


def ssd_minimal(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    block_len: int,
    initial_states: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mamba-2 SSD minimal implementation.
    This implementation support auto padding of the sequence length.
    Args:
        X (torch.Tensor): (B, L ,H, P)
        A (torch.Tensor): (B, L, H)
        B (torch.Tensor): (B, L, H, N)
        C (torch.Tensor): (B, L, H, N)
        block_len:
        initial_states: (B, H, P, N)

    Returns:
        Y: (B, L, H, P)
        last_state: (B, H, P, N)
    """
    Bsz, L, H, P = X.shape
    N = B.size(-1)

    # `L` must be dividable by `block_len`, so auto padding
    pad = (block_len - (L % block_len)) % block_len
    if pad > 0:
        # X: (B, L, H, P) -> (B, L+pad, H, P)
        X = F.pad(X, (0, 0, 0, 0, 0, pad))  # pad the last dimension is L
        # A: (B, L, H) -> (B, L+pad, H)
        A = F.pad(A, (0, 0, 0, pad))
        # B/C: (B, L, H, N) -> (B, L+pad, H, N)
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
        L_pad = L + pad
    else:
        L_pad = L

    assert L_pad % block_len == 0
    n_chunks = L_pad // block_len

    # X: (b, L_pad, h, p) -> (b, c, l, h, p)
    X = X.view(Bsz, n_chunks, block_len, H, P)  # (b, c, l, h, p)
    # A: (b, L_pad, h) -> (b, c, l, h)
    A = A.view(Bsz, n_chunks, block_len, H)  # (b, c, l, h)
    # B/C: (b, L_pad, h, n) -> (b, c, l, h, n)
    B = B.view(Bsz, n_chunks, block_len, H, N)  # (b, c, l, h, n)
    C = C.view(Bsz, n_chunks, block_len, H, N)  # (b, c, l, h, n)

    # A: (b, h, c, l)
    A_h = A.permute(0, 3, 1, 2)  # (b, h, c, l)
    A_cumsum = torch.cumsum(A_h, dim=-1)  # (b, h, c, l)
    # L: (b, h, c, l, l)
    L_mat = torch.exp(_segsum(A_h))  # (b, h, c, l, l)

    # einsum: "bclhn,bcshn,bhcls,bcshp->bclhp"
    C_e = C  # (b, c, l, h, n)
    B_e = B  # (b, c, s, h, n)  (s=l)
    L_e = L_mat  # (b, h, c, l, s)
    X_e = X  # (b, c, s, h, p)

    Y_diag = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        C_e,
        B_e,
        L_e,
        X_e,
    )  # (b, c, l, h, p)

    # decay_states: (b, h, c, l)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:].clone() - A_cumsum)  # (b, h, c, l)

    # states: (b, c, h, p, n)
    # "bclhn,bhcl,bclhp->bchpn"
    states = torch.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        B,
        decay_states,
        X,
    )  # (b, c, h, p, n)

    if initial_states is None:
        init = states.new_zeros(Bsz, 1, H, P, N)  # (b, 1, h, p, n)
    else:
        if initial_states.dim() == 4:
            init = initial_states.unsqueeze(1)  # (b, 1, h, p, n)
        elif initial_states.dim() == 5 and initial_states.size(1) == 1:
            init = initial_states  # (b, 1, h, p, n)
        else:
            raise ValueError(
                f"`initial_states` shape is expected (B,H,P,N) or (B,1,H,P,N), but gets {initial_states.shape}"
            )

    states = torch.cat([init, states], dim=1)  # (b, c+1, h, p, n)

    last_A = A_cumsum[:, :, :, -1]  # (b, h, c)
    last_A_padded = F.pad(last_A, (1, 0))  # (b, h, c+1)
    decay_chunk = torch.exp(_segsum(last_A_padded))  # (b, h, c+1, c+1)

    # scan between chunk: "bhzc,bchpn->bzhpn"
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, last_state = new_states[:, :-1], new_states[:, -1]  # (b,c,h,p,n), (b,h,p,n)

    # use state generating off-diagonal of the chunk
    state_decay_out = torch.exp(A_cumsum)  # (b, h, c, l)
    # "bclhn,bchpn,bhcl->bclhp"
    Y_off = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        C,
        states,
        state_decay_out,
    )  # (b, c, l, h, p)

    Y = Y_diag + Y_off  # (b, c, l, h, p)
    Y = Y.view(Bsz, L_pad, H, P)  # (b, L_pad, h, p)

    if pad > 0:
        Y = Y[:, :L]

    return Y, last_state  # (B, L, H, P), (B, H, P, N)
