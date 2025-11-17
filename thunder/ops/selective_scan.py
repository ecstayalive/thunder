import torch
import torch.nn.functional as F


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    s0: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure pytorch parallel scan version of `selective scan`.
    The time complexity of this algorithm is approximately `O(logL)`,
    whereas the official implementation's time complexity is approximately `O(1)`.
    For details: http://arxiv.org/abs/2311.06281

    Args:
        x (torch.Tensor): the input tensor. [B, ED, L]
        delta (torch.Tensor): without bias and also softplus. [B, ED, L]
        A (torch.Tensor): [ED, N]
        B (torch.Tensor): [B, N, L]
        C (torch.Tensor): [B, N, L]
        D (torch.Tensor): [ED, ]
        z: [B, ED, L]
        delta_bias: [ED, ]
        s0: [B, ED, N]
        delta_softplus:
    Notice:
        - hidden state: s_t shape is [B, ED, N]
        - The last dimension stand for sequence dimension `L`
    """

    Bsz, ED, L = x.shape
    N = A.shape[1]
    # delta -> dt
    # delta: (B, ED, L)
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, ED, 1)  # broadcast to (B, ED, L)
    if delta_softplus:
        dt = F.softplus(delta)  # (B, ED, L)
    else:
        dt = delta
    # constructing the step-by-step affine coefficients a_t and b_t
    # for every (b, ed, n), there are：s_t = a_t * s_{t-1} + b_t
    # A_bar_t = exp(dt * A)  -> a_t
    # B_bar_t = dt * B_t
    # b_t = B_bar_t * x_t
    # x: (B, ED, L)
    # dt: (B, ED, L)
    # A:  (ED, N)
    # B:  (B, N, L)
    # dt -> (B, ED, 1, L)
    dt4 = dt.unsqueeze(2)  # (B, ED, 1, L)
    # A  -> (1, ED, N, 1)
    A4 = A.unsqueeze(0).unsqueeze(-1)  # (1, ED, N, 1)

    # a_t: (B, ED, N, L)
    a = torch.exp(dt4 * A4)  # (B, ED, N, L)

    # B_t: (B, N, L) -> (B, 1, N, L)
    B4 = B.unsqueeze(1)  # (B, 1, N, L)
    # x_t: (B, ED, L) -> (B, ED, 1, L)
    x4 = x.unsqueeze(2)  # (B, ED, 1, L)

    # b_t: (B, ED, N, L)
    b = dt4 * B4 * x4  # (B, ED, N, L)

    # parallel prefix-scan to calculate A_prefix[t], B_prefix[t]
    # T_t(s) = a_t * s + b_t
    # T_t ... T_0(s) = A_prefix[t] * s + B_prefix[t]
    A_prefix = a.clone()
    B_prefix = b.clone()

    k = 0
    while (1 << k) < L:
        shift = 1 << k  # 2^k
        # the right half is the current step t, the left half is t - shift.
        A_right = A_prefix[..., shift:]  # (B, ED, N, L-shift)
        B_right = B_prefix[..., shift:]
        A_left = A_prefix[..., :-shift]  # (B, ED, N, L-shift)
        B_left = B_prefix[..., :-shift]
        # T_right ∘ T_left
        # (A2, B2) ∘ (A1, B1) = (A2*A1, A2*B1 + B2)
        A_new = A_prefix.clone()
        B_new = B_prefix.clone()
        A_new[..., shift:] = A_right * A_left
        B_new[..., shift:] = A_right * B_left + B_right
        A_prefix, B_prefix = A_new, B_new
        k += 1
    # use A_prefix, B_prefix + s0 getting s_t
    if s0 is None:
        s0 = x.new_zeros(Bsz, ED, N)  # (B, ED, N)
    s0_expanded = s0.unsqueeze(-1)  # (B, ED, N, 1)
    # s_all[t] = A_prefix[t] * s0 + B_prefix[t]
    s_all = A_prefix * s0_expanded + B_prefix  # (B, ED, N, L)
    # use C_t, D, z to get y_t
    # C: (B, N, L) -> (B, 1, N, L)
    C4 = C.unsqueeze(1)  # (B, 1, N, L)
    # s_all * C4: (B, ED, N, L)
    # sum on N -> (B, ED, L)
    y = (s_all * C4).sum(dim=2)  # (B, ED, L)
    # D skip: D: (ED,) -> (1, ED, 1)
    y = y + D.view(1, ED, 1) * x  # (B, ED, L)
    if z is not None:
        y = y * z  # (B, ED, L)
    # state: (B, ED, N)
    last_state = s_all[..., -1]  # (B, ED, N)
    return y, last_state
