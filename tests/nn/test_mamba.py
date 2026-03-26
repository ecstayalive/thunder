import torch

from thunder.nn.torch import Mamba2Block, MambaBlock


def _assert_chunk_consistency(model, x, split_idx, atol=1e-4, rtol=1e-4):
    full_y, full_state = model(x, None)
    y0, state0 = model(x[:, :split_idx], None)
    y1, state1 = model(x[:, split_idx:], state0)
    chunk_y = torch.cat([y0, y1], dim=1)

    assert torch.allclose(chunk_y, full_y, atol=atol, rtol=rtol)
    assert torch.allclose(state1[0], full_state[0], atol=atol, rtol=rtol)
    assert torch.allclose(state1[1], full_state[1], atol=atol, rtol=rtol)


def test_mamba_forward():
    B = 50
    L = 24
    D_MODEL = 64
    D_STATE = 4
    D_EXPEND = 2
    D_CONV = 4

    model = MambaBlock(
        D_MODEL, d_state=D_STATE, expand=D_EXPEND, device="cuda", official_ops=False
    ).eval()
    x = torch.randn(B, L, D_MODEL, device="cuda")
    y0, state0 = model(x, None)
    y, state = model(x, state0)
    assert (
        y0.shape == torch.Size([B, L, D_MODEL])
        and y.shape == torch.Size([B, L, D_MODEL])
        and state[0].shape == torch.Size([B, D_EXPEND * D_MODEL, D_CONV - 1])
        and state[1].shape == torch.Size([B, D_EXPEND * D_MODEL, D_STATE])
    )
    _assert_chunk_consistency(model, x, split_idx=L // 2)


def test_mamba2_forward():
    B = 50
    L = 24
    D_MODEL = 128
    D_STATE = 4
    D_EXPEND = 2
    D_HEAD = 128
    D_CONV = 4

    model = Mamba2Block(
        D_MODEL, d_state=D_STATE, expand=D_EXPEND, headdim=D_HEAD, device="cuda", official_ops=False
    ).eval()
    x = torch.randn(B, L, D_MODEL, device="cuda")
    y0, state0 = model(x, None)
    y, state = model(x, state0)
    assert (
        y0.shape == torch.Size([B, L, D_MODEL])
        and y.shape == torch.Size([B, L, D_MODEL])
        and state[0].shape == torch.Size([B, D_EXPEND * D_MODEL + 2 * D_STATE, D_CONV - 1])
        and state[1].shape == torch.Size([B, (D_EXPEND * D_MODEL) // D_HEAD, D_HEAD, D_STATE])
    )
    _assert_chunk_consistency(model, x, split_idx=L // 2)
