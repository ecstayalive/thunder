import torch

from thunder.nn import Mamba2Block, MambaBlock


def test_mamba_forward():
    B = 50
    L = 24
    D_MODEL = 64
    D_STATE = 4
    D_EXPEND = 2

    model = MambaBlock(D_MODEL, d_state=D_STATE, expand=D_EXPEND)
    x = torch.randn(B, L, D_MODEL)
    y, state = model(x)
    assert y.shape == torch.Size([B, L, D_MODEL]) and state.shape == torch.Size(
        [B, D_EXPEND * D_MODEL, D_STATE]
    )


def test_mamba2_forward():
    B = 50
    L = 24
    D_MODEL = 128
    D_STATE = 4
    D_EXPEND = 2
    D_HEAD = 128

    model = Mamba2Block(D_MODEL, d_state=D_STATE, expand=D_EXPEND, headdim=D_HEAD)
    x = torch.randn(B, L, D_MODEL)
    y, state = model(x)
    assert y.shape == torch.Size([B, L, D_MODEL]) and state.shape == torch.Size(
        [B, (D_EXPEND * D_MODEL) // D_HEAD, D_HEAD, D_STATE]
    )
