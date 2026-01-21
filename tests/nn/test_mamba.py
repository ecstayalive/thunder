import torch

from thunder.nn.torch import Mamba2Block, MambaBlock


def test_mamba_forward():
    B = 50
    L = 24
    D_MODEL = 64
    D_STATE = 4
    D_EXPEND = 2
    D_CONV = 4

    model = MambaBlock(D_MODEL, d_state=D_STATE, expand=D_EXPEND, device="cuda")
    x = torch.randn(B, L, D_MODEL, device="cuda")
    conv_state = torch.randn([B, D_EXPEND * D_MODEL, D_CONV - 1], device="cuda")
    ssm_state = torch.randn([B, D_EXPEND * D_MODEL, D_STATE], device="cuda")
    y, state = model(x, (conv_state, ssm_state))
    assert (
        y.shape == torch.Size([B, L, D_MODEL])
        and state[0].shape == torch.Size([B, D_EXPEND * D_MODEL, D_CONV - 1])
        and state[1].shape == torch.Size([B, D_EXPEND * D_MODEL, D_STATE])
    )


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
    )
    x = torch.randn(B, L, D_MODEL, device="cuda")
    conv_state = torch.randn([B, D_EXPEND * D_MODEL, D_CONV - 1], device="cuda")
    ssm_state = torch.randn([B, (D_EXPEND * D_MODEL) // D_HEAD, D_HEAD, D_STATE], device="cuda")
    y, state = model(x, (conv_state, ssm_state))
    assert (
        y.shape == torch.Size([B, L, D_MODEL])
        and state[0].shape == torch.Size([B, D_EXPEND * D_MODEL, D_CONV - 1])
        and state[1].shape == torch.Size([B, (D_EXPEND * D_MODEL) // D_HEAD, D_HEAD, D_STATE])
    )
