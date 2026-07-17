import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from thunder.nn.torch.distributions import NormalHead
from thunder.nn.torch.functional import swiglu
from thunder.nn.torch.modules import Conv2dBlock, LinearBlock, LinearBlockSpec, LstmMlp
from thunder.nn.torch.modules.activation import SwiGLU, resolve_activation


def _linears(block: LinearBlock):
    return [m for m in block.linear_block if isinstance(m, nn.Linear)]


def test_swiglu_matches_manual_gated_silu():
    x = torch.randn(4, 10)
    gate, value = x.chunk(2, dim=-1)

    assert torch.allclose(swiglu(x), F.silu(gate) * value)


def test_swiglu_rejects_odd_feature_dim():
    with pytest.raises(ValueError):
        swiglu(torch.randn(4, 7))


def test_swiglu_module_halves_feature_dim():
    out = SwiGLU()(torch.randn(4, 16))

    assert out.shape == (4, 8)


def test_swiglu_module_declares_halved_dim():
    assert SwiGLU.halves_dim is True


def test_resolve_activation_finds_thunder_swiglu():
    assert resolve_activation("swiglu", allow_halving=True) is SwiGLU
    assert resolve_activation("SwiGLU", allow_halving=True) is SwiGLU


def test_resolve_activation_rejects_halving_by_default():
    with pytest.raises(ValueError, match="halves"):
        resolve_activation("swiglu")


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("leaky_relu", nn.LeakyReLU),
        ("leakyrelu", nn.LeakyReLU),
        ("relu", nn.ReLU),
        ("relu6", nn.ReLU6),
        ("softsign", nn.Softsign),
        ("sigmoid", nn.Sigmoid),
        ("silu", nn.SiLU),
        ("mish", nn.Mish),
        ("tanh", nn.Tanh),
        ("hard_tanh", nn.Hardtanh),
        ("hardtanh", nn.Hardtanh),
    ],
)
def test_resolve_activation_covers_torch_nn_names(name, expected_cls):
    assert resolve_activation(name) is expected_cls


def test_resolve_activation_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown activation"):
        resolve_activation("does_not_exist")


def test_conv_block_rejects_swiglu_activation():
    with pytest.raises(ValueError, match="halves"):
        Conv2dBlock((3, 8, 8), channels=(4,), kernel_sizes=(3,), activation="swiglu")


def test_swiglu_exported_from_modules_namespace():
    import thunder.nn.torch.modules as modules

    assert modules.SwiGLU is SwiGLU
    assert modules.resolve_activation is resolve_activation


def test_linear_block_swiglu_doubles_hidden_projection():
    block = LinearBlock(64, 32, hidden_features=[256], activation="swiglu")

    linears = _linears(block)
    assert (linears[0].in_features, linears[0].out_features) == (64, 512)
    assert (linears[1].in_features, linears[1].out_features) == (256, 32)


def test_linear_block_swiglu_forward_shape():
    block = LinearBlock(64, 32, hidden_features=[256, 128], activation="swiglu")

    assert block(torch.randn(5, 64)).shape == (5, 32)


def test_linear_block_swiglu_activate_output_doubles_final_projection():
    block = LinearBlock(
        64, 32, hidden_features=[256], activation="swiglu", activate_output=True
    )

    linears = _linears(block)
    assert (linears[-1].in_features, linears[-1].out_features) == (256, 64)
    assert block(torch.randn(5, 64)).shape == (5, 32)


def test_linear_block_mish_structure_unchanged():
    block = LinearBlock(64, 32, hidden_features=[256], activation="mish")

    linears = _linears(block)
    assert (linears[0].in_features, linears[0].out_features) == (64, 256)
    assert (linears[1].in_features, linears[1].out_features) == (256, 32)
    assert block(torch.randn(5, 64)).shape == (5, 32)


def test_lstm_mlp_swiglu_forward_shape():
    block = LstmMlp(64, 32, rnn_hidden_size=128, mlp_shape=[256], activation="swiglu")

    out, (h, c) = block(torch.randn(5, 7, 64))
    assert out.shape == (5, 7, 32)


def test_normal_head_swiglu_keeps_mean_std_split():
    head = NormalHead(64, 3, hidden_features=[128], activation="swiglu")

    dist = head(torch.randn(5, 64))
    assert head.ffn.linear_block[-1].out_features == 6
    assert dist.mean().shape == (5, 3)
    assert torch.isfinite(dist.mean()).all()


def test_linear_block_spec_builds_swiglu_block():
    spec = LinearBlockSpec(out_shape=32, hidden_features=(256,), activation="swiglu")
    model = spec.factory(64)

    out, carry = model(torch.randn(5, 64), None)
    assert out.shape == (5, 32)
    assert carry is None
