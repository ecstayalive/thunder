import pytest
import torch

from thunder.nn.torch import (
    CnnSpec,
    ConsistentNormalHeadSpec,
    GruMlpSpec,
    LinearBlockSpec,
    LstmMlpSpec,
    NormalHeadSpec,
    RecurrentMlpSpec,
)
from thunder.rl.torch.agent.ppo import PpoAgentSpec
from thunder.rl.torch.models import Actor, ActorSpec, Critic, CriticSpec
from thunder.utils.arguments import ArgParser

VECTOR_BACKBONES = [
    GruMlpSpec(out_shape=32, mlp_shape=(16,)),
    LstmMlpSpec(out_shape=32),
    RecurrentMlpSpec(out_shape=32, rnn_type="lstm"),
    LinearBlockSpec(out_shape=32),
]


@pytest.mark.parametrize("spec", VECTOR_BACKBONES, ids=lambda s: type(s).__name__)
def test_backbone_contract_emits_out_shape_and_carry(spec):
    """forward(input, carry) -> (output[..., out_shape], carry)."""
    module = spec.factory(10)
    out, carry = module(torch.randn(2, 5, 10), None)
    assert out.shape == (2, 5, spec.out_shape)


def test_cnn_backbone_folds_time_axis_and_flattens():
    cnn = CnnSpec(out_shape=64).factory((3, 16, 16))
    out, carry = cnn(torch.randn(2, 4, 3, 16, 16), None)
    assert out.shape == (2, 4, 64)
    assert carry is None


def test_actor_wires_backbone_out_shape_into_head():
    spec = ActorSpec(backbone=GruMlpSpec(out_shape=48), head=ConsistentNormalHeadSpec())
    actor = spec.factory({"policy": 10}, action_dim=7)
    assert isinstance(actor, Actor)
    dist, _ = actor({"policy": torch.randn(2, 5, 10)}, None)
    assert actor.dist.ffn.linear_block[0].in_features == spec.backbone.out_shape
    assert dist.mean().shape[-1] == 7


def test_actor_is_backbone_interchangeable():
    """The same ActorSpec accepts any ModelSpec backbone."""
    for backbone in (GruMlpSpec(out_shape=24), LinearBlockSpec(out_shape=24)):
        actor = ActorSpec(backbone=backbone).factory({"policy": 8}, action_dim=3)
        dist, _ = actor({"policy": torch.randn(2, 6, 8)}, None)
        assert dist.mean().shape[-1] == 3


def test_critic_emits_scalar_value():
    critic = CriticSpec().factory({"policy": 10})
    assert isinstance(critic, Critic)
    value, _ = critic({"policy": torch.randn(2, 5, 10)}, None)
    assert value.shape[-1] == 1


def test_actor_obs_keys_select_stream():
    actor = ActorSpec(obs_keys=("state",), backbone=LinearBlockSpec(out_shape=16)).factory(
        {"state": 5, "other": 99}, action_dim=2
    )
    dist, _ = actor({"state": torch.randn(2, 3, 5), "other": torch.randn(2, 3, 99)}, None)
    assert dist.mean().shape[-1] == 2


def test_sequential_composes_blocks_and_threads_carry():
    from thunder.nn.torch import SequentialSpec

    seq = SequentialSpec(blocks=(CnnSpec(out_shape=32), GruMlpSpec(out_shape=48)))
    assert seq.out_shape == 48  # chain out == last block out
    module = seq.factory((3, 16, 16))
    out, carry = module(torch.randn(2, 4, 3, 16, 16), None)
    assert out.shape == (2, 4, 48)
    assert isinstance(carry, tuple) and len(carry) == 2
    assert carry[0] is None  # stateless CNN block
    # carry re-feeds without shape error
    out2, _ = module(torch.randn(2, 4, 3, 16, 16), carry)
    assert out2.shape == (2, 4, 48)


def test_mamba_spec_is_dim_preserving_and_guards():
    from thunder.nn.torch import MambaSpec

    # standalone with mismatched dim raises a clear error
    with pytest.raises(ValueError):
        MambaSpec(out_shape=128).factory(10)
    # composed with a projection builds fine
    from thunder.nn.torch import SequentialSpec

    seq = SequentialSpec(blocks=(LinearBlockSpec(out_shape=64), MambaSpec(out_shape=64)))
    assert seq.out_shape == 64
    assert seq.factory(10) is not None


def test_normal_head_spec_builds_normal_head():
    head = NormalHeadSpec(hidden_features=(16,)).factory(12, 4)
    dist = head(torch.randn(3, 12))
    assert dist.mean().shape[-1] == 4


def test_ppo_spec_defaults():
    spec = PpoAgentSpec()
    assert spec.actor.obs_keys == ("policy",)
    assert spec.actor.backbone.out_shape == 256
    assert spec.critic.backbone.out_shape == 1
    assert spec.compile is True


def test_cli_overrides_nested_polymorphic_spec_leaves():
    spec = ArgParser(PpoAgentSpec).parse(
        [
            "--actor.backbone.out-shape", "128",
            "--actor.head.init-std", "0.3",
            "--critic.obs-keys", "critic",
        ]
    )
    assert spec.actor.backbone.out_shape == 128
    assert abs(spec.actor.head.init_std - 0.3) < 1e-9
    assert spec.critic.obs_keys == ("critic",)
