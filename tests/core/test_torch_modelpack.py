"""Checkpoint canonicalization of TorchModelPack.

The executor decorates models at runtime (torch.compile's ``_orig_mod``, DDP's
``module``); the pack's ``state_dict``/``load_state_dict`` must stay independent
of those wrappers so a checkpoint written under any compile/distributed
configuration loads under any other.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from thunder.core import ModelPack


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)

    def forward(self, x):
        return self.backbone(x)


@pytest.fixture
def gloo_group():
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29559")
    dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _wrapped_pack(gloo: bool):
    actor, critic = Tiny(), Tiny()
    actor_w = torch.compile(actor)
    critic_w = torch.compile(critic)
    if gloo:
        actor_w = nn.parallel.DistributedDataParallel(actor_w)
        critic_w = nn.parallel.DistributedDataParallel(critic_w)
    return ModelPack(actor=actor_w, critic=critic_w), actor, critic


def test_state_dict_is_canonical_under_compile_and_ddp(gloo_group):
    pack, actor, _ = _wrapped_pack(gloo=True)
    keys = pack.state_dict().keys()
    assert keys == ModelPack(actor=Tiny(), critic=Tiny()).state_dict().keys()
    assert not any(".module." in k or "._orig_mod." in k for k in keys)
    assert torch.equal(
        pack.state_dict()["actor.backbone.weight"], actor.backbone.weight
    )


def test_wrapped_checkpoint_loads_into_bare_pack(gloo_group):
    wrapped, _, _ = _wrapped_pack(gloo=True)
    bare = ModelPack(actor=Tiny(), critic=Tiny())
    bare.load_state_dict(wrapped.state_dict(), strict=True)
    assert torch.equal(
        bare.actor.backbone.weight, wrapped.state_dict()["actor.backbone.weight"]
    )


def test_legacy_wrapper_prefixed_checkpoint_loads():
    legacy = {
        f"{model}.module._orig_mod.backbone.{kind}": tensor
        for model in ("actor", "critic")
        for kind, tensor in (("weight", torch.randn(4, 4)), ("bias", torch.randn(4)))
    }
    pack = ModelPack(actor=Tiny(), critic=Tiny())
    pack.load_state_dict(legacy, strict=True)
    assert torch.equal(
        pack.actor.backbone.weight, legacy["actor.module._orig_mod.backbone.weight"]
    )


def test_load_into_wrapped_pack_updates_underlying_module(gloo_group):
    wrapped, actor, _ = _wrapped_pack(gloo=True)
    source = ModelPack(actor=Tiny(), critic=Tiny())
    wrapped.load_state_dict(source.state_dict(), strict=True)
    assert torch.equal(actor.backbone.weight, source.actor.backbone.weight)


def test_strict_load_rejects_unknown_model():
    pack = ModelPack(actor=Tiny(), critic=Tiny())
    state = pack.state_dict()
    state["ghost.backbone.weight"] = torch.randn(4, 4)
    with pytest.raises(KeyError, match="ghost"):
        pack.load_state_dict(state, strict=True)
