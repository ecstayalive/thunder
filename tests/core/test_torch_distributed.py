"""Distributed-training wiring for the torch backend.

Covers the single-process invariants (no-op DistributedContext, Executor
auto-detection, metric-reduction no-op) and a real 2-rank gloo group that exercises
topology resolution, live-group auto-detection, and ``all_reduce`` averaging.
"""

import os

import pytest
import torch


def test_distributed_context_is_noop_without_torchrun():
    from thunder.core import DistributedContextManager, DistributedSpec

    with DistributedContextManager() as info:
        assert isinstance(info, DistributedSpec)
        assert info.enabled is False
        assert info.rank == 0 and info.local_rank == 0 and info.world_size == 1
        assert info.is_main is True


def test_executor_autodetects_no_process_group():
    from thunder.core import Executor

    # No group initialized -> distributed must resolve to False.
    assert Executor(precision="fp32").distributed is False
    # Explicit override is still honored.
    assert Executor(precision="fp32", distributed=False).distributed is False


def test_reduce_mean_is_noop_when_not_distributed():
    from thunder.core import Executor

    ex = Executor(precision="fp32")
    metrics = {"kl": torch.tensor(0.5), "scalar": 7}
    ex.reduce_mean(metrics)
    assert torch.equal(metrics["kl"], torch.tensor(0.5))
    assert metrics["scalar"] == 7


def _dist_worker(rank: int, world_size: int, port: int):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from thunder.core import DistributedContextManager, Executor

    with (
        DistributedContextManager() as info
    ):  # CUDA hidden by the parent -> cpu/gloo branch
        assert info.enabled and info.rank == rank and info.world_size == world_size
        assert info.device == "cpu"

        ex = Executor(precision="fp32", device="cpu")
        assert ex.distributed is True  # auto-detects the live group

        metrics = {"kl": torch.tensor(float(rank)), "scalar": 7}
        ex.reduce_mean(metrics)
        expected = sum(range(world_size)) / world_size
        assert abs(metrics["kl"].item() - expected) < 1e-6
        assert metrics["scalar"] == 7  # non-tensors untouched


def test_two_rank_gloo_group_reduces_metrics():
    # Hide CUDA so each spawned rank takes the cpu/gloo branch; a single-GPU host
    # cannot otherwise bind two ranks to distinct devices.
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch.multiprocessing as mp

    mp.spawn(_dist_worker, args=(2, 29557), nprocs=2, join=True)


def _ddp_init_worker(rank: int, world_size: int, port: int):
    import torch.nn as nn

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from torch.nn.parallel import DistributedDataParallel as DDP

    from thunder.core import DistributedContextManager, Executor, OptimGroupSpec
    from thunder.core.module import ModelPack

    class Normalizer(nn.Module):
        """A running-stats module: state in buffers, no gradient parameters."""

        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.zeros(4))

        def forward(self, x):
            return x - self.mean

    class TinyActor(nn.Module):
        """A module with an inference helper beyond ``forward``, like Actor."""

        def __init__(self):
            super().__init__()
            self.ffn = nn.Linear(4, 2)

        def forward(self, x):
            return self.ffn(x)

        def explore(self, x):
            return self.ffn(x) + 1.0

    with DistributedContextManager():
        ex = Executor(precision="fp32", device="cpu")
        models = ModelPack(
            actor=TinyActor(), critic=nn.Linear(4, 1), normalizer=Normalizer()
        )
        ctx = ex.init(
            models,
            {"ppo": OptimGroupSpec(targets=("actor", "critic"), optimizer="AdamW", lr=1e-3)},
        )
        # Trainable modules are wrapped; the param-less normalizer is left alone
        # (DDP rejects a module with no gradient-requiring parameter).
        assert isinstance(ctx.models.actor, DDP)
        assert isinstance(ctx.models.critic, DDP)
        assert not isinstance(ctx.models.normalizer, DDP)
        # Custom methods/attributes delegate through the wrapper to the module,
        # so rollout-style calls (`actor.explore(...)`) work on the wrapped model.
        x = torch.randn(3, 4)
        with torch.inference_mode():
            assert torch.allclose(ctx.models.actor.explore(x), x @ ctx.models.actor.ffn.weight.T + ctx.models.actor.ffn.bias + 1.0)
        # `forward` still routes through DDP (gradient sync path stays intact).
        assert type(ctx.models.actor).__mro__[1] is DDP


def test_ddp_init_skips_parameterless_modules():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch.multiprocessing as mp

    mp.spawn(_ddp_init_worker, args=(2, 29558), nprocs=2, join=True)
