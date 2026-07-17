import math
from types import SimpleNamespace

import pytest
import torch

from thunder.core.data import Batch
from thunder.nn.torch import BetaHeadSpec, ConsistentNormalHeadSpec, LinearBlockSpec
from thunder.rl.torch.models import ActorSpec
from thunder.rl.torch.operations import CriticLoss, PpoSurrogateLoss
from thunder.utils import Scalar


def _build_ctx(head_spec, obs_dim=6, action_dim=3, num_trajs=2, max_len=4):
    actor = ActorSpec(backbone=LinearBlockSpec(out_shape=16), head=head_spec).factory(
        {"policy": obs_dim}, action_dim
    )
    obs = {"policy": torch.randn(num_trajs, max_len, obs_dim)}
    actions = torch.zeros(num_trajs, max_len, action_dim)  # interior for any support

    dist, _ = actor(obs, None)
    # Old policy differs from current by a constant (event-summed) log-ratio of 0.1.
    old_log_prob = dist.log_prob(actions).detach() - 0.1

    batch = Batch(mask=torch.ones(num_trajs, max_len, dtype=torch.bool))
    batch.obs = obs
    batch.actions = actions
    batch.advantages = torch.randn(num_trajs, max_len)
    batch["log_prob"] = old_log_prob
    # Per-trajectory initial carry lives in cache.initial (set by SplitTraj upstream).
    cache = Batch(initial=Batch(policy_carry=None, critic_carry=None))
    ctx = SimpleNamespace(
        batch=batch, cache=cache, models=SimpleNamespace(actor=actor)
    )
    return ctx


def test_surrogate_loss_kl_is_distribution_agnostic_for_beta():
    loss = PpoSurrogateLoss()
    ctx = _build_ctx(BetaHeadSpec(low=-1.0, high=1.0))

    value, metrics = loss.compute(ctx)

    # k3 estimator with a constant per-element log-ratio of 0.1: ((e^0.1 - 1) - 0.1).
    expected_kl = (math.exp(0.1) - 1.0) - 0.1
    assert torch.isfinite(value)
    assert isinstance(metrics["kl"], Scalar)
    assert torch.allclose(metrics["kl"].value, torch.tensor(expected_kl), atol=1e-5)


def test_surrogate_loss_kl_matches_k3_for_normal():
    loss = PpoSurrogateLoss()
    ctx = _build_ctx(ConsistentNormalHeadSpec())

    _, metrics = loss.compute(ctx)

    expected_kl = (math.exp(0.1) - 1.0) - 0.1
    assert isinstance(metrics["kl"], Scalar)
    assert torch.allclose(metrics["kl"].value, torch.tensor(expected_kl), atol=1e-5)


def test_surrogate_loss_compute_does_not_break_torch_compile_graph():
    dynamo = pytest.importorskip("torch._dynamo")
    loss = PpoSurrogateLoss()
    ctx = _build_ctx(ConsistentNormalHeadSpec())

    dynamo.reset()
    with dynamo.config.patch(capture_dynamic_output_shape_ops=False):
        explanation = dynamo.explain(lambda ctx: loss.compute(ctx))(ctx)

    assert explanation.graph_break_count == 0
    assert explanation.break_reasons == []


def test_critic_loss_compute_does_not_break_torch_compile_graph():
    dynamo = pytest.importorskip("torch._dynamo")

    class TinyCritic(torch.nn.Module):
        def forward(self, obs, carry):
            return obs["policy"].sum(dim=-1, keepdim=True), carry

    batch = Batch(
        obs={"policy": torch.randn(2, 4, 3)},
        values=torch.zeros(2, 4),
        returns=torch.randn(2, 4),
        mask=torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.bool),
    )
    ctx = SimpleNamespace(
        batch=batch,
        cache=Batch(initial=Batch(critic_carry=None)),
        models=SimpleNamespace(critic=TinyCritic()),
    )
    loss = CriticLoss()

    dynamo.reset()
    with dynamo.config.patch(capture_dynamic_output_shape_ops=False):
        explanation = dynamo.explain(lambda ctx: loss.compute(ctx))(ctx)

    assert explanation.graph_break_count == 0
    assert explanation.break_reasons == []


def test_actor_with_beta_head_emits_bounded_actions():
    actor = ActorSpec(
        backbone=LinearBlockSpec(out_shape=24),
        head=BetaHeadSpec(low=-1.0, high=1.0),
    ).factory({"policy": 8}, action_dim=3)

    step = actor.explore({"policy": torch.randn(2, 5, 8)}, None)

    assert step.action.shape == torch.Size([2, 5, 3])
    assert torch.all(step.action > -1.0) and torch.all(step.action < 1.0)
