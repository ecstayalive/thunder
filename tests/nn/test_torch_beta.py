import math

import torch
import torch.distributions as D

from thunder.nn.torch.distributions import (
    BetaHead,
    Normal,
    NormalParams,
    ScaledBeta,
    ScaledBetaParams,
)


def _ref_beta(c1, c0):
    """torch reference Beta(concentration1=c1, concentration0=c0)."""
    return D.Beta(c1, c0)


def test_scaled_beta_samples_stay_within_bounds():
    c1 = torch.full((4, 3), 2.0)
    c0 = torch.full((4, 3), 3.0)
    dist = ScaledBeta(c1, c0, low=-1.0, high=1.0, event_shape=torch.Size([3]))

    sample, log_prob = dist.rsample(torch.Size([8]))

    assert sample.shape == torch.Size([8, 4, 3])
    assert log_prob.shape == torch.Size([8, 4])
    assert torch.all(sample > -1.0) and torch.all(sample < 1.0)


def test_scaled_beta_mean_and_std_match_affine_of_unit_beta():
    c1 = torch.tensor([2.0, 5.0, 1.0])
    c0 = torch.tensor([3.0, 1.0, 1.0])
    low, high = -2.0, 4.0
    scale = high - low
    dist = ScaledBeta(c1, c0, low=low, high=high, event_shape=torch.Size([3]))
    ref = _ref_beta(c1, c0)

    expected_mean = low + scale * ref.mean
    expected_std = scale * ref.variance.sqrt()

    assert torch.allclose(dist.mean(), expected_mean, atol=1e-6)
    assert torch.allclose(dist.std(), expected_std, atol=1e-6)


def test_scaled_beta_log_prob_matches_change_of_variables():
    c1 = torch.tensor([[2.0, 5.0, 1.5]])
    c0 = torch.tensor([[3.0, 1.0, 4.0]])
    low, high = -1.0, 1.0
    scale = high - low
    dist = ScaledBeta(c1, c0, low=low, high=high, event_shape=torch.Size([3]))
    ref = _ref_beta(c1, c0)

    y = torch.tensor([[-0.5, 0.25, 0.0]])
    x = (y - low) / scale
    expected = (ref.log_prob(x) - math.log(scale)).sum(-1)

    assert torch.allclose(dist.log_prob(y), expected, atol=1e-5)


def test_scaled_beta_entropy_matches_unit_beta_plus_log_scale():
    c1 = torch.tensor([[2.0, 5.0, 1.5]])
    c0 = torch.tensor([[3.0, 1.0, 4.0]])
    low, high = -1.0, 3.0
    scale = high - low
    dist = ScaledBeta(c1, c0, low=low, high=high, event_shape=torch.Size([3]))
    ref = _ref_beta(c1, c0)

    expected = (ref.entropy() + math.log(scale)).sum(-1)

    assert torch.allclose(dist.entropy(), expected, atol=1e-5)


def test_scaled_beta_kl_is_affine_invariant_and_matches_torch():
    c1a = torch.tensor([[2.0, 5.0, 1.5]])
    c0a = torch.tensor([[3.0, 1.0, 4.0]])
    c1b = torch.tensor([[1.2, 4.0, 2.0]])
    c0b = torch.tensor([[2.5, 2.0, 3.0]])
    low, high = -1.0, 1.0
    p = ScaledBeta(c1a, c0a, low=low, high=high, event_shape=torch.Size([3]))
    q = ScaledBeta(c1b, c0b, low=low, high=high, event_shape=torch.Size([3]))

    expected = D.kl.kl_divergence(_ref_beta(c1a, c0a), _ref_beta(c1b, c0b)).sum(-1)

    assert torch.allclose(p.kl_divergence(q), expected, atol=1e-5)


def test_scaled_beta_kl_to_itself_is_zero():
    c1 = torch.rand(4, 3) + 1.0
    c0 = torch.rand(4, 3) + 1.0
    dist = ScaledBeta(c1, c0, low=-1.0, high=1.0, event_shape=torch.Size([3]))
    rebuilt = dist.rebuild(dist.stats())

    kl = dist.kl_divergence(rebuilt)

    assert isinstance(dist.stats(), ScaledBetaParams)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)


def test_normal_kl_divergence_matches_torch():
    loc0 = torch.randn(4, 3)
    std0 = torch.rand(4, 3) + 0.1
    loc1 = torch.randn(4, 3)
    std1 = torch.rand(4, 3) + 0.1
    p = Normal(loc0, std0, event_shape=torch.Size([3]))
    q = Normal(loc1, std1, event_shape=torch.Size([3]))

    expected = D.kl.kl_divergence(
        D.Normal(loc0, std0), D.Normal(loc1, std1)
    ).sum(-1)

    assert torch.allclose(p.kl_divergence(q), expected, atol=1e-5)


def test_normal_rebuild_roundtrips_through_stats():
    loc = torch.randn(4, 3)
    std = torch.rand(4, 3) + 0.1
    dist = Normal(loc, std, event_shape=torch.Size([3]))

    params = dist.stats()
    rebuilt = dist.rebuild(params)

    assert isinstance(params, NormalParams)
    assert torch.allclose(rebuilt.mean(), loc)
    assert torch.allclose(rebuilt.std(), std)
    assert rebuilt.event_shape == torch.Size([3])


def test_beta_head_emits_bounded_actions_and_valid_concentrations():
    features = torch.randn(6, 5)
    head = BetaHead(5, 3, event_shape=torch.Size([3]), low=-1.0, high=1.0)

    dist = head(features)
    sample, log_prob = dist.rsample()

    assert isinstance(dist, ScaledBeta)
    assert dist.batch_shape == torch.Size([6])
    assert dist.event_shape == torch.Size([3])
    assert sample.shape == torch.Size([6, 3])
    assert log_prob.shape == torch.Size([6])
    assert torch.all(sample > -1.0) and torch.all(sample < 1.0)
    # concentrations must be >= 1 for a unimodal Beta
    assert torch.all(dist.concentration1 >= 1.0)
    assert torch.all(dist.concentration0 >= 1.0)


def test_beta_head_is_reparameterized():
    head = BetaHead(5, 3)
    features = torch.randn(6, 5, requires_grad=True)

    dist = head(features)
    sample, _ = dist.rsample()
    sample.sum().backward()

    last_layer = head.ffn.linear_block[-1]
    assert last_layer.weight.grad is not None
    assert torch.any(last_layer.weight.grad != 0.0)
