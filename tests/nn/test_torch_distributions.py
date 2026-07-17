import math

import torch

from thunder.nn.torch.distributions import (
    AffineTransform,
    ConsistentNormalHead,
    Normal,
    NormalHead,
    TanhTransform,
    TransformedDistHead,
    TransformedDistribution,
)


def test_normal_without_event_shape_keeps_elementwise_log_prob():
    loc = torch.zeros(2, 3)
    std = torch.ones(3)
    dist = Normal(loc, std)
    value = torch.zeros(2, 3)

    log_prob = dist.log_prob(value)

    assert dist.batch_shape == torch.Size([2, 3])
    assert dist.event_shape == torch.Size([])
    assert log_prob.shape == torch.Size([2, 3])
    assert torch.allclose(log_prob, torch.full((2, 3), -0.5 * math.log(2.0 * math.pi)))


def test_normal_sums_log_prob_and_entropy_over_event_shape():
    loc = torch.zeros(2, 3)
    std = torch.ones(3)
    dist = Normal(loc, std, event_shape=torch.Size([3]))
    value = torch.zeros(2, 3)

    log_prob = dist.log_prob(value)
    entropy = dist.entropy()
    sample, sample_log_prob = dist.rsample(torch.Size([4]))

    expected_log_prob = torch.full((2,), 3 * -0.5 * math.log(2.0 * math.pi))
    expected_entropy = torch.full((2,), 3 * (0.5 + 0.5 * math.log(2.0 * math.pi)))
    assert dist.batch_shape == torch.Size([2])
    assert dist.event_shape == torch.Size([3])
    assert log_prob.shape == torch.Size([2])
    assert entropy.shape == torch.Size([2])
    assert sample.shape == torch.Size([4, 2, 3])
    assert sample_log_prob.shape == torch.Size([4, 2])
    assert torch.allclose(log_prob, expected_log_prob)
    assert torch.allclose(entropy, expected_entropy)


def test_transformed_distribution_sums_jacobian_over_event_shape():
    loc = torch.zeros(2, 3)
    std = torch.ones(3)
    base = Normal(loc, std, event_shape=torch.Size([3]))
    dist = TransformedDistribution(base, [AffineTransform(scale=2.0), TanhTransform()])
    value = torch.zeros(2, 3)

    log_prob = dist.log_prob(value)
    sample, sample_log_prob = dist.rsample(torch.Size([4]))

    expected = base.log_prob(torch.zeros_like(value)) - 3 * math.log(2.0)
    assert log_prob.shape == torch.Size([2])
    assert sample.shape == torch.Size([4, 2, 3])
    assert sample_log_prob.shape == torch.Size([4, 2])
    assert torch.allclose(log_prob, expected)


def test_normal_heads_pass_event_shape_to_distribution():
    features = torch.zeros(2, 5)
    event_shape = torch.Size([3])
    normal_head = NormalHead(5, 3, event_shape=event_shape)
    consistent_head = ConsistentNormalHead(5, 3, event_shape=event_shape)

    dist = normal_head(features)
    consistent_dist = consistent_head(features)

    assert dist.batch_shape == torch.Size([2])
    assert dist.event_shape == event_shape
    assert consistent_dist.batch_shape == torch.Size([2])
    assert consistent_dist.event_shape == event_shape


def test_normal_heads_support_exp_std_mode():
    features = torch.zeros(2, 5)
    init_std = 0.25
    expected_std = torch.full((2, 3), init_std)
    normal_head = NormalHead(5, 3, init_std=init_std, std_mode="exp")
    consistent_head = ConsistentNormalHead(5, 3, init_std=init_std, std_mode="exp")

    dist = normal_head(features)
    consistent_dist = consistent_head(features)

    assert torch.allclose(dist.std(), expected_std)
    assert torch.allclose(consistent_dist.std(), expected_std)


def test_normal_heads_initialize_mean_near_zero():
    features = torch.randn(32, 5)
    zeros = torch.zeros(2, 5)
    normal_head = NormalHead(5, 3, [16])
    consistent_head = ConsistentNormalHead(5, 3, [16])

    assert torch.allclose(normal_head(zeros).mean(), torch.zeros(2, 3), atol=1e-4)
    assert torch.allclose(consistent_head(zeros).mean(), torch.zeros(2, 3), atol=1e-4)
    assert normal_head(features).mean().abs().max() < 0.1
    assert consistent_head(features).mean().abs().max() < 0.1


def test_exp_std_mode_clamps_log_std_before_exp():
    features = torch.zeros(2, 5)
    min_std = 0.01
    max_std = 2.0
    normal_head = NormalHead(5, 3, min_std=min_std, max_std=max_std, std_mode="exp")
    consistent_head = ConsistentNormalHead(
        5, 3, min_std=min_std, max_std=max_std, std_mode="exp"
    )
    last_layer = normal_head.ffn.linear_block[-1]

    with torch.no_grad():
        last_layer.bias[normal_head.out_size :].fill_(1000.0)
        consistent_head.std_param.fill_(1000.0)
    assert torch.allclose(normal_head(features).std(), torch.full((2, 3), max_std))
    assert torch.allclose(consistent_head(features).std(), torch.full((2, 3), max_std))

    with torch.no_grad():
        last_layer.bias[normal_head.out_size :].fill_(-1000.0)
        consistent_head.std_param.fill_(-1000.0)
    assert torch.allclose(normal_head(features).std(), torch.full((2, 3), min_std))
    assert torch.allclose(consistent_head(features).std(), torch.full((2, 3), min_std))


def test_transformed_head_passes_std_mode_to_base_head():
    features = torch.zeros(2, 5)
    init_std = 0.25
    head = TransformedDistHead(
        5,
        3,
        init_std=init_std,
        std_mode="exp",
        consistent_std=True,
    )

    dist = head.base_head(features)

    assert torch.allclose(dist.std(), torch.full((2, 3), init_std))
