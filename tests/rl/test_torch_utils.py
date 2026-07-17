import importlib.util
from pathlib import Path

import pytest
import torch

_FUNCTIONAL_PATH = (
    Path(__file__).resolve().parents[2] / "thunder" / "rl" / "torch" / "functional.py"
)
_SPEC = importlib.util.spec_from_file_location("thunder_rl_torch_functional_test", _FUNCTIONAL_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

compute_lambda_returns = _MODULE.compute_lambda_returns
compute_gae = _MODULE.compute_gae


def test_compute_lambda_returns_bootstraps_with_next_value():
    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[10.0], [20.0]]])
    next_values = torch.tensor([[[20.0], [30.0]]])
    continues = torch.ones_like(rewards)

    returns = compute_lambda_returns(
        rewards, values, continues, next_values=next_values, gamma=1.0, lambda_=0.0
    )

    expected = torch.tensor([[[21.0], [32.0]]])
    assert torch.allclose(returns, expected)


def test_compute_lambda_returns_uses_final_next_value_for_lambda_bootstrap():
    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[10.0], [20.0]]])
    next_values = torch.tensor([[[20.0], [30.0]]])
    continues = torch.ones_like(rewards)

    returns = compute_lambda_returns(
        rewards, values, continues, next_values=next_values, gamma=1.0, lambda_=1.0
    )

    expected = torch.tensor([[[33.0], [32.0]]])
    assert torch.allclose(returns, expected)


def test_compute_gae_matches_discounted_td_residual_sum():
    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[10.0], [20.0]]])
    next_values = torch.tensor([[[20.0], [30.0]]])
    continues = torch.ones_like(rewards)

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        continues,
        gamma=1.0,
        lambda_=1.0,
        normalize=False,
    )

    expected_advantages = torch.tensor([[[23.0], [12.0]]])
    expected_returns = torch.tensor([[[33.0], [32.0]]])
    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_compute_gae_resets_across_terminations():
    rewards = torch.tensor([[[1.0], [2.0], [3.0]]])
    values = torch.tensor([[[0.5], [1.0], [1.5]]])
    next_values = torch.tensor([[[1.0], [1.5], [2.0]]])
    continues = torch.tensor([[[1.0], [0.0], [1.0]]])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        continues,
        gamma=0.9,
        lambda_=0.95,
        normalize=False,
    )

    expected_advantages = torch.tensor([[[2.2550], [1.0000], [3.3000]]])
    expected_returns = torch.tensor([[[2.7550], [2.0000], [4.8000]]])
    assert torch.allclose(advantages, expected_advantages, atol=1e-5)
    assert torch.allclose(returns, expected_returns, atol=1e-5)


def test_compute_gae_returns_match_lambda_returns_plus_baseline():
    rewards = torch.tensor([[[1.0], [2.0], [3.0]]])
    values = torch.tensor([[[0.5], [1.0], [1.5]]])
    next_values = torch.tensor([[[1.0], [1.5], [2.0]]])
    continues = torch.ones_like(rewards)

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        continues,
        gamma=0.9,
        lambda_=0.8,
        normalize=False,
    )
    lambda_returns = compute_lambda_returns(
        rewards, values, continues, next_values=next_values, gamma=0.9, lambda_=0.8
    )

    assert torch.allclose(returns, lambda_returns, atol=1e-6)
    assert torch.allclose(advantages, lambda_returns - values, atol=1e-6)


def test_compute_gae_rejects_values_with_bootstrap_timestep():
    rewards = torch.zeros(1, 2, 1)
    values = torch.zeros(1, 3, 1)
    next_values = torch.zeros_like(rewards)
    continues = torch.ones_like(rewards)

    with pytest.raises(
        ValueError, match="rewards, values, next_values and continues must have same shape"
    ):
        compute_gae(rewards, values, next_values, continues)
