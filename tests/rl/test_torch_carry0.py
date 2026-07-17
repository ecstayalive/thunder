"""Equivalence tests for ``SplitTraj``'s per-trajectory initial carry.

The rollout buffer no longer stores the recurrent carry at every timestep
(prohibitively large for big SSM/Mamba states). Instead only the window-start
carry rides in the cache, and ``SplitTraj`` rebuilds the per-trajectory initial
carry that it used to obtain by splitting the per-step buffer field and taking
``leaf[:, 0]``.

These tests pin that the new path is *numerically identical* to the old one, so
the change is a pure memory optimization with zero effect on PPO semantics.
"""

from types import SimpleNamespace

import torch

from thunder.core.data import Batch
from thunder.rl.torch.functional import get_trajectory_lengths
from thunder.rl.torch.operations import SplitTraj


def _old_carry0(start: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """The carry0 the *old* code produced: store the carry entering every step
    (zeroed on done by ``agent.reset``), split it like ``SplitTraj``, then take
    ``leaf[:, 0]`` for each trajectory."""
    n_envs, horizon = dones.shape
    feat = start.shape[1:]
    per_step = start.new_zeros((n_envs, horizon, *feat))
    carry = start.clone()
    for t in range(horizon):
        per_step[:, t] = carry
        keep = (~dones[:, t]).reshape(n_envs, *([1] * len(feat))).to(carry.dtype)
        carry = carry * keep
    traj_lengths = get_trajectory_lengths(dones)
    num_trajs = traj_lengths.shape[0]
    mask = traj_lengths.unsqueeze(1) > torch.arange(horizon).unsqueeze(0)
    pooled = per_step.new_zeros((num_trajs, horizon, *feat))
    pooled[mask] = per_step.flatten(0, 1)
    return pooled[:, 0]


def _run_split(terminated, cache):
    dones = terminated
    batch = Batch(
        terminated=terminated,
        timeouts=torch.zeros_like(terminated),
    )
    ctx = SimpleNamespace(batch=batch, cache=cache)
    ctx, _ = SplitTraj().forward(ctx)
    return ctx, dones


def test_split_traj_builds_initial_carry_matching_old_path():
    torch.manual_seed(0)
    terminated = torch.tensor(
        [
            [False, True, False, False],  # done mid-window -> 2 trajectories
            [False, False, False, False],  # no done -> 1 trajectory
            [True, False, True, False],  # two dones -> 3 trajectories
            [False, False, False, True],  # done on last step -> 1 trajectory
        ]
    )
    # Mamba-style structured policy carry (conv_state, ssm_state) + plain critic carry.
    conv_start = torch.randn(4, 6)
    ssm_start = torch.randn(4, 2, 3, 5)
    critic_start = torch.randn(4, 8)
    cache = Batch(
        initial=Batch(
            policy_carry=(conv_start, ssm_start),
            critic_carry=critic_start,
        )
    )

    ctx, dones = _run_split(terminated, cache)
    initial = ctx.cache.initial  # re-aligned to per-trajectory rows, in place

    assert torch.equal(initial.policy_carry[0], _old_carry0(conv_start, dones))
    assert torch.equal(initial.policy_carry[1], _old_carry0(ssm_start, dones))
    assert torch.equal(initial.critic_carry, _old_carry0(critic_start, dones))
    # one row per trajectory, no time axis.
    assert initial.policy_carry[1].shape == (ctx.batch.mask.shape[0], 2, 3, 5)


def test_split_traj_with_none_carry_stays_none():
    terminated = torch.tensor([[False, True, False], [False, False, False]])
    cache = Batch(initial=Batch(policy_carry=None, critic_carry=None))

    ctx, _ = _run_split(terminated, cache)

    assert ctx.cache.initial.policy_carry is None
    assert ctx.cache.initial.critic_carry is None
