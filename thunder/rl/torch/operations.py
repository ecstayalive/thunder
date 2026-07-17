from __future__ import annotations

import math
from typing import Iterable, TYPE_CHECKING

import gymnasium as gym
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._cxx_pytree as pytree

from thunder.core import (
    Batch,
    ExecutionContext,
    Executor,
    ModelPack,
    Objective,
    Operation,
    Pipeline,
    Ref,
)
from thunder.env.env import ThunderEnv
from thunder.utils import Matrix, Scalar
from .buffer import gather, gather_env
from .env import EnvMetrics, NormalizeObsWrapper
from .functional import get_trajectory_lengths

if TYPE_CHECKING:
    from thunder.nn.torch import Distribution, GruMlp, LstmMlp
    from thunder.utils import Workspace
    from .agent import Agent
    from .buffer import BatchSampler, Buffer
    from .models import Actor


class SIGRegObj(Objective):
    """Sketched Isotropic Gaussian Regularization
    For details: https://arxiv.org/abs/2511.08544
    Args:

    """

    def __init__(
        self,
        weight=1.0,
        key: str = "states",
        name="sigreg",
        num_slices=128,
        t_points=17,
        t_range=3.0,
    ):
        super().__init__(weight, name)
        self.key = key
        self.num_slices = num_slices
        self.device = Executor.default_device()
        self.t = torch.linspace(
            0, t_range, t_points, device=self.device, dtype=torch.float32
        )
        dt = t_range / (t_points - 1)
        weights = torch.full(
            (t_points,), 2 * dt, device=self.device, dtype=torch.float32
        )
        weights[[0, -1]] = dt
        self.phi = torch.exp(-0.5 * self.t.square())
        self.integration_weights = weights * self.phi
        self.global_step = torch.zeros((), device=self.device, dtype=torch.long)
        self._generator = None
        # Scheme
        self.requires = frozenset({Ref(f"batch[{key!r}]"), Ref("batch.mask")})
        self.provides = frozenset()

    def _get_generator(self, device, seed):
        if self._generator is None:
            self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        return self._generator

    def compute(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        embeddings: torch.Tensor = batch[self.key]  # [B, L ,D]
        mask: torch.Tensor = batch.mask  # [B, L]
        _, L, D = embeddings.shape
        device = embeddings.device
        with torch.no_grad():
            if dist.is_available() and dist.is_initialized():
                seed_tensor = self.global_step.clone()
                dist.all_reduce(seed_tensor, op=dist.ReduceOp.MAX)
                seed = seed_tensor.item()
            else:
                seed = self.global_step.item()
            g = self._get_generator(device, seed)
            A = torch.randn((D, self.num_slices), device=device, generator=g)
            A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
            self.global_step.add_(1)

        # Apply the regularizer across the batch axis for each time index,
        # then average the per-time losses over valid time steps.
        x_proj = embeddings @ A  # [B, L, K]
        x_t = x_proj.unsqueeze(-1) * self.t  # [B, L, K, T]
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        stats_mask = mask.to(dtype=cos_vals.dtype).unsqueeze(-1).unsqueeze(-1)
        cos_sum = (cos_vals * stats_mask).sum(dim=0)  # [L, K, T]
        sin_sum = (sin_vals * stats_mask).sum(dim=0)  # [L, K, T]
        counts = mask.to(dtype=cos_vals.dtype).sum(dim=0)  # [L]

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sin_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        valid_times = counts > 0
        if not valid_times.any():
            return embeddings.new_zeros(()), {}

        safe_counts = counts.clamp_min(1.0).view(L, 1, 1)
        cos_mean = cos_sum / safe_counts
        sin_mean = sin_sum / safe_counts

        err_sq = (cos_mean - self.phi).square() + sin_mean.square()
        # The trapezoidal rule integrals
        loss_per_time = (err_sq @ self.integration_weights).mean(dim=-1) * counts
        loss = loss_per_time[valid_times].mean()
        return loss, {}


class Rollout(Operation):
    """ """

    provides = ("batch", "cache")

    def __init__(
        self,
        env: ThunderEnv,
        agent: Agent,
        step: int = 32,
        explore: bool = True,
        name="rollout",
    ):
        super().__init__(name)
        self.env = env
        self.step = step
        self.autoreset_mode = self.env.autoreset_mode
        self.explore = explore
        self.agent = agent
        self.metrics = EnvMetrics()
        if isinstance(env, NormalizeObsWrapper):
            env.train()
        self.obs, _ = env.reset()

    def forward(self, ctx: ExecutionContext | None = None):
        self.metrics.reset(clear_episodes=False)
        initial: Batch = self.agent.snapshot()
        final: Batch | None = None
        with torch.inference_mode():
            for i in range(self.step):
                action = self.agent.act(self.obs, self.explore)
                next_obs, rewards, terminated, timeouts, info = self.env.step(action)
                self.metrics.update(rewards, terminated, timeouts, info)
                self.agent.collect(
                    next_obs=next_obs,
                    rewards=rewards,
                    terminated=terminated,
                    timeouts=timeouts,
                    info=info,
                )
                if i == self.step - 1:
                    final = self.agent.snapshot()
                dones: torch.Tensor = (terminated | timeouts).bool()
                if self.autoreset_mode is not gym.vector.AutoresetMode.SAME_STEP:
                    reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                    if reset_idx.numel() > 0:
                        next_obs, _ = self.env.reset(indices=reset_idx)
                self.agent.reset(dones)
                self.obs = next_obs
        cache = Batch(initial=initial, final=final)
        ctx = ctx.replace(batch=self.agent.buffer.data(), cache=cache)
        return ctx, self.metrics.compute()

    def __repr__(self):
        """ """
        return super().__repr__()

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {
                "env": type(self.env).__name__,
                "agent": type(self.agent).__name__,
                "step": self.step,
                "autoreset_mode": self.autoreset_mode.name,
                "explore": self.explore,
            }
        )
        return fields


class Play(Operation):
    """Actor-only inference loop for evaluation/play."""

    def __init__(self, env: ThunderEnv, agent: Agent, explore: bool = False, name=""):
        super().__init__(name)
        self.env = env
        self.autoreset_mode = self.env.autoreset_mode
        self.explore = explore
        self.agent = agent
        self.env_metrics = EnvMetrics()
        self.reward_manager = getattr(env.unwrapped, "reward_manager", None)
        self.observation_manager = getattr(env.unwrapped, "observation_manager", None)
        if isinstance(env, NormalizeObsWrapper):
            env.eval()
        self.obs, _ = env.reset()

    def reward_terms(self) -> dict[str, torch.Tensor]:
        """"""
        if self.reward_manager is None:
            return {}
        step_reward = self.reward_manager._step_reward
        return {
            name: step_reward[:, i]
            for i, name in enumerate(self.reward_manager.active_terms)
        }

    def observation_terms(self) -> dict[str, torch.Tensor]:
        """"""
        if self.observation_manager is None:
            return {}
        manager = self.observation_manager
        terms = {}
        for group, names in manager.active_terms.items():
            group_obs = self.obs.get(group)
            if group_obs is None:
                continue
            if manager.group_obs_concatenate[group]:
                widths = [math.prod(s) for s in manager.group_obs_term_dim[group]]
                group_obs = group_obs.split(widths, dim=-1)
            terms.update({f"{group}/{n}": v for n, v in zip(names, group_obs)})
        return terms

    def reward_metrics(self) -> dict[str, Matrix]:
        return {
            f"RewardTerm/{name}": Matrix(value.detach())
            for name, value in self.reward_terms().items()
        }

    def observation_metrics(self) -> dict[str, Matrix]:
        return {
            f"Observation/{name}": Matrix(value.detach())
            for name, value in self.observation_terms().items()
        }

    def forward(self, ctx: ExecutionContext | None = None):
        metrics = {}
        self.env_metrics.reset(clear_episodes=False)
        with torch.inference_mode():
            action = self.agent.infer(self.obs, self.explore)
            next_obs, rewards, terminated, timeouts, info = self.env.step(action)
            self.env_metrics.update(rewards, terminated, timeouts, info)
            dones: torch.Tensor = (terminated | timeouts).bool()
            if self.autoreset_mode is not gym.vector.AutoresetMode.SAME_STEP:
                reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if reset_idx.numel() > 0:
                    next_obs, _ = self.env.reset(indices=reset_idx)
            self.agent.reset(dones)
            self.obs = next_obs
            metrics.update(self.env_metrics.compute())
            metrics.update(self.reward_metrics())
            metrics.update(self.observation_metrics())
            metrics.update(
                {
                    "Action": Matrix(action.detach()),
                    "Reward": Matrix(rewards.detach()),
                    "Terminated": Matrix(terminated.float()),
                    "Timeout": Matrix(timeouts.float()),
                }
            )
        return ctx, metrics

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {
                "env": type(self.env).__name__,
                "agent": type(self.agent).__name__,
                "autoreset_mode": self.autoreset_mode.name,
                "explore": self.explore,
            }
        )
        return fields


class MiniBatchLoop(Pipeline):
    """Run a pipeline body over loader batches in a local scope."""

    def __init__(
        self,
        sampler: BatchSampler,
        pipeline: Iterable[Operation],
        jit: bool = False,
        epoch: int = 5,
        name="",
    ):
        self.sampler = sampler
        self.epoch = epoch
        super().__init__(pipeline, name=name, jit=jit)

    def setup(self):
        self._pipeline = tuple(self.pipeline)
        pipeline_requires, pipeline_provides = self.analyze_contract()
        self._bound_batch_requires = frozenset(
            ref for ref in pipeline_requires if ref.startswith("batch")
        )
        external_requires = frozenset(
            ref for ref in pipeline_requires if not ref.startswith("batch")
        )
        self.requires = external_requires
        if self._bound_batch_requires:
            self.requires = frozenset((*self.requires, Ref("batch")))
        self.provides = frozenset(
            ref for ref in pipeline_provides if not ref.startswith("batch")
        )
        self.validate(
            initial_refs=(
                *self._bound_batch_requires,
                *external_requires,
            )
        )
        self.forward_fn = self._compile_forward()

    def forward(self, ctx: ExecutionContext):
        m = {}
        batch = ctx.batch
        cache = ctx.cache
        for _ in range(self.epoch):
            for rows, cols in self.sampler(batch):
                ctx = ctx.replace(
                    batch=gather(batch, rows, cols),
                    cache=gather_env(cache, cols),
                )
                ctx, m = self.forward_fn(ctx)
                ctx = ctx.replace(batch=batch, cache=cache)
        return ctx, m

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update({"epoch": self.epoch, "loader": type(self.sampler).__name__})
        return fields


class SoftUpdate(Operation):
    """ """

    def __init__(self, source: str, target: str, tau: float, name="soft_update"):
        super().__init__(name)
        self.source = source
        self.target = target
        self.tau = tau

        self.provides = frozenset()

    def forward(self, ctx: ExecutionContext):
        with torch.inference_mode():
            models: ModelPack = ctx.models
            source: nn.Module = models[self.source]
            target: nn.Module = models[self.target]
            for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
                tgt.data.mul_(1 - self.tau).add_(src.data, alpha=self.tau)
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update({"source": self.source, "target": self.target, "tau": self.tau})
        return fields


class HardUpdate(Operation):
    """ """

    def __init__(self, source: str, target: str, name="hard_update"):
        super().__init__(name)
        self.source = source
        self.target = target
        self.provides = frozenset()

    def forward(self, ctx: ExecutionContext):
        with torch.inference_mode():
            models: ModelPack = ctx.models
            source: nn.Module = models.get(self.source)
            target: nn.Module = models.get(self.target)
            for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
                tgt.data.copy_(src.data)
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update({"source": self.source, "target": self.target})
        return fields


class SplitTraj(Operation):
    """Re-granulate an env-major rollout from per-env rows to per-trajectory rows.

    Fully generic — no field is named. Both containers are transformed via a
    parallel ``tree_map``:

    - ``batch`` (dense per-step ``[N, L, ...]``): ``_split_leaf`` cuts each env's
      timeline at dones and zero-pads the pieces into ``[num_trajs, chunk_len, ...]``
      with a validity ``mask``;
    - ``cache`` (sparse per-env ``[N, ...]`` boundary snapshots): ``_scatter_leaf``
      scatters each env's value into that env's first trajectory and zeros the
      rest — which equals the reset state ``agent.reset`` applies at episode
      boundaries during rollout.

    A recurrent forward regenerates intermediate states from the initial one, so
    only the trajectory-start carry is needed. The carry layout is never inspected
    here; consumers (e.g. a PPO loss) read whatever they placed in the cache.
    """

    requires = ("batch.terminated", "batch.timeouts")
    provides = ("batch.mask",)

    def __init__(self, name="split_traj"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.terminated | batch.timeouts
        device = dones.device

        chunk_len = dones.shape[1]
        traj_lengths: torch.Tensor = get_trajectory_lengths(dones)
        num_trajs = traj_lengths.shape[0]
        row_indices = torch.arange(chunk_len, device=device).unsqueeze(0)
        # Batch
        mask = traj_lengths.unsqueeze(1) > row_indices
        # Cache
        starts = torch.cumsum(traj_lengths, dim=0) - traj_lengths
        leading = (starts % chunk_len) == 0

        def _split_leaf(leaf: torch.Tensor | None) -> torch.Tensor:
            """Per-step leaf [N, L, *] -> zero-padded per-trajectory [num_trajs, chunk_len, *]."""
            if leaf is None:
                return None
            pooled = leaf.new_zeros((num_trajs, chunk_len, *leaf.shape[2:]))
            pooled[mask] = leaf.flatten(0, 1)
            return pooled

        def _scatter_leaf(leaf: torch.Tensor | None) -> torch.Tensor:
            """Per-env leaf [N, *] -> per-trajectory [num_trajs, *]: each env's value
            into its first trajectory, zeros (reset) for the rest."""
            if leaf is None:
                return None
            scattered = leaf.new_zeros((num_trajs, *leaf.shape[1:]))
            scattered[leading] = leaf
            return scattered

        ctx.batch = pytree.tree_map(_split_leaf, ctx.batch).replace(mask=mask)
        ctx.cache = pytree.tree_map(_scatter_leaf, ctx.cache)
        return ctx, {}


class StaticSplitTraj(Operation):
    """ """

    requires = ("batch.terminated", "batch.timeouts")
    provides = ("batch.mask",)

    def __init__(self, name="static_split_traj", top: bool = True):
        super().__init__(name)
        self.top = top

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.terminated | batch.timeouts
        device = dones.device

        batch_size, chunk_len = dones.shape[0], dones.shape[1]

        traj_lengths: torch.Tensor = get_trajectory_lengths(dones)
        num_trajs = traj_lengths.shape[0]
        row_indices = torch.arange(chunk_len, device=device).unsqueeze(0)
        mask = traj_lengths.unsqueeze(1) > row_indices

        if num_trajs == batch_size:
            idxs = torch.randperm(batch_size, device=device)
        else:
            if self.top:
                _, idxs = torch.topk(traj_lengths, k=batch_size, largest=True)
                idxs = idxs[torch.randperm(batch_size, device=device)]
            else:
                idxs = torch.randperm(num_trajs, device=device)[:batch_size]

        def _transform_leaf(leaf: torch.Tensor) -> torch.Tensor:
            if leaf is None:
                return None
            flat_leaf = leaf.flatten(0, 1)
            embedding_shape = flat_leaf.shape[1:]
            # Allocate [Num_Trajs, Max_Len, ...]
            pooled = torch.zeros(
                (num_trajs, chunk_len, *embedding_shape),
                dtype=flat_leaf.dtype,
                device=device,
            )
            pooled[mask] = flat_leaf
            return pooled[idxs]

        ctx.batch = pytree.tree_map(_transform_leaf, ctx.batch)
        ctx.batch = ctx.batch.replace(mask=mask[idxs])
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields["top"] = self.top
        return fields


class UnpadTraj(Operation):
    """ """

    def __init__(self, name="unpad_traj"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        return ctx, {}


class SaveModels(Operation):
    """Periodically checkpoint model weights into a workspace.

    The workspace is the single authority for the checkpoint directory and file
    naming; this op no longer creates directories ad hoc.
    """

    def __init__(
        self, interval: int, workspace: Workspace, name="save_models", enable=True
    ):
        super().__init__(name)
        self.interval = interval
        self.enable = enable
        self.workspace = workspace

    def forward(self, ctx):
        if (
            self.enable
            and ctx.manager.is_main_process
            and (ctx.step + 1) % self.interval == 0
        ):
            self.workspace.ensure()
            torch.save(
                ctx.models.state_dict(), self.workspace.checkpoint_path(ctx.step + 1)
            )
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {
                "interval": self.interval,
                "path": str(self.workspace.checkpoint_dir),
                "enable": self.enable,
            }
        )
        return fields


class ClearBuffer(Operation):
    def __init__(self, buffer: Buffer, name="clear_buffer"):
        super().__init__(name=name)
        self.buffer = buffer

    def forward(self, ctx: ExecutionContext):
        self.buffer.clear()
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields["buffer"] = type(self.buffer).__name__
        return fields


class ComputeLastValue(Operation):
    requires = (
        "batch.terminated",
        "batch.values",
        "batch.timeouts",
        "batch.next_obs",
        "cache.final.critic_carry",
    )
    provides = ("batch.next_values",)

    def __init__(
        self,
        autoreset_mode: gym.vector.AutoresetMode = gym.vector.AutoresetMode.NEXT_STEP,
        name="",
    ):
        super().__init__(name=name)
        self.autoreset_mode = autoreset_mode

    def forward(self, ctx):
        critic: LstmMlp | GruMlp = ctx.models.critic
        batch: Batch = ctx.batch
        cache: Batch = ctx.cache

        terminated: torch.Tensor = batch.terminated.bool()
        timeouts: torch.Tensor = batch.timeouts.bool()
        next_values = torch.empty_like(batch.values)
        next_values[:, :-1] = batch.values[:, 1:]

        last_obs = {
            k: v[:, -1].unsqueeze(1).contiguous() for k, v in batch.next_obs.items()
        }
        with torch.inference_mode():
            last_values, _ = critic(last_obs, cache.final.critic_carry)
        next_values[:, -1] = last_values.squeeze(-1).squeeze(1)
        next_values[terminated] = 0.0
        if self.autoreset_mode is gym.vector.AutoresetMode.SAME_STEP:
            next_values[timeouts & ~terminated] = batch.values[timeouts & ~terminated]
        ctx.batch = batch.replace(next_values=next_values)
        ctx.cache = cache.replace(final=None)

        return ctx, {}


class ComputeGae(Operation):
    """ """

    requires = (
        "batch.rewards",
        "batch.values",
        "batch.next_values",
        "batch.terminated",
        "batch.timeouts",
    )
    provides = ("batch.advantages", "batch.returns")

    def __init__(self, gamma=0.99, lambda_=0.95, normalize=True, eps=1e-6, name="gae"):
        super().__init__(name)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.normalize = normalize
        self.eps = eps

    def forward(self, ctx):
        batch: Batch = ctx.batch
        rewards: torch.Tensor = batch.rewards
        values: torch.Tensor = batch.values
        next_values: torch.Tensor = batch.next_values
        advantage = torch.zeros_like(rewards[:, -1])
        returns = torch.zeros_like(rewards)
        terminated = batch.terminated.bool()
        timeouts = batch.timeouts.bool()

        bootstrap_continue = (~terminated).to(values.dtype)
        trace_continue = (~(terminated | timeouts)).to(values.dtype)

        for step in reversed(range(rewards.shape[1])):
            delta = (
                rewards[:, step]
                + self.gamma * bootstrap_continue[:, step] * next_values[:, step]
                - values[:, step]
            )
            advantage = (
                delta + self.gamma * self.lambda_ * trace_continue[:, step] * advantage
            )
            returns[:, step] = advantage + values[:, step]
        advantages = returns - values
        if self.normalize:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(self.eps)
            advantages = (advantages - adv_mean) / adv_std
        ctx.batch = batch.replace(advantages=advantages, returns=returns)
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {
                "gamma": self.gamma,
                "lambda_": self.lambda_,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return fields


class PpoSurrogateLoss(Objective):
    requires = (
        "batch.obs",
        "batch.actions",
        "batch.advantages",
        "batch.mask",
        "batch.log_prob",
        "cache.initial.policy_carry",
    )

    def __init__(
        self,
        weight: float = 1.0,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.1,
        name: str = "ppo_surrogate_loss",
    ):
        super().__init__(weight=weight, name=name)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

    def compute(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        actor: Actor = ctx.models.actor

        dist, _ = actor(batch.obs, ctx.cache.initial.policy_carry)
        dist: Distribution
        log_prob = dist.log_prob(batch.actions)
        mask = batch.mask.to(dtype=log_prob.dtype)
        mask_count = mask.sum().clamp(min=1.0)

        log_ratio = log_prob - batch.log_prob
        ratio = torch.exp(log_ratio)
        unclipped = ratio * batch.advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * batch.advantages
        )
        suggorate_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / mask_count
        entropy = dist.entropy()
        entropy_mean = (entropy * mask).sum() / mask_count
        loss = suggorate_loss - self.entropy_coef * entropy_mean

        # Metrics
        kl = (torch.exp(log_ratio) - 1.0) - log_ratio
        clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).to(dtype=mask.dtype)
        kl_mean = ((kl * mask).sum() / mask_count).detach()
        ratio_mean = (ratio * mask).sum() / mask_count
        clip_fraction_mean = (clip_fraction * mask).sum() / mask_count
        # exports
        self.exports["kl"] = kl_mean
        return loss, {
            "policy_loss": Scalar(suggorate_loss.detach()),
            "entropy": Scalar(entropy_mean.detach()),
            "ratio": Scalar(ratio_mean.detach()),
            "kl": Scalar(kl_mean),
            "clip_fraction": Scalar(clip_fraction_mean.detach()),
            "std": Scalar(dist.std().detach().mean()),
        }

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {"clip_ratio": self.clip_ratio, "entropy_coef": self.entropy_coef}
        )
        return fields


class CriticLoss(Objective):

    requires = (
        "batch.mask",
        "batch.obs",
        "cache.initial.critic_carry",
        "batch.returns",
        "batch.values",
    )

    def __init__(
        self, weight: float = 1.0, value_clip: float = 0.2, name: str = "critic_loss"
    ):
        super().__init__(weight=weight, name=name)
        self.value_clip = value_clip

    def compute(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        critic: GruMlp | LstmMlp = ctx.models.critic
        value, _ = critic(batch.obs, ctx.cache.initial.critic_carry)
        value = value.squeeze(-1)
        mask = batch.mask.to(dtype=value.dtype)
        mask_count = mask.sum().clamp(min=1.0)
        delta_value = value - batch.values
        clipped_value = batch.values + delta_value.clamp(
            -self.value_clip, self.value_clip
        )
        value_loss = (value - batch.returns).square()
        clipped_value_loss = (clipped_value - batch.returns).square()
        loss = (torch.maximum(value_loss, clipped_value_loss) * mask).sum() / mask_count
        # Metrics
        clip_fraction = (delta_value.abs() > self.value_clip).to(dtype=mask.dtype)
        return loss, {
            "value_loss": Scalar(((value_loss * mask).sum() / mask_count).detach()),
            "clipped_value_loss": Scalar(
                ((clipped_value_loss * mask).sum() / mask_count).detach()
            ),
            "value_clip_fraction": Scalar(
                ((clip_fraction * mask).sum() / mask_count).detach()
            ),
            "value": Scalar(((value * mask).sum() / mask_count).detach()),
            "returns": Scalar(((batch.returns * mask).sum() / mask_count).detach()),
        }

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields["value_clip"] = self.value_clip
        return fields
