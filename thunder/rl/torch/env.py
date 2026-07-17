import time
from dataclasses import dataclass
from typing import Any

import torch

from thunder.env import ObservationType, ObservationWrapper, ThunderEnv, WrapperObsType
from thunder.nn.torch import DictRunningNorm1d
from thunder.utils import Scalar


class NormalizeObsWrapper(ObservationWrapper):
    """Normalizes observations using running mean and variance."""

    def __init__(
        self, env: ThunderEnv, normalizer: DictRunningNorm1d, update: bool = True
    ):
        super().__init__(env)
        self.normalizer = normalizer
        self.update = update

    def observation(self, observation: ObservationType) -> WrapperObsType:
        if self.update:
            self.normalizer.update(observation)
        return self.normalizer(observation)

    def train(self, mode: bool = True):
        self.update = mode
        return self

    def eval(self):
        return self.train(False)


@dataclass
class EnvMetrics:
    """Accumulates rollout and episode metrics from vectorized env steps.

    Core reward and episode statistics are computed from ``rewards``,
    ``terminated`` and ``timeouts`` instead of trusting environment ``info``. This
    keeps episode return and episode length well-defined across Gymnasium,
    IsaacLab DirectRLEnv and IsaacLab ManagerBasedRLEnv.

    ``info["log"]`` is treated as optional environment diagnostics and is
    recorded on every step by default. Set ``log_on_done_only=True`` for envs
    that only refresh diagnostics when episodes finish.
    """

    num_envs: int | None = None
    log_on_done_only: bool = False

    def __post_init__(self):
        self.episode_returns: torch.Tensor | None = None
        self.episode_lengths: torch.Tensor | None = None
        self.reset(clear_episodes=True)

    def update(self, rewards, terminated, timeouts, info: dict[str, Any] | None):
        """Update metrics from one vectorized environment step.

        Args:
            rewards: Batched rewards returned by ``env.step``. Extra trailing
                reward dimensions are summed into one scalar per environment.
            terminated: True for environments that ended by task termination.
            timeouts: True for environments that ended by time limit/truncation.
            info: Optional environment info. Only ``info["log"]`` is summarized;
                top-level metadata is ignored.
        """

        terminated = torch.as_tensor(terminated).bool().reshape(-1)
        timeouts = torch.as_tensor(timeouts).bool().reshape(-1)
        dones = terminated | timeouts
        reward = self._reward_per_env(rewards, dones.numel())
        self._ensure_episode_state(reward)
        self._ensure_rollout_accumulators(reward.device)

        self.rollout_reward_sum = self.rollout_reward_sum + reward.sum()
        self.rollout_reward_sq_sum = self.rollout_reward_sq_sum + reward.square().sum()
        self.rollout_steps += 1
        self.rollout_transitions += reward.numel()
        self.terminated_count = self.terminated_count + terminated.sum()
        self.timeout_count = self.timeout_count + timeouts.sum()

        self.episode_returns.add_(reward)
        self.episode_lengths.add_(1)
        done_count = dones.sum()
        done_returns = torch.where(
            dones, self.episode_returns, torch.zeros_like(self.episode_returns)
        )
        done_lengths = torch.where(
            dones, self.episode_lengths, torch.zeros_like(self.episode_lengths)
        )
        self.completed_count = self.completed_count + done_count
        self.completed_return_sum = self.completed_return_sum + done_returns.sum()
        self.completed_length_sum = self.completed_length_sum + done_lengths.sum()
        self.completed_return_min = torch.minimum(
            self.completed_return_min,
            torch.where(
                dones,
                self.episode_returns,
                torch.full_like(self.episode_returns, torch.inf),
            ).min(),
        )
        self.completed_return_max = torch.maximum(
            self.completed_return_max,
            torch.where(
                dones,
                self.episode_returns,
                torch.full_like(self.episode_returns, -torch.inf),
            ).max(),
        )
        self.completed_length_min = torch.minimum(
            self.completed_length_min,
            torch.where(
                dones,
                self.episode_lengths,
                torch.full_like(self.episode_lengths, torch.iinfo(torch.long).max),
            ).min(),
        )
        self.completed_length_max = torch.maximum(
            self.completed_length_max,
            done_lengths.max(),
        )
        self.episode_returns = torch.where(
            dones, torch.zeros_like(self.episode_returns), self.episode_returns
        )
        self.episode_lengths = torch.where(
            dones, torch.zeros_like(self.episode_lengths), self.episode_lengths
        )

        self._record_info(info, dones)

    def compute(self):
        """Return metrics for the current rollout window.

        Metric definitions:
            ``Reward/mean``: Mean per-environment-step reward in this rollout.
            ``Reward/std``: Standard deviation of per-environment-step rewards.
            ``Reward/sum``: Sum of all rewards over all envs in this rollout.
            ``Step/count``: Number of vectorized ``env.step`` calls.
            ``Transition/count``: Number of per-env transitions.
            ``Step/fps``: Per-env transition throughput in this rollout window.
            ``Done/count``: Number of environments that ended in this rollout.
            ``Terminated/count``: Done count caused by task termination.
            ``Timeout/count``: Done count caused by time limit/truncation.
            ``Episode/count``: Number of complete episodes observed in this
                rollout. Episodes that started in previous rollouts are included
                when they finish.
            ``Episode/return_*``: Statistics over completed episode returns.
            ``Episode/length_*``: Statistics over completed episode lengths.
            ``EnvLog/<key>``: Mean of numeric values from ``info["log"][key]``
                sampled according to ``log_on_done_only``.
        """

        metrics = {}
        if self.rollout_transitions > 0:
            count = max(self.rollout_transitions, 1)
            reward_mean = self.rollout_reward_sum / count
            reward_var = self.rollout_reward_sq_sum / count - reward_mean.square()
            fps = self.rollout_transitions / max(
                time.perf_counter() - self.rollout_start_time, 1.0e-12
            )
            metrics.update(
                {
                    "Reward/mean": Scalar(reward_mean),
                    "Reward/std": Scalar(reward_var.clamp_min(0.0).sqrt()),
                    "Reward/sum": Scalar(self.rollout_reward_sum),
                    "Step/count": Scalar(self.rollout_steps),
                    "Transition/count": Scalar(self.rollout_transitions),
                    "Step/fps": Scalar(fps),
                    "Done/count": Scalar(self.terminated_count + self.timeout_count),
                    "Terminated/count": Scalar(self.terminated_count),
                    "Timeout/count": Scalar(self.timeout_count),
                }
            )

        episode_count = self.completed_count
        safe_episode_count = episode_count.clamp_min(1).to(
            dtype=self.completed_return_sum.dtype
        )
        has_episode = episode_count > 0
        metrics.update(
            {
                "Episode/count": Scalar(episode_count),
                "Episode/return_mean": Scalar(
                    self.completed_return_sum / safe_episode_count
                ),
                "Episode/return_min": Scalar(
                    torch.where(
                        has_episode,
                        self.completed_return_min,
                        torch.zeros_like(self.completed_return_min),
                    )
                ),
                "Episode/return_max": Scalar(
                    torch.where(
                        has_episode,
                        self.completed_return_max,
                        torch.zeros_like(self.completed_return_max),
                    )
                ),
                "Episode/length_mean": Scalar(
                    self.completed_length_sum.to(dtype=self.completed_return_sum.dtype)
                    / safe_episode_count
                ),
                "Episode/length_min": Scalar(
                    torch.where(
                        has_episode,
                        self.completed_length_min,
                        torch.zeros_like(self.completed_length_min),
                    )
                ),
                "Episode/length_max": Scalar(
                    torch.where(
                        has_episode,
                        self.completed_length_max,
                        torch.zeros_like(self.completed_length_max),
                    )
                ),
            }
        )

        for key, total in self.log_sums.items():
            count = self.log_counts[key]
            metrics[f"EnvLog/{key}"] = Scalar(total / count.clamp_min(1.0))
        return metrics

    def reset(self, clear_episodes: bool = False):
        """Reset rollout-window accumulators.

        Args:
            clear_episodes: If True, also clears unfinished episode return and
                length state. Use True only for a full environment/session reset.
                Rollout should normally call ``reset(clear_episodes=False)`` so
                episodes spanning multiple rollout windows remain accurate.
        """

        if clear_episodes or self.episode_returns is None:
            self.episode_returns = None
            self.episode_lengths = None
        device = (
            self.episode_returns.device
            if isinstance(self.episode_returns, torch.Tensor)
            else None
        )
        self.rollout_reward_sum = torch.zeros((), device=device)
        self.rollout_reward_sq_sum = torch.zeros((), device=device)
        self.rollout_steps = 0
        self.rollout_transitions = 0
        self.rollout_start_time = time.perf_counter()
        self.terminated_count = torch.zeros((), dtype=torch.long, device=device)
        self.timeout_count = torch.zeros((), dtype=torch.long, device=device)
        self.completed_count = torch.zeros((), dtype=torch.long, device=device)
        self.completed_return_sum = torch.zeros((), device=device)
        self.completed_return_min = torch.full((), torch.inf, device=device)
        self.completed_return_max = torch.full((), -torch.inf, device=device)
        self.completed_length_sum = torch.zeros((), dtype=torch.long, device=device)
        self.completed_length_min = torch.full(
            (), torch.iinfo(torch.long).max, dtype=torch.long, device=device
        )
        self.completed_length_max = torch.zeros((), dtype=torch.long, device=device)
        self.log_sums = {}
        self.log_counts = {}

    def _ensure_episode_state(self, reward: torch.Tensor):
        if self.episode_returns is not None:
            return
        self.num_envs = reward.numel()
        self.episode_returns = torch.zeros_like(reward)
        self.episode_lengths = torch.zeros(
            reward.numel(), dtype=torch.long, device=reward.device
        )

    def _ensure_rollout_accumulators(self, device: torch.device):
        if self.rollout_reward_sum.device == device:
            return
        self.rollout_reward_sum = self.rollout_reward_sum.to(device)
        self.rollout_reward_sq_sum = self.rollout_reward_sq_sum.to(device)
        self.terminated_count = self.terminated_count.to(device)
        self.timeout_count = self.timeout_count.to(device)
        self.completed_count = self.completed_count.to(device)
        self.completed_return_sum = self.completed_return_sum.to(device)
        self.completed_return_min = self.completed_return_min.to(device)
        self.completed_return_max = self.completed_return_max.to(device)
        self.completed_length_sum = self.completed_length_sum.to(device)
        self.completed_length_min = self.completed_length_min.to(device)
        self.completed_length_max = self.completed_length_max.to(device)

    def _reward_per_env(self, rewards, num_envs: int):
        reward = torch.as_tensor(rewards, dtype=torch.float32)
        if reward.numel() % num_envs != 0:
            raise ValueError(
                f"reward with {reward.numel()} values cannot be split across "
                f"{num_envs} environments"
            )
        return reward.reshape(num_envs, -1).sum(dim=1)

    def _record_info(self, info: dict[str, Any] | None, dones: torch.Tensor):
        """Record numeric diagnostics from ``info["log"]``.

        Only the ``log`` subtree is consumed. For Gymnasium-style vector info,
        masks named ``_<key>`` are honored. When a log value has a leading
        environment dimension and ``log_on_done_only`` is True, only done envs
        contribute to the mean.
        """

        if not isinstance(info, dict):
            return
        log = info.get("log")
        if not isinstance(log, dict):
            return
        done_mask = dones if self.log_on_done_only else None
        self._record_info_dict("", log, done_mask=done_mask)

    def _record_info_dict(
        self, prefix: str, info: dict[str, Any], done_mask: torch.Tensor | None
    ):
        for key, value in info.items():
            if str(key).startswith("_"):
                continue
            name = str(key)
            path = name if not prefix else f"{prefix}/{name}"
            if isinstance(value, dict):
                self._record_info_dict(path, value, done_mask=done_mask)
            else:
                mask = info.get(f"_{key}", done_mask)
                self._record_numeric(path, value, mask)

    def _record_numeric(self, path: str, value: Any, mask: Any = None):
        """Accumulate one numeric log value as an arithmetic mean.

        Scalars contribute one sample. Tensors contribute one sample per element
        after optional masking. Non-numeric, empty, and complex values are
        ignored.
        """

        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError):
            return
        if tensor.numel() == 0 or tensor.is_complex():
            return
        tensor = tensor.detach().to(dtype=torch.float32)
        flat = tensor.reshape(-1)
        selected = torch.isfinite(flat)
        if mask is not None:
            try:
                mask = torch.as_tensor(mask, device=tensor.device).bool()
            except (TypeError, ValueError):
                mask = None
            if mask is not None:
                mask = mask.reshape(-1)
                if tensor.ndim == 0:
                    selected = selected & mask.any()
                elif mask.numel() == tensor.shape[0]:
                    selected = selected & mask.reshape(-1, 1).expand_as(
                        tensor.reshape(mask.numel(), -1)
                    ).reshape(-1)
                elif mask.numel() == tensor.numel():
                    selected = selected & mask
        total = torch.where(selected, flat, torch.zeros_like(flat)).sum()
        count = selected.sum().to(dtype=torch.float32)
        if path not in self.log_sums:
            self.log_sums[path] = total
            self.log_counts[path] = count
        else:
            if self.log_sums[path].device != total.device:
                self.log_sums[path] = self.log_sums[path].to(total.device)
                self.log_counts[path] = self.log_counts[path].to(total.device)
            self.log_sums[path] = self.log_sums[path] + total
            self.log_counts[path] = self.log_counts[path] + count
