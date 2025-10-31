from dataclasses import dataclass
from typing import Any, Generator, Optional, Tuple

import torch

from thunder.rl.utils import (
    get_hidden_mask,
    get_trajectory_lengths,
    get_trajectory_mask,
    split_trajectory,
    unpad_trajectory,
)

from ..ppo.buffer import RolloutBuffer


class DenoiseBuffer(RolloutBuffer):
    """ """

    @dataclass(slots=True)
    class Transition(RolloutBuffer.Transition):
        # NB: Does the Transition need storing denoise_state
        latent_features: Any = None
        next_latent_features: Any = None
        denoise_state: Any = None

    @dataclass(slots=True)
    class Batch(RolloutBuffer.Batch):
        # assigned every ppo update
        latent_features: torch.Tensor = None
        next_latent_features: torch.Tensor = None
        curr_reconstruction_state: torch.Tensor = None

    def __init__(self, length, device):
        super().__init__(length, device)
        self.latent_features: Optional[torch.Tensor] = None
        self.next_latent_features: Optional[torch.Tensor] = None
        self.construction_state: Optional[torch.Tensor] = None

    def _lazy_init(self, t: Transition):
        super()._lazy_init(t)
        self.latent_features = torch.zeros(self.length, *t.latent_features.shape)
        self.next_latent_features = torch.zeros_like(self.latent_features)
        self.construction_state = torch.zeros_like(self._critic_obs)

    def _add(self, t: Transition):
        super()._add(t)
        self.latent_features[self.step] = self.as_th(t.latent_features)
        self.next_latent_features[self.step] = self.as_th(t.next_latent_features)
        # self.denoise_state[self.step] = self.as_th(t.denoise_state)

    def _recurrent_sampler(self, num_mini_batches, num_repetitions) -> Generator[Batch, None, None]:
        mini_batch_size = self.num_envs // num_mini_batches
        batches = []

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size
            batches.append(self._recurrent_minibatch_slicer(slice(start, stop)))

        yield from batches * num_repetitions

    def _recurrent_minibatch_slicer(self, indices: slice, batch: Batch = None) -> Batch:
        """This function slices and organizes data into batches for recurrent neural networks.

        Args:
            indices: The `indices` parameter is a list or array of indices that specify which
                samples from the data should be included in the batch. These indices are used
                to slice the data along the batch dimension
            batch: The `batch` parameter is an instance of the `Batch` class. It is used to
                store the sliced data for a mini-batch
        Returns:
            a mini-batch of data.

        NB: For the hidden state of the recurrent neural network, we only need the starting
        hidden state of each sequence
        """
        if batch is None:
            batch = self.Batch()
        batch.hidden_masks = get_hidden_mask(self.dones[:, indices])
        batch.trajectory_lengths = get_trajectory_lengths(self.dones[:, indices])
        batch.actor_obs = split_trajectory(self.actor_obs[:, indices], batch.trajectory_lengths)
        batch.critic_obs = split_trajectory(self._critic_obs[:, indices], batch.trajectory_lengths)
        batch.actor_hidden = self.slice_hidden(self.actor_hidden, indices, batch.hidden_masks)
        batch.critic_hidden = self.slice_hidden(self.critic_hidden, indices, batch.hidden_masks)

        batch.actions = self.actions[:, indices]
        batch.sigma = self.sigma[:, indices]
        batch.mu = self.mu[:, indices]
        batch.values = self.values[:, indices]
        batch.advantages = self.advantages[:, indices]
        batch.returns = self.returns[:, indices]
        batch.actions_log_prob = self.actions_log_prob[:, indices]
        batch.trajectory_masks = get_trajectory_mask(batch.trajectory_lengths)
        return batch


class DenoiseGgfBuffer(DenoiseBuffer):
    """This buffer is a custom buffer, and
    the default reward is a vector. One of the problems
    is modeled as an infinite mdp problem, and the other
    is modeled as a limited mdp problem.
    """

    def __init__(self, length: int, device: torch.device) -> None:
        super().__init__(length, device)
        self.ggf_weight = torch.tensor([0.63, 0.37], device=self.device)
        self._eps = torch.tensor([1e-8], device=self.device)

    def _lazy_init(self, t: RolloutBuffer.Transition) -> None:
        super()._lazy_init(t)
        self._norm_advantages_dims = tuple(i for i in range(self.advantages.dim() - 1))
        self.actions_log_prob = self.zeros_th(*self.rewards.shape[:-1], 1)
        self.advantages = self.zeros_th(*self.rewards.shape[:-1], 1)

    def compute_returns(
        self, gamma: float, lamda: float, normalize_adv: bool = True
    ) -> torch.Tensor:
        """Calculates GAE
        Calculates the mixture GAE. It is assumed that the
        reward and value of the system are composed of two
        parts.
        """
        advantage = 0

        with torch.inference_mode():
            for step in reversed(range(self.length)):
                mask = self.dones[step].logical_not().logical_or(self.timeouts[step]) * gamma
                delta = self.rewards[step] + mask * self.next_values[step] - self.values[step]
                advantage = delta + mask * lamda * advantage
                self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages, _ = torch.sort(self.advantages, dim=-1)
        self.advantages = torch.sum(self.advantages * self.ggf_weight, dim=-1, keepdim=True)
        if normalize_adv:
            std, mean = torch.std_mean(self.advantages)
            self.advantages = (self.advantages - mean) / torch.max(std, self._eps)


class DenoiseMixAdvBuffer(DenoiseBuffer):
    """This buffer is a custom buffer, and
    the default reward is a vector. One of the problems
    is modeled as an infinite mdp problem, and the other
    is modeled as a limited mdp problem.
    """

    def __init__(self, length: int, device: torch.device) -> None:
        super().__init__(length, device)

    def _lazy_init(self, t: RolloutBuffer.Transition):
        super()._lazy_init(t)
        self._norm_advantages_dims = tuple(i for i in range(self.advantages.dim() - 1))
        self.actions_log_prob = torch.zeros_like(self.actions)
        self._default_mixing_index = (
            torch.arange(1, self.length + 1, dtype=torch.float32, device=self.device)
            .view([-1, 1, 1])
            .repeat([1, *self.dones.shape[1:]])
        )
        self._last_mixing_index = self.zeros_th(*self.dones.shape[1:])
        self._last_dones = self.zeros_th(*self.dones.shape[1:])

    def compute_returns(self, gamma: float, lamda: float, normalize_adv: bool = True) -> None:
        """Calculates GAE
        Calculates the mixture GAE. It is assumed that the
        reward and value of the system are composed of two
        parts.
        Args:
            gamma: A parameter for reducing the summary of the reward.
                Determine the field of view length of the agent.
            lamda: Parameters used to estimate advantage in GAE
            normalize_adv: Whether to perform advantage regularization
        Property:
            beta: In mixed gradient advantage, the parameter
                used to mix the advantages of two tasks, the
                value is between 0 and 1
        """
        advantage = 0

        with torch.inference_mode():
            for step in reversed(range(self.length)):
                mask = self.dones[step].logical_not().logical_or(self.timeouts[step]) * gamma
                delta = self.rewards[step] + mask * self.next_values[step] - self.values[step]
                advantage = delta + mask * lamda * advantage
                self.returns[step] = advantage + self.values[step]
        separate_advantages = self.returns - self.values
        beta = self.calculate_beta(self.dones)
        self.advantages = separate_advantages + beta * separate_advantages.flip((-1,))
        # Compute and normalize the advantages
        if normalize_adv:
            # std, mean = torch.std_mean(self.advantages)
            std, mean = torch.std_mean(self.advantages, dim=self._norm_advantages_dims)
            self.advantages = (self.advantages - mean) / (std + 1e-8)

    def calculate_beta(self, dones: torch.Tensor, mixing_time_steps: int = 1000) -> torch.Tensor:
        """Calculate the coefficient of the mixing advantage"""
        reset_flag = torch.zeros_like(dones)
        reset_flag[1:] = dones[:-1]
        mixing_index = (
            self._last_mixing_index * self._last_dones.logical_not() + self._default_mixing_index
        )
        traj_len = get_trajectory_lengths(dones)
        traj_mask = get_trajectory_mask(traj_len)
        reset_flag_traj = split_trajectory(reset_flag, traj_len)
        mixing_index_traj = split_trajectory(mixing_index, traj_len)
        mixing_index_traj[:, reset_flag_traj.any(dim=0)] -= (
            mixing_index_traj[0, reset_flag_traj.any(dim=0)] - 1
        )
        mixing_index = unpad_trajectory(mixing_index_traj, traj_mask, self.length)
        self._last_mixing_index = mixing_index[-1]
        self._last_dones = dones[-1]
        beta = torch.clamp_max(mixing_index / mixing_time_steps, 1.0)
        return beta
