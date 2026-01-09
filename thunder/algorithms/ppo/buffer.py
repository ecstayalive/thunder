from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, Tuple

import torch

from thunder.rl.buffer import Buffer
from thunder.rl.utils import (
    get_hidden_mask,
    get_trajectory_lengths,
    get_trajectory_mask,
    split_trajectory,
)


class RolloutBuffer(Buffer):
    @dataclass(slots=True)
    class Transition(Buffer.Transition):
        values: Any = None
        next_values: Any = None
        mu: Any = None
        sigma: Any = None
        actions_log_prob: Any = None
        actor_hidden: Any = None
        critic_hidden: Any = None

    @dataclass(slots=True)
    class Batch:
        actor_obs: torch.Tensor = None
        critic_obs: torch.Tensor = None
        actions: torch.Tensor = None
        sigma: torch.Tensor = None
        mu: torch.Tensor = None
        values: torch.Tensor = None
        advantages: torch.Tensor = None
        returns: torch.Tensor = None
        actions_log_prob: torch.Tensor = None
        # assigned every ppo update
        curr_mu: torch.Tensor = None
        curr_sigma: torch.Tensor = None
        curr_values: torch.Tensor = None
        # RNNs only
        actor_hidden: torch.Tensor = None
        critic_hidden: torch.Tensor = None
        trajectory_masks: torch.Tensor = None
        hidden_masks: torch.Tensor = None
        trajectory_lengths: torch.Tensor = None

    def __init__(self, length, device):
        super().__init__(length, device)

        # For PPO
        self.actions_log_prob: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.next_values: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        self.advantages: Optional[torch.Tensor] = None
        self.mu: Optional[torch.Tensor] = None
        self.sigma: Optional[torch.Tensor] = None

        # rnn
        self.actor_hidden: Optional[list[torch.Tensor]] = None
        self.critic_hidden: Optional[list[torch.Tensor]] = None

    def _lazy_init(self, t: Transition):
        super()._lazy_init(t)
        self.values = torch.zeros_like(self.rewards)
        self.next_values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.actions_log_prob = torch.zeros_like(self.rewards)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)

    def _add(self, t: Transition):
        super()._add(t)
        self.values[self.step] = self.as_th(t.values)
        self.next_values[self.step] = self.as_th(t.next_values)
        self.mu[self.step] = self.as_th(t.mu)
        self.sigma[self.step] = self.as_th(t.sigma)
        self.actions_log_prob[self.step] = self.as_th(t.actions_log_prob)
        self.save_hidden_states(t.actor_hidden, "actor_hidden")
        self.save_hidden_states(t.critic_hidden, "critic_hidden")

    def clear(self):
        self.step = 0

    def compute_returns(self, gamma, lamda, normalize_adv=True):
        """Calculates GAE
        Calculates the mixture GAE. It is assumed that the
        reward and value of the system are composed of two
        parts.
        Args:
            gamma: A parameter for reducing the summary of the reward.
                Determine the field of view length of the agent.
            lamda: Parameters used to estimate advantage in GAE
            normalize_adv: Whether to perform advantage regularization
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
        if normalize_adv:
            std, mean = torch.std_mean(self.advantages)
            self.advantages = (self.advantages - mean) / (std + 1e-8)

    @property
    def sampler(self):
        if self.actor_hidden is None and self.critic_hidden is None:
            return self._sampler
        else:
            return self._recurrent_sampler

    def _sampler(
        self, num_mini_batches, num_repetitions, batch: Batch = None
    ) -> Generator[Batch, None, None]:
        batch_size = self.num_envs * self.length
        mini_batch_size = batch_size // num_mini_batches
        collections = (
            self.actor_obs.flatten(0, 1),
            self.critic_obs.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.sigma.flatten(0, 1),
            self.mu.flatten(0, 1),
            self.values.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.actions_log_prob.flatten(0, 1),
        )

        for _ in range(num_repetitions):
            rand_indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
            for i in range(num_mini_batches):
                indices = rand_indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                if batch is None:
                    batch = self.Batch(*[item[indices] for item in collections])
                else:
                    batch.__init__(*[item[indices] for item in collections])
                yield batch

    def _recurrent_sampler(self, num_mini_batches, num_repetitions) -> Generator[Batch, None, None]:
        mini_batch_size = self.num_envs // num_mini_batches
        batches = []

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size
            batches.append(self._recurrent_minibatch_slicer(slice(start, stop)))

        yield from batches * num_repetitions

    def _recurrent_sampler_with_shuffle(self, num_mini_batches, num_repetitions):
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_repetitions):
            rand_indices = torch.randperm(
                num_mini_batches * mini_batch_size, device=self.device
            ).reshape(num_mini_batches, -1)
            for i in range(num_mini_batches):
                yield self._recurrent_minibatch_slicer(rand_indices[i])

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
        batch.actor_obs, batch.actor_hidden = self.slice_hidden_relevant(
            self.actor_obs, self.actor_hidden, indices, batch.hidden_masks, batch.trajectory_lengths
        )
        batch.critic_obs, batch.critic_hidden = self.slice_hidden_relevant(
            self.critic_obs,
            self.critic_hidden,
            indices,
            batch.hidden_masks,
            batch.trajectory_lengths,
        )

        batch.actions = self.actions[:, indices]
        batch.sigma = self.sigma[:, indices]
        batch.mu = self.mu[:, indices]
        batch.values = self.values[:, indices]
        batch.advantages = self.advantages[:, indices]
        batch.returns = self.returns[:, indices]
        batch.actions_log_prob = self.actions_log_prob[:, indices]
        batch.trajectory_masks = get_trajectory_mask(batch.trajectory_lengths)
        return batch

    @staticmethod
    def slice_hidden_relevant(
        obs: torch.Tensor,
        hidden: list[torch.Tensor],
        indices: slice,
        mask: torch.Tensor,
        traj_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """This function takes in observations, hidden states, indices, masks,
        and trajectory lengths, and returns a sliced version of the observations
        and hidden states based on the given indices and masks.
        Args:
            obs: observations. :shape: `[time_steps, num_envs, obs_dim]
            hidden: a list of tensor representing the hidden state of the
                recurrent neural network.
                :shape: `[time_steps, num_layers, num_envs, hidden_dim]`
            indices: a slice object specifies which elements of the `obs` tensor should
                be selected. It is used to slice the `obs` tensor along the second dimension
            mask: a boolean mask that indicates which elements of the hidden state should be
                selected. It is used to filter out hidden states that correspond to done
                trajectories
            traj_len: the length of a trajectory. It is used to split the `obs` tensor
                into batches of trajectories
        Returns:
            a tuple including `obs_batch` and `hidden_batch`.
            obs_batch: the batch of observations
                :shape: `[max_time_steps, num_trajectories, ...]`
            hidden_batch: the hidden variables corresponding to the beginning of the trajectory
                :shape: `[num_layers, batch, hidden_dim]`
        """
        # TODO: Simplify pipeline by saving trajectory lengths and init hidden
        if hidden is None:
            return obs[:, indices], None

        obs_batch = split_trajectory(obs[:, indices], traj_len)
        hidden_batch = RolloutBuffer.slice_hidden(hidden, indices, mask)
        return obs_batch, hidden_batch

    @staticmethod
    def slice_hidden(
        hidden: list[torch.Tensor], indices: slice, mask: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """This function takes in hidden state features, indices, masks, and
        returns a sliced version of the hidden states based on the given indices
        and masks.

        Args:
            hidden: A list of tensor representing the hidden state of the
                recurrent neural network.
                :shape: `[time_steps, num_layers, num_envs, hidden_dim]`
            indices: A slice object specifies which elements of the `obs` tensor should
                be selected. It is used to slice the `obs` tensor along the second dimension
            mask: A boolean mask that indicates which elements of the hidden state should be
                selected. It is used to filter out hidden states that correspond to done
                trajectories
        Process:
            Original shape: [time_steps, num_layers, num_envs(mini-batch_size), hidden_dim]
            Reshape to: [num_envs(mini-batch_size), time_steps, num_layers, hidden_dim]
            Then take only time steps after dones (flattens num envs and time dimensions),
            because lstm accept hidden'shape is [num_layers, batch, hidden_dim], thus reshape to it.
        """
        slice_hidden_fn = (
            lambda hidden: hidden[:, :, indices]
            .permute(2, 0, 1, 3)[mask]
            .transpose(1, 0)
            .contiguous()
        )
        hidden_batch = tuple(map(slice_hidden_fn, hidden))
        if len(hidden_batch) == 1:
            hidden_batch = hidden_batch[0]
        return hidden_batch
