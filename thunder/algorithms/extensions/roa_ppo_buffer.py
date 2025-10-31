from dataclasses import dataclass
from typing import Any, Generator, Optional, Tuple

import torch

from thunder.rl.utils import (
    get_hidden_mask,
    get_trajectory_lengths,
    get_trajectory_mask,
    split_trajectory,
)

from ..ppo.buffer import RolloutBuffer
from .advantage_mix_buffer import AdvantageMixBuffer


class RoaBuffer(RolloutBuffer):
    """Rollout Buffer for Regular Online Adaptation"""

    @dataclass(slots=True)
    class Transition(RolloutBuffer.Transition):
        obs_enc_hidden: Any = None
        state_enc_hidden: Any = None
        obs_latent: Any = None
        state_latent: Any = None

    @dataclass(slots=True)
    class Batch(RolloutBuffer.Batch):
        # assigned every ppo update
        obs_enc_hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] = None
        state_enc_hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] = None
        obs_latent: torch.Tensor = None
        state_latent: torch.Tensor = None

    def __init__(self, length: int, device: torch.device) -> None:
        super().__init__(length, device)
        self.obs_latent: Optional[torch.Tensor] = None
        self.state_latent: Optional[torch.Tensor] = None
        self.obs_enc_hidden: Optional[list[torch.Tensor]] = None
        self.state_enc_hidden: Optional[list[torch.Tensor]] = None

    def _lazy_init(self, t: Transition):
        super()._lazy_init(t)
        self.obs_latent = self.zeros_th(self.length, *t.obs_latent.shape)
        self.state_latent = self.zeros_th(self.length, *t.state_latent.shape)

    def _add(self, t: Transition) -> None:
        super()._add(t)
        self.obs_latent[self.step] = t.obs_latent
        self.state_latent[self.step] = t.state_latent
        self.save_hidden_states(t.obs_enc_hidden, "obs_enc_hidden")
        self.save_hidden_states(t.state_enc_hidden, "state_enc_hidden")

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
        For the hidden state of the recurrent neural network, we only need the starting
        hidden state of each sequence
        """
        if batch is None:
            batch = self.Batch()
        batch.hidden_masks = get_hidden_mask(self.dones[:, indices])
        batch.trajectory_lengths = get_trajectory_lengths(self.dones[:, indices])
        batch.actor_obs = split_trajectory(self.actor_obs[:, indices], batch.trajectory_lengths)
        batch.critic_obs = split_trajectory(self.critic_obs[:, indices], batch.trajectory_lengths)
        batch.obs_enc_hidden = self.slice_hidden(self.obs_enc_hidden, indices, batch.hidden_masks)
        batch.state_enc_hidden = self.slice_hidden(
            self.state_enc_hidden, indices, batch.hidden_masks
        )
        batch.critic_hidden = self.slice_hidden(self.critic_hidden, indices, batch.hidden_masks)
        batch.obs_latent = self.obs_latent[:, indices]
        batch.state_latent = self.state_latent[:, indices]
        batch.actions = self.actions[:, indices]
        batch.sigma = self.sigma[:, indices]
        batch.mu = self.mu[:, indices]
        batch.values = self.values[:, indices]
        batch.advantages = self.advantages[:, indices]
        batch.returns = self.returns[:, indices]
        batch.actions_log_prob = self.actions_log_prob[:, indices]
        batch.trajectory_masks = get_trajectory_mask(batch.trajectory_lengths)
        return batch


class MixRoaBuffer(AdvantageMixBuffer):
    """Mix-Advantage Rollout Buffer for Regular Online Adaptation"""

    @dataclass(slots=True)
    class Transition(RolloutBuffer.Transition):
        obs_enc_hidden: Any = None
        state_enc_hidden: Any = None
        obs_latent: Any = None
        state_latent: Any = None

    @dataclass(slots=True)
    class Batch(RolloutBuffer.Batch):
        # assigned every ppo update
        obs_enc_hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] = None
        state_enc_hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] = None
        obs_latent: torch.Tensor = None
        state_latent: torch.Tensor = None
        curr_obs_latent: torch.Tensor = None
        curr_state_latent: torch.Tensor = None

    def __init__(self, length: int, device: torch.device) -> None:
        super().__init__(length, device)
        self.obs_latent: Optional[torch.Tensor] = None
        self.state_latent: Optional[torch.Tensor] = None
        self.obs_enc_hidden: Optional[list[torch.Tensor]] = None
        self.state_enc_hidden: Optional[list[torch.Tensor]] = None

    def _lazy_init(self, t: Transition):
        super()._lazy_init(t)
        self.obs_latent = self.zeros_th(self.length, *t.obs_latent.shape)
        self.state_latent = self.zeros_th(self.length, *t.state_latent.shape)

    def _add(self, t: Transition) -> None:
        super()._add(t)
        self.obs_latent[self.step] = t.obs_latent
        self.state_latent[self.step] = t.state_latent
        self.save_hidden_states(t.obs_enc_hidden, "obs_enc_hidden")
        self.save_hidden_states(t.state_enc_hidden, "state_enc_hidden")

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
        batch.obs_enc_hidden = self.slice_hidden(self.obs_enc_hidden, indices, batch.hidden_masks)
        batch.state_enc_hidden = self.slice_hidden(
            self.state_enc_hidden, indices, batch.hidden_masks
        )
        batch.critic_hidden = self.slice_hidden(self.critic_hidden, indices, batch.hidden_masks)
        batch.obs_latent = self.obs_latent[:, indices]
        batch.state_latent = self.state_latent[:, indices]
        batch.actions = self.actions[:, indices]
        batch.sigma = self.sigma[:, indices]
        batch.mu = self.mu[:, indices]
        batch.values = self.values[:, indices]
        batch.advantages = self.advantages[:, indices]
        batch.returns = self.returns[:, indices]
        batch.actions_log_prob = self.actions_log_prob[:, indices]
        batch.trajectory_masks = get_trajectory_mask(batch.trajectory_lengths)
        return batch
