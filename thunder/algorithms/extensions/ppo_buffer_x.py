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


class PpoBufferX(RolloutBuffer):
    """ """

    @dataclass(slots=True)
    class Transition(RolloutBuffer.Transition):
        # NB: Does the Transition need storing denoise_state
        denoise_state: Any = None
        state_encoder_hidden: Any = None
        # action_decoder_hidden: Any = None

    @dataclass(slots=True)
    class Batch(RolloutBuffer.Batch):
        # assigned every ppo update
        curr_denoise_state: torch.Tensor = None
        state_encoder_hidden: torch.Tensor | Tuple[torch.Tensor, ...] = None
        # action_decoder_hidden: torch.Tensor | Tuple[torch.Tensor, ...] = None

    def __init__(self, length, device):
        super().__init__(length, device)
        self.denoise_state: Optional[torch.Tensor] = None
        self.state_encoder_hidden: Optional[list[torch.Tensor]] = None
        # self.action_decoder_hidden: Optional[list[torch.Tensor]] = None

    def _lazy_init(self, t: Transition):
        super()._lazy_init(t)
        self.denoise_state = torch.zeros_like(self._critic_obs)

    def _add(self, t: Transition):
        super()._add(t)
        self.save_hidden_states(t.state_encoder_hidden, "state_encoder_hidden")
        # self.save_hidden_states(t.action_decoder_hidden, "action_decoder_hidden")
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
        batch.state_encoder_hidden = self.slice_hidden(
            self.state_encoder_hidden, indices, batch.hidden_masks
        )
        # batch.action_decoder_hidden = self.slice_hidden_features(
        #     self.action_decoder_hidden, indices, batch.hidden_masks
        # )
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
