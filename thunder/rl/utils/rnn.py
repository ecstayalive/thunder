from typing import Iterator, Optional, Tuple, overload

import torch
import torch.nn as nn

from thunder.nn import Conv2dBlock, EmbedLstmMlp, GruMlp, LstmMlp

__all__ = [
    "get_trajectory_lengths",
    "split_trajectory",
    "get_trajectory_mask",
    "unpad_trajectory",
    "get_hidden_mask",
    "EmbedConvRMlp",
    "DimAdaptRMlp",
    "is_recurrent",
    "any_recurrent",
    "all_recurrent",
]


@torch.jit.script
def get_trajectory_lengths(dones: torch.Tensor) -> torch.Tensor:
    """The function takes a tensor of boolean values indicating
    whether a trajectory is done or not, and returns a tensor
    containing the lengths of each trajectory. This function is
    used to process a batch of dataset.
    Args:
        dones: The `dones` parameter is a tensor representing whether each
            step in a trajectory is a terminal step or not.
            :shape: `[time_steps, num_envs(mini-batch_size), 1]`.
    Returns:
        a tensor containing the lengths of trajectories. :shape: `[1, num_trajectories]`
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, sequence_length, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).flatten()

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (
            torch.tensor([-1], dtype=torch.int64, device=dones.device),
            flat_dones.nonzero().squeeze(),
        )
    )

    # done_indices = torch.nn.functional.pad(flat_dones.nonzero().squeeze(), (1, 0), value=-1)
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    return trajectory_lengths


@torch.jit.script
def split_trajectory(obs: torch.Tensor, traj_len: torch.Tensor) -> torch.Tensor:
    """The function observation data and a tensor of trajectory lengths,
    and returns a padded tensor of trajectories.
    Args:
        obs: a tensor representing the observations of a trajectory.
            :shape: `[time_steps, num_envs(mini-batch_size), ob_dimension]`
            :type: `torch.Tensor`
        traj_len: a tensor that represents the length of each trajectory.
            :shape: `[1, num_trajectories]`
            :type: `torch.Tensor`
    Returns:
        the padded trajectories, :shape: `[max_time_steps, num_trajectories, ...]`
    """
    # Extract the individual trajectories
    lengths_list: list[int] = traj_len.tolist()
    trajectories: list[torch.Tensor] = torch.split(obs.transpose(1, 0).flatten(0, 1), lengths_list)
    # padded_trajectories' shape = [max_time_steps, num_trajectories, ...]
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    return padded_trajectories


@torch.jit.script
def get_trajectory_mask(traj_len: torch.Tensor) -> torch.Tensor:
    """This function returns a mask indicating which elements
    in a trajectory are valid based on their length.
    Args:
        traj_len: a tensor representing the length of each trajectory.
            It is used to create a mask that indicates which elements
            of each trajectory are valid
            :shape: `[1, num_trajectories]`
            :type: `torch.Tensor`
    Returns:
        a tensor of trajectory masks.
        :shape: `[max_time_steps, num_trajectories]`
        :type: `torch.Tensor`,  where the element are `bool` type
    """
    trajectory_masks = traj_len > torch.arange(
        0, traj_len.max(), device=traj_len.device
    ).unsqueeze_(1)

    return trajectory_masks


@torch.jit.script
def unpad_trajectory(
    trajectories: torch.Tensor, masks: torch.Tensor, sequence_len: int
) -> torch.Tensor:
    """This function performs the inverse operation of
    the `split_trajectory` function by removing padding
    from a tensor of trajectories.
    Args:
        trajectories: a tensor representing the trajectories.
            :shape: `[max_time_steps, num_trajectories, ...]`
            :type: `torch.Tensor`
        masks: a tensor that marks which parts of the trajectory
            are padded and which ones are the original real trajectories.
            :shape: `[max_time_steps, num_trajectories]`
            :type: `torch.Tensor`, where the elements are `torch.bool`
        sequence_len: which represents the true time step of the collecting data.
            :type: `int`
    Returns:
        a tensor represents the origin, not padded trajectory or output of the network.
            :shape: `[sequence_len, num_envs(mini-batch_size), ...]`
            :type: `torch.Tensor`
    """
    # Need to transpose before to ensure its order
    # and after the masking need to transpose again to have proper reshaping
    # [max_time_steps, num_trajectories, ...] ->
    # [true_time_step * num_envs(mini-batch_size), ...] ->
    # [num_envs(mini-batch_size), true_time_step, ...] ->
    # [true_time_step, num_envs(mini-batch_size), ...]
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .reshape(-1, sequence_len, trajectories.shape[-1])
        .transpose(1, 0)
    )


@torch.jit.script
def get_hidden_mask(dones: torch.Tensor) -> torch.Tensor:
    """This function takes a tensor of boolean values indicating
    whether each step is a terminal step, and returns a tensor
    indicating whether the previous step was a terminal step.
    Args:
        dones: The `dones` parameter is a tensor representing
            whether each step in a sequence is the last step of
            an episode.
            :shape: `[time_steps, num_envs(mini-batch_size), 1]`
    Returns:
        a bool tensor that represents a mask indicating whether the
        previous time step was a "done" state, that is standing whether
        the current time step was a beginning state.
        :shape: `[mini-batch_size, time_steps]`
    """
    dones = dones.squeeze(-1)
    last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    last_was_done[1:] = dones[:-1]
    last_was_done[0] = True
    return last_was_done.permute(1, 0)


class EmbedConvRMlp(nn.Module):
    """
    Args:
        conv_head: The convolutional head of the network.
        recurrent_mlp: The recurrent multi-layer perception.
    """

    def __init__(self, conv_head: Conv2dBlock, recurrent_mlp: LstmMlp | GruMlp | EmbedLstmMlp):
        super().__init__()
        self.conv_in_shape = conv_head.in_shape
        self.conv_in_features = conv_head.in_features
        self.conv_head = conv_head
        self.recurrent_mlp = recurrent_mlp

    def forward(self, input: torch.Tensor, hx) -> Tuple[torch.Tensor, ...]:
        seq_len, batch_size, _ = input.shape
        rnn_input: torch.Tensor = input[..., : -self.conv_in_features]
        conv_input: torch.Tensor = (
            input[..., -self.conv_in_features :]
            .unflatten(-1, self.conv_in_shape)
            .flatten(end_dim=1)
        )
        conv_output: torch.Tensor = (
            self.conv_head(conv_input)
            .flatten(start_dim=-3, end_dim=-1)
            .unflatten(0, (seq_len, batch_size))
        )
        input = torch.cat([rnn_input, conv_output], dim=-1)
        return self.recurrent_mlp(input, hx)

    def scriptable(self):
        """ """
        ...


class DimAdaptRMlp(nn.Module):
    """Dimension Adaptive Recurrent Multi-Layer Perception
    Because we created thousands of simulation instances
    for training, during the exploration phase, the dimensions
    of the data observed in the simulation are `[num_envs, features_dim]`.
    However, for recurrent neural networks, the `num_envs` of the data in
    data dimension means the time step, This is obviously wrong,
    so this situation needs to be dealt with. This problem will not
    occur during the training and verification phases. And it is the
    reason why this module exists
    Args:
        recurrent_mlp: The recurrent multi-layer perception
        normalizer: Use ```RunningNorm1d``` to normalize the input data.
    """

    is_recurrent = True

    def __init__(self, recurrent_mlp: LstmMlp | GruMlp | EmbedConvRMlp):
        super().__init__()
        self.recurrent_mlp = recurrent_mlp

    @overload
    def forward(
        self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """For LSTM"""
        ...

    @overload
    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For GRU"""
        ...

    def forward(self, input: torch.Tensor, hx=None):
        if input.dim() <= 2:
            input = input.unsqueeze(0)
            output, hidden = self.recurrent_mlp(input, hx)
            return output.squeeze(0), hidden
        else:
            output, hidden = self.recurrent_mlp(input, hx)
            return output, hidden

    def scriptable(self):
        if isinstance(self.recurrent_mlp, EmbedConvRMlp):
            return self.recurrent_mlp.scriptable()
        else:
            return self.recurrent_mlp


def is_recurrent(network: nn.Module) -> bool:
    """Determine whether the given network contains a recurrent
    neural network module. If the network does, then the network
    is a recurrent neural network.
    Returns:
        Whether the given network contains a recurrent network module.
    """
    # if the network includes some information
    # then return the information directly
    if hasattr(network, "is_recurrent"):
        return network.is_recurrent
    for module in network.modules():
        if isinstance(module, nn.RNNBase):
            return True
    return False


def any_recurrent(*networks: Iterator[nn.Module]) -> bool:
    """Determine whether the given networks contain a recurrent
    neural network.
    Returns:
        Whether the given networks contains a recurrent network.
    """
    return any(map(is_recurrent, networks))


def all_recurrent(*networks: Iterator[nn.Module]) -> bool:
    """Determine whether the given networks are all recurrent
    neural network.
    Returns:
        Whether the given networks are all recurrent network.
    """
    return all(map(is_recurrent, networks))
