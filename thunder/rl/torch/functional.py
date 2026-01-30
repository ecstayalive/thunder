from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def all_reduce(tensor, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        if op == "AVG":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        elif op == "MAX":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


@torch.inference_mode()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Softly update the parameters of target module
    towards the parameters of source module.
    """

    for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
        tgt.data.mul_(1 - tau).add_(src.data, alpha=tau)


@torch.compile(mode="max-autotune")
def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = torch.sum(
        torch.log(sigma2 / sigma1 + 1.0e-5)
        + (torch.square(sigma1) + torch.square(mu1 - mu2)) / (2.0 * torch.square(sigma2))
        - 0.5,
        dim=-1,
    )
    return kl.mean()


@torch.jit.script
def get_trajectory_lengths(dones: torch.Tensor) -> torch.Tensor:
    """The function takes a tensor of boolean values indicating
    whether a trajectory is done or not, and returns a tensor
    containing the lengths of each trajectory. This function is
    used to process a batch of dataset.
    Args:
        dones: The `dones` parameter is a tensor representing whether each
            step in a trajectory is a terminal step or not.
            :shape: `[num_envs(mini-batch_size), time_steps , 1]`.
    Returns:
        a tensor containing the lengths of trajectories. :shape: `[num_trajectories, ]`
    """
    dones = dones.clone()
    dones[:, -1] = 1
    flat_dones = dones.flatten()
    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (
            torch.tensor([-1], dtype=torch.int64, device=dones.device),
            flat_dones.nonzero().view(-1),
        )
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    return trajectory_lengths


@torch.jit.script
def split_trajectory(
    obs: torch.Tensor, traj_len: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The function observation data and a tensor of trajectory lengths,
    and returns a padded tensor of trajectories.
    Args:
        obs: a tensor representing the observations of a trajectory.
            :shape: `[num_envs(mini-batch_size), time_steps, ob_dimension]`
            :type: `torch.Tensor`
        traj_len: a tensor that represents the length of each trajectory.
            :shape: `[num_trajectories, ]`
            :type: `torch.Tensor`
    Returns:
        the padded trajectories, :shape: `[num_trajectories, max_time_steps, ...]`
        mask: `[Num_Trajs, Max_Len]` (Boolean mask of valid steps)
    """
    flat_obs = obs.flatten(0, 1)
    num_trajs = traj_len.shape[0]
    max_len = int(traj_len.max().item())
    embedding_dim = flat_obs.shape[1]
    row_indices = torch.arange(max_len, device=obs.device).unsqueeze(0)  # [1, MaxLen]
    # [NumTrajs, 1] => [NumTrajs, MaxLen]
    mask = traj_len.unsqueeze(1) > row_indices
    padded_trajectories = torch.zeros(
        (num_trajs, max_len, embedding_dim), dtype=flat_obs.dtype, device=flat_obs.device
    )
    padded_trajectories[mask] = flat_obs
    return padded_trajectories, mask


@torch.jit.script
def get_trajectory_mask(traj_len: torch.Tensor) -> torch.Tensor:
    """This function returns a mask indicating which elements
    in a trajectory are valid based on their length.
    Args:
        traj_len: a tensor representing the length of each trajectory.
            It is used to create a mask that indicates which elements
            of each trajectory are valid
            :shape: `[num_trajectories, ]`
            :type: `torch.Tensor`
    Returns:
        a tensor of trajectory masks.
        :shape: `[num_trajectories, max_time_steps]`
        :type: `torch.Tensor`,  where the element are `bool` type
    """
    trajectory_masks = traj_len.unsqueeze(1) > torch.arange(
        0, traj_len.max(), device=traj_len.device
    ).unsqueeze_(0)

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
            :shape: `[num_trajectories, max_time_steps, ...]`
            :type: `torch.Tensor`
        masks: a tensor that marks which parts of the trajectory
            are padded and which ones are the original real trajectories.
            :shape: `[num_trajectories, max_time_steps]`
            :type: `torch.Tensor`, where the elements are `torch.bool`
        sequence_len: which represents the true time step of the collecting data.
            :type: `int`
    Returns:
        a tensor represents the origin, not padded trajectory or output of the network.
            :shape: `[num_envs(mini-batch_size), sequence_len, ...]`
            :type: `torch.Tensor`
    """
    # [num_trajectories, max_time_steps, ...] ->
    # [true_time_step * num_envs(mini-batch_size), ...] ->
    # [num_envs(mini-batch_size), true_time_step, ...] ->
    return trajectories[masks].reshape(-1, sequence_len, trajectories.shape[-1])
