import dataclasses
from typing import Any, Tuple

import torch

from thunder.core import Batch, ExecutionContext, Operation


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


class SplitTrajOp(Operation):
    def __init__(self, name="split_traj", interval=1, condition=None):
        super().__init__(name, interval, condition)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        traj_lengths: torch.Tensor = get_trajectory_lengths(batch.dones)
        num_trajs = traj_lengths.shape[0]
        max_len = int(traj_lengths.max().item())
        row_indices = torch.arange(max_len, device=self.device).unsqueeze(0)
        scatter_mask = traj_lengths.unsqueeze(1) > row_indices

        def transform_tensor_core(
            tensor: torch.Tensor, target_idxs: torch.Tensor = None
        ) -> torch.Tensor:
            """ """
            flat_tensor = tensor.flatten(0, 1)
            embedding_shape = flat_tensor.shape[1:]
            # Allocate [Num_Trajs, Max_Len, ...]
            pooled = torch.zeros(
                (num_trajs, max_len, *embedding_shape), dtype=flat_tensor.dtype, device=self.device
            )
            pooled[scatter_mask] = flat_tensor
            if target_idxs is not None:
                # [Target_Batch, Max_Len, F]
                return pooled[target_idxs]
            return pooled

        def map_fn_var(node: Any) -> Any:
            if torch.is_tensor(node):
                return transform_tensor_core(node, target_idxs=None)
            return node

        ctx.batch = ctx.batch.map(map_fn_var)
        ctx.batch.replace(mask=scatter_mask)
        return ctx, {}


class FixShapeSplitTrajOp(Operation):
    def __init__(self, name="fix_shape_split_traj", top: bool = True, interval=1, condition=None):
        super().__init__(name, interval, condition)
        self.top = top

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.dones
        device = dones.device

        traj_lengths: torch.Tensor = get_trajectory_lengths(batch.dones)
        num_trajs = traj_lengths.shape[0]
        max_len = int(traj_lengths.max().item())
        row_indices = torch.arange(max_len, device=device).unsqueeze(0)
        scatter_mask = traj_lengths.unsqueeze(1) > row_indices

        def transform_tensor_core(
            tensor: torch.Tensor, target_idxs: torch.Tensor = None
        ) -> torch.Tensor:
            """ """
            flat_tensor = tensor.flatten(0, 1)
            embedding_shape = flat_tensor.shape[1:]
            # Allocate [Num_Trajs, Max_Len, ...]
            pooled = torch.zeros(
                (num_trajs, max_len, *embedding_shape), dtype=flat_tensor.dtype, device=device
            )
            pooled[scatter_mask] = flat_tensor
            if target_idxs is not None:
                # [Target_Batch, Max_Len, ...]
                return pooled[target_idxs]
            return pooled

        target_batch_size = dones.shape[0]
        chunk_len = dones.shape[1]
        pool_size = num_trajs
        if pool_size == target_batch_size:
            idxs = torch.randperm(target_batch_size, device=device)
        else:
            if self.top:
                _, idxs = torch.topk(traj_lengths, k=target_batch_size, largest=True)
                idxs = idxs[torch.randperm(target_batch_size, device=device)]
            else:
                idxs = torch.randperm(pool_size, device=device)[:target_batch_size]

        def transform_fixed(tensor: torch.Tensor) -> torch.Tensor:
            selected = transform_tensor_core(tensor, target_idxs=idxs)
            curr_len = selected.shape[1]
            pad_size = chunk_len - curr_len
            if pad_size > 0:
                shape = selected.shape
                shape[1] = pad_size
                padding = torch.zeros(shape, dtype=selected.dtype, device=device)
                selected = torch.cat([selected, padding], dim=1)
            return selected

        def map_fn_fixed(node: Any) -> Any:
            if torch.is_tensor(node):
                return transform_fixed(node)
            return node

        ctx.batch = ctx.batch.map(map_fn_fixed)
        mask_subset = scatter_mask[idxs]
        curr_mask_len = mask_subset.shape[1]
        mask_pad = chunk_len - curr_mask_len
        if mask_pad > 0:
            padding = torch.zeros((target_batch_size, mask_pad), dtype=torch.bool, device=device)
            mask_subset = torch.cat([mask_subset, padding], dim=1)
        ctx.batch = ctx.batch.replace(mask=mask_subset)

        return ctx, {}
