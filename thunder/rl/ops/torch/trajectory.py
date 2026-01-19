import dataclasses
from typing import Any, Tuple

import torch

from thunder.core import Batch, ExecutionContext, Operation
from thunder.rl.func.torch import get_trajectory_lengths


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
